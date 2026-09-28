from app.memory.working_memory import WorkingMemory, MemoryItem
from app.memory.evidence_scorer import score_evidence
from app.memory.conflict_detector import detect_conflict
from app.memory.claim_extractor import extract_claims
from app.utils.embeddings import get_cached_embedding, compute_cosine_similarity
from typing import List, Dict
import os

MEMORY_REDUNDANCY_THRESHOLD = 0.85
CONFLICT_SIMILARITY_THRESHOLD = 0.75

def process_chunk(chunk: Dict, question: str, current_memory: WorkingMemory, current_hop: int, metrics: dict) -> List[MemoryItem]:
    claims = extract_claims(chunk["text"])
    
    chunk_embedding = chunk.get("embedding")
    if not chunk_embedding:
        chunk_embedding = get_cached_embedding(chunk["text"])
    
    items = []
    for claim in claims:
        metrics["total_claims"] += 1
        
        claim_embedding = get_cached_embedding(claim)
        
        query_embedding = get_cached_embedding(question)
        sim_to_query = compute_cosine_similarity(query_embedding, claim_embedding)
        
        RETRIEVAL_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.30"))
        if sim_to_query < RETRIEVAL_SIMILARITY_THRESHOLD:
            continue
            
        item = MemoryItem(
            source_chunk_ids=[chunk["chunk_id"]],
            claim=claim,
            text=chunk["text"],
            source_query=question,
            hop_added=current_hop,
            embedding=claim_embedding
        )
        
        metrics["evidence_llm_calls"] += 1
        scores = score_evidence(question, item.claim)
        item.relevance_score = scores.get("relevance", 0.0)
        item.evidence_score = scores.get("support", 0.0)
        item.reliability = scores.get("reliability", 0.0)
        
        max_redundancy = 0.0
        for existing_item in current_memory.items:
            if existing_item.embedding and claim_embedding:
                sim = compute_cosine_similarity(claim_embedding, existing_item.embedding)
                if sim > max_redundancy:
                    max_redundancy = sim
                    
        item.redundancy_score = max_redundancy
        item.novelty_score = max(0.0, 1.0 - max_redundancy)
        
        max_conflict_score = 0.0
        max_conflict_importance = 0.0
        for existing_item in current_memory.items:
            metrics["possible_conflict_pairs"] += 1
            
            sim = 0.0
            if existing_item.embedding and claim_embedding:
                sim = compute_cosine_similarity(claim_embedding, existing_item.embedding)
            
            # Semantic filter before LLM
            if sim >= CONFLICT_SIMILARITY_THRESHOLD:
                metrics["candidate_conflict_pairs"] += 1
                metrics["conflict_llm_calls"] += 1
                
                conflict = detect_conflict(item.claim, existing_item.claim)
                c_score = conflict.get("conflict_score", 0.0)
                c_imp = conflict.get("conflict_importance", 0.0)
                
                if c_score > 0.0:
                    item.in_conflict_with.append(existing_item.id)
                    existing_item.in_conflict_with.append(item.id)
                    
                    if c_score > max_conflict_score:
                        max_conflict_score = c_score
                    if c_imp > max_conflict_importance:
                        max_conflict_importance = c_imp
            else:
                metrics["conflict_calls_saved"] += 1
                        
        item.conflict_score = max_conflict_score
        item.conflict_importance = max_conflict_importance
        
        item.calculate_retention_score(current_hop)
        items.append(item)
        
    return items

def select_evidence(chunks: List[Dict], question: str, memory: WorkingMemory, current_hop: int = 1, retention_threshold: float = 0.5, metrics: dict = None):
    if metrics is None:
        metrics = {
            "evidence_llm_calls": 0,
            "total_claims": 0,
            "possible_conflict_pairs": 0,
            "candidate_conflict_pairs": 0,
            "conflict_llm_calls": 0,
            "llm_calls_saved": 0
        }
        
    for chunk in chunks:
        if any(chunk["chunk_id"] in item.source_chunk_ids for item in memory.items):
            continue
            
        memory_items = process_chunk(chunk, question, memory, current_hop, metrics)
        
        for item in memory_items:
            if item.retention_score >= retention_threshold or item.conflict_importance > 0.5:
                memory.add_item(item)
