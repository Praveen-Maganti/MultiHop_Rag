"""
Central Adaptive Memory Manager
Acts as the central cognitive controller between retrieval, reasoning, and reflection.
Implements the 11 core dynamic memory operations defined in Phase 3, 14, and 15.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import uuid

from app.core.config import get_config, MemoryWeightsConfig
from app.core.state import (
    MemoryItem,
    MemoryStatus,
    MemoryOperation,
    MemoryUpdate,
    Provenance,
    Evidence
)
from app.memory.scoring import calculate_memory_retention_score, explain_scoring_decision
from app.utils.embeddings import get_cached_embedding, compute_cosine_similarity
from app.memory.evidence_scorer import score_evidence
from app.memory.conflict_detector import detect_conflict


class AdaptiveMemoryManager:
    """
    Dynamic Controller managing the lifecycle, curation, and consolidation
    of working memory across multi-hop reasoning steps.
    """
    def __init__(self, max_items: int = None):
        self.config = get_config()
        self.max_items = max_items or self.config.memory.max_items
        self.items: List[MemoryItem] = []
        self.history_log: List[MemoryUpdate] = []
        self.total_duplicates_merged: int = 0
        self.total_evicted_count: int = 0

    def get_active_items(self) -> List[MemoryItem]:
        return [i for i in self.items if i.status in [MemoryStatus.ACTIVE, MemoryStatus.VERIFIED]]

    def _score_item(self, item: MemoryItem, hop: int, weights: Optional[MemoryWeightsConfig] = None) -> float:
        """Helper that scores memory items using state-conditioned dynamic weights."""
        total_hops = getattr(self.config.planning, "max_hops", 4)
        active_count = len(self.get_active_items())
        conflicts = sum(1 for i in self.items if i.contradiction_status == "DETECTED")
        return calculate_memory_retention_score(
            item=item,
            current_hop=hop,
            weights=weights,
            total_hops=total_hops,
            active_items_count=active_count,
            max_items=self.max_items,
            conflict_count=conflicts
        )

    # =========================================================================
    # Core Operation 1: ADD
    # =========================================================================
    def add_item(
        self,
        content: str,
        source_chunk_ids: List[str],
        source_text: str,
        query: str,
        hop: int,
        relevance: float = 0.8,
        evidence_strength: float = 0.8,
        reliability: float = 0.8,
        provenance_list: List[Provenance] = None
    ) -> MemoryItem:
        emb = get_cached_embedding(content)
        item = MemoryItem(
            content=content,
            source_chunk_ids=source_chunk_ids,
            source_text=source_text,
            hop=hop,
            confidence=round((relevance + evidence_strength) / 2.0, 3),
            relevance_score=relevance,
            evidence_strength=evidence_strength,
            reliability=reliability,
            provenance=provenance_list or [],
            embedding=emb,
            status=MemoryStatus.ACTIVE
        )
        # Calculate initial score
        self._score_item(item, hop=hop)
        self.items.append(item)
        
        self.history_log.append(MemoryUpdate(
            operation=MemoryOperation.ADD,
            target_item_id=item.id,
            reason=f"Added during Hop {hop} for query: {query}",
            score_after=item.retention_score
        ))
        return item

    # =========================================================================
    # Core Operation 2: UPDATE
    # =========================================================================
    def update_item(self, item_id: str, new_content: str = None, confidence_delta: float = 0.0, hop: int = 1) -> Optional[MemoryItem]:
        for item in self.items:
            if item.id == item_id:
                if new_content:
                    item.content = new_content
                    item.embedding = get_cached_embedding(new_content)
                item.confidence = max(0.0, min(1.0, item.confidence + confidence_delta))
                item.last_accessed_hop = hop
                item.access_count += 1
                item.updated_at = time.time()
                self._score_item(item, hop=hop)
                
                self.history_log.append(MemoryUpdate(
                    operation=MemoryOperation.UPDATE,
                    target_item_id=item.id,
                    reason=f"Updated confidence/content at Hop {hop}",
                    score_after=item.retention_score
                ))
                return item
        return None

    # =========================================================================
    # Core Operation 3: MERGE
    # =========================================================================
    def merge_items(self, primary_item: MemoryItem, duplicate_item: MemoryItem, hop: int) -> MemoryItem:
        # Combine chunk IDs and provenance
        for cid in duplicate_item.source_chunk_ids:
            if cid not in primary_item.source_chunk_ids:
                primary_item.source_chunk_ids.append(cid)
        for p in duplicate_item.provenance:
            primary_item.provenance.append(p)

        # Longer or more detailed content wins
        if len(duplicate_item.content) > len(primary_item.content):
            primary_item.content = duplicate_item.content
            primary_item.embedding = duplicate_item.embedding

        # Maximize scores
        primary_item.evidence_strength = max(primary_item.evidence_strength, duplicate_item.evidence_strength)
        primary_item.relevance_score = max(primary_item.relevance_score, duplicate_item.relevance_score)
        primary_item.confidence = max(primary_item.confidence, duplicate_item.confidence)
        primary_item.last_accessed_hop = hop
        
        duplicate_item.status = MemoryStatus.DISCARDED
        duplicate_item.decision_reason = f"Merged into {primary_item.id}"
        
        self._score_item(primary_item, hop=hop)
        
        self.history_log.append(MemoryUpdate(
            operation=MemoryOperation.MERGE,
            target_item_id=primary_item.id,
            reason=f"Merged duplicate {duplicate_item.id} into {primary_item.id}",
            score_after=primary_item.retention_score
        ))
        return primary_item

    # =========================================================================
    # Core Operation 4: DELETE / DISCARD
    # =========================================================================
    def delete_item(self, item_id: str, reason: str = "Low utility eviction"):
        for item in self.items:
            if item.id == item_id:
                item.status = MemoryStatus.DISCARDED
                item.decision_reason = reason
                self.history_log.append(MemoryUpdate(
                    operation=MemoryOperation.DELETE,
                    target_item_id=item.id,
                    reason=reason,
                    score_before=item.retention_score
                ))

    # =========================================================================
    # Core Operation 7: VERIFY
    # =========================================================================
    def verify_item(self, item_id: str, is_valid: bool, hop: int, feedback: str = ""):
        for item in self.items:
            if item.id == item_id:
                if is_valid:
                    item.verification_status = "VERIFIED"
                    item.status = MemoryStatus.VERIFIED
                    item.confidence = min(1.0, item.confidence + 0.15)
                else:
                    item.verification_status = "REFUTED"
                    item.status = MemoryStatus.REJECTED
                    item.confidence = max(0.0, item.confidence - 0.40)
                item.last_accessed_hop = hop
                self._score_item(item, hop=hop)
                self.history_log.append(MemoryUpdate(
                    operation=MemoryOperation.VERIFY,
                    target_item_id=item.id,
                    reason=f"Verification: {'PASS' if is_valid else 'FAIL'} - {feedback}",
                    score_after=item.retention_score
                ))

    # =========================================================================
    # Core Operations 8 & 9: CONTRADICTION DETECTION & RESOLUTION
    # =========================================================================
    def handle_contradiction(self, item_a: MemoryItem, item_b: MemoryItem, hop: int):
        item_a.contradiction_status = "DETECTED"
        item_b.contradiction_status = "DETECTED"
        if item_b.id not in item_a.conflicting_item_ids:
            item_a.conflicting_item_ids.append(item_b.id)
        if item_a.id not in item_b.conflicting_item_ids:
            item_b.conflicting_item_ids.append(item_a.id)

        self.history_log.append(MemoryUpdate(
            operation=MemoryOperation.MARK_CONTRADICTION,
            target_item_id=item_a.id,
            reason=f"Contradiction detected with {item_b.id}"
        ))

        # Resolution heuristic: item with higher evidence strength / reliability survives
        if abs(item_a.evidence_strength - item_b.evidence_strength) > 0.2:
            winner = item_a if item_a.evidence_strength > item_b.evidence_strength else item_b
            loser = item_b if winner == item_a else item_a
            loser.status = MemoryStatus.CONTRADICTED
            winner.status = MemoryStatus.ACTIVE
            winner.contradiction_status = "RESOLVED"
            self.history_log.append(MemoryUpdate(
                operation=MemoryOperation.RESOLVE_CONTRADICTION,
                target_item_id=winner.id,
                reason=f"Resolved contradiction: {winner.id} kept over {loser.id} due to higher evidence strength"
            ))

    # =========================================================================
    # Core Operations 10 & 11: CONSOLIDATION & DEDUPLICATION PIPELINE
    # =========================================================================
    def consolidate_memory(self, current_hop: int, main_question: str) -> Dict[str, Any]:
        """
        Executes full consolidation:
        1. Semantic Deduplication & Merging
        2. Pairwise Contradiction Check
        3. Multi-objective Re-scoring with Age Decay
        4. Strict pruning to max_items
        """
        active_items = self.get_active_items()
        duplicates_merged = 0
        conflicts_checked = 0
        evicted_count = 0

        # 1. Semantic Deduplication & Merging
        deduped: List[MemoryItem] = []
        for item in active_items:
            is_dup = False
            item_emb = item.embedding or get_cached_embedding(item.content)
            item.embedding = item_emb

            for existing in deduped:
                exist_emb = existing.embedding or get_cached_embedding(existing.content)
                existing.embedding = exist_emb
                sim = compute_cosine_similarity(item_emb, exist_emb)
                
                if sim >= self.config.memory.redundancy_threshold:
                    self.merge_items(existing, item, current_hop)
                    duplicates_merged += 1
                    is_dup = True
                    break
                elif sim >= self.config.memory.conflict_similarity_threshold:
                    # Potential conflict candidate
                    conflicts_checked += 1
                    c_res = detect_conflict(item.content, existing.content)
                    if c_res.get("conflict_score", 0.0) >= 0.6:
                        self.handle_contradiction(item, existing, current_hop)

            if not is_dup and item.status in [MemoryStatus.ACTIVE, MemoryStatus.VERIFIED]:
                deduped.append(item)

        # 2. Re-score all items
        for item in deduped:
            self._score_item(item, hop=current_hop)

        # 3. Sort by Retention Score descending
        deduped.sort(key=lambda x: x.retention_score, reverse=True)

        # 4. Enforce max_items limit
        final_kept: List[MemoryItem] = []
        for idx, item in enumerate(deduped):
            if idx < self.max_items and item.retention_score >= self.config.memory.retention_threshold:
                final_kept.append(item)
            else:
                self.delete_item(item.id, reason="Pruned during memory compaction / capacity limit")
                evicted_count += 1

        self.total_duplicates_merged += duplicates_merged
        self.total_evicted_count += evicted_count

        return {
            "active_before": len(active_items),
            "active_after": len(final_kept),
            "duplicates_merged": duplicates_merged,
            "conflicts_checked": conflicts_checked,
            "evicted_count": evicted_count
        }

    # =========================================================================
    # Intake New Evidence Chunks
    # =========================================================================
    def process_incoming_evidence(
        self,
        evidence_chunks: List[Dict[str, Any]],
        subquestion: str,
        hop: int,
        source_doc_id: str = "doc"
    ) -> List[MemoryItem]:
        """
        Extracts, scores, and registers incoming evidence into memory.
        """
        new_items = []
        for chunk in evidence_chunks:
            text = chunk.get("text", "").strip()
            cid = chunk.get("chunk_id", f"chk_{uuid.uuid4().hex[:6]}")
            if not text:
                continue

            # Check if this exact chunk was already ingested
            if any(cid in item.source_chunk_ids for item in self.get_active_items()):
                continue

            # Score evidence chunk
            scores = score_evidence(subquestion, text)
            rel = float(scores.get("relevance", 0.5))
            sup = float(scores.get("support", 0.5))
            relb = float(scores.get("reliability", 0.8))

            if rel < 0.35 and sup < 0.35:
                continue  # Skip irrelevant incoming chunk

            prov = Provenance(
                source_document=source_doc_id,
                retrieval_query=subquestion,
                retrieval_rank=chunk.get("rank", 1),
                hop_number=hop,
                chunk_id=cid,
                retriever_score=chunk.get("score", 0.0),
                reranker_score=chunk.get("rerank_score", 0.0)
            )

            # Extract fact statement or use clean snippet
            fact_claim = text[:280] if len(text) > 280 else text
            item = self.add_item(
                content=fact_claim,
                source_chunk_ids=[cid],
                source_text=text,
                query=subquestion,
                hop=hop,
                relevance=rel,
                evidence_strength=sup,
                reliability=relb,
                provenance_list=[prov]
            )
            new_items.append(item)

        # Run consolidation immediately if auto_consolidate is enabled
        if self.config.memory.auto_consolidate:
            self.consolidate_memory(current_hop=hop, main_question=subquestion)

        return new_items

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": i.id,
                "claim": i.content,
                "text": i.source_text,
                "status": i.status.value if hasattr(i.status, "value") else str(i.status),
                "relevance_score": i.relevance_score,
                "evidence_score": i.evidence_strength,
                "reliability": i.reliability,
                "novelty_score": i.novelty_score,
                "utility_score": i.retention_score,
                "retention_score": i.retention_score,
                "hop_added": i.hop,
                "source_chunk_ids": i.source_chunk_ids,
                "provenance": [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in i.provenance],
                "decision_reason": explain_scoring_decision(i, current_hop=i.hop, total_hops=getattr(self.config.planning, "max_hops", 4))
            }
            for i in self.items
        ]
