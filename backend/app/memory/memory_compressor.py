from app.utils.embeddings import compute_cosine_similarity
from app.memory.utility_estimator import estimate_utility
from typing import Dict

def compress_memory(memory, reasoning_state: dict, max_items: int) -> Dict:
    active_items = [i for i in memory.items if i.status in ["ACTIVE", "VERIFIED"]]
    
    metrics = {
        "before_compression": len(active_items),
        "after_compression": 0,
        "duplicates_removed": 0,
        "claims_merged": 0,
        "conflicts_preserved": 0,
        "low_utility_removed": 0,
        "compression_ratio": "0%"
    }
    
    # 1. Semantic Deduplication & Merging Compatible Evidence
    unique_items = []
    for item in active_items:
        is_duplicate = False
        for unique in unique_items:
            # Bidirectional conflict check
            is_conflict = (item.id in unique.in_conflict_with or unique.id in item.in_conflict_with)
            if is_conflict:
                continue
                
            if item.embedding and unique.embedding:
                sim = compute_cosine_similarity(item.embedding, unique.embedding)
                if sim >= 0.90:  # Compatible threshold
                    is_duplicate = True
                    metrics["duplicates_removed"] += 1
                    item.status = "DISCARDED"
                    
                    # True merge logic
                    if len(item.claim) > len(unique.claim):
                        unique.claim = item.claim
                        
                    for chunk_id in item.source_chunk_ids:
                        if chunk_id not in unique.source_chunk_ids:
                            unique.source_chunk_ids.append(chunk_id)
                            
                    unique.evidence_score = max(unique.evidence_score, item.evidence_score)
                    unique.relevance_score = max(unique.relevance_score, item.relevance_score)
                    unique.novelty_score = max(unique.novelty_score, item.novelty_score)
                    unique.in_conflict_with = list(set(unique.in_conflict_with + item.in_conflict_with))
                    
                    metrics["claims_merged"] += 1
                    break
                    
        if not is_duplicate:
            unique_items.append(item)
            
    # Always run utility estimation — skip only if memory is tiny AND no conflicts
    if len(unique_items) <= max(3, max_items // 3) and not any(i.conflict_importance >= 0.7 for i in unique_items):
        final_active = [i for i in memory.items if i.status in ["ACTIVE", "VERIFIED"]]
        metrics["after_compression"] = len(final_active)
        if metrics["before_compression"] > 0:
            ratio = (metrics["before_compression"] - metrics["after_compression"]) / metrics["before_compression"]
            metrics["compression_ratio"] = f"{round(ratio * 100)}%"
        return metrics
            
    # 2. Advanced Utility Estimation
    scored_items = []
    for item in unique_items:
        claim_data = {
            "claim": item.claim,
            "relevance_score": item.relevance_score,
            "evidence_score": item.evidence_score,
            "novelty_score": item.novelty_score,
            "conflict_importance": item.conflict_importance
        }
        
        utility_res = estimate_utility(reasoning_state, claim_data)
        item.utility_score = utility_res.get("utility_score", item.retention_score)
        item.keep = utility_res.get("keep", True)
        
        if item.conflict_importance >= 0.7:
            item.utility_score = max(item.utility_score, 0.9)  # Priority preservation
            metrics["conflicts_preserved"] += 1
            
        scored_items.append(item)
        
    # 3. Discard Low Utility & Enforce max_items
    scored_items.sort(key=lambda x: x.utility_score, reverse=True)
    
    important_conflicts = [i for i in scored_items if i.conflict_importance >= 0.7]
    normal_items = [i for i in scored_items if i.conflict_importance < 0.7]
    
    items_to_keep = []
    
    for item in important_conflicts:
        if len(items_to_keep) < max_items:
            items_to_keep.append(item)
        else:
            item.status = "DISCARDED"
            metrics["low_utility_removed"] += 1
            
    for item in normal_items:
        if len(items_to_keep) < max_items:
            if not getattr(item, 'keep', True):
                item.status = "DISCARDED"
                metrics["low_utility_removed"] += 1
            else:
                items_to_keep.append(item)
        else:
            item.status = "DISCARDED"
            metrics["low_utility_removed"] += 1
            
    final_active = [i for i in memory.items if i.status in ["ACTIVE", "VERIFIED"]]
    metrics["after_compression"] = len(final_active)
    
    if metrics["before_compression"] > 0:
        ratio = (metrics["before_compression"] - metrics["after_compression"]) / metrics["before_compression"]
        metrics["compression_ratio"] = f"{round(ratio * 100)}%"
        
    return metrics
