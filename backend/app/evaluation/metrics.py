import re
from app.utils.embeddings import compute_cosine_similarity

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def calculate_token_f1(prediction: str, ground_truth: str) -> dict:
    if not prediction or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    pred_tokens = set(normalize(prediction).split())
    gt_tokens = set(normalize(ground_truth).split())
    
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    common_tokens = pred_tokens.intersection(gt_tokens)
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def match_claims_to_evidence(answer_claims: list, memory_items: list, threshold: float = 0.70) -> dict:
    supported_claims = 0
    unsupported_claims = 0
    
    for ans_claim in answer_claims:
        is_supported = False
        
        # We re-embed for evaluation matching if not cached
        from app.utils.embeddings import get_cached_embedding
        ans_emb = get_cached_embedding(ans_claim)
        
        for item in memory_items:
            # Check against both generated claim and raw evidence text
            mem_claim = item.get("claim", "")
            mem_text = item.get("text", "")
            
            for evidence in [mem_claim, mem_text]:
                if not evidence:
                    continue
                mem_emb = get_cached_embedding(evidence)
                sim = compute_cosine_similarity(ans_emb, mem_emb)
                if sim >= threshold:
                    is_supported = True
                    break
            
            if is_supported:
                break
                
        if is_supported:
            supported_claims += 1
        else:
            unsupported_claims += 1
            
    total = len(answer_claims)
    return {
        "total_claims": total,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "evidence_coverage": round(supported_claims / total, 4) if total > 0 else 0.0,
        "unsupported_ratio": round(unsupported_claims / total, 4) if total > 0 else 0.0
    }
