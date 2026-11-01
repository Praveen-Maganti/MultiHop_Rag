import re
import sys
from collections import Counter

# Safeguard against incompatible torchvision packages
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None

_BERTSCORE_AVAILABLE = False
try:
    from bert_score import score as _bert_score_func
    _BERTSCORE_AVAILABLE = True
except Exception:
    _bert_score_func = None

def compute_cosine_similarity(vec1: list, vec2: list) -> float:
    try:
        import numpy as np
        v1 = np.array(vec1).astype('float32')
        v2 = np.array(vec2).astype('float32')
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    except Exception:
        return 0.0

def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_f1(predicted: str, ground_truth: str) -> float:
    """Token-level F1 score (standard multiset overlap as used in SQuAD / HotpotQA)."""
    pred_tokens = normalize(predicted).split()
    gt_tokens = normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return round(2 * precision * recall / (precision + recall), 4)

def compute_em(predicted: str, ground_truth: str) -> float:
    """Exact Match (EM) binary score after normalization."""
    return float(normalize(predicted) == normalize(ground_truth))

def compute_bertscore(
    predicted: str,
    ground_truth: str,
    model_type: str = "distilbert-base-uncased"
) -> float:
    """
    Computes BERTScore F1 metric measuring semantic similarity via contextual embeddings.
    Catches paraphrased or synonymous answers that token overlap F1 punishes.
    """
    if not predicted or not ground_truth:
        return 0.0
    if not _BERTSCORE_AVAILABLE or _bert_score_func is None:
        return 0.0
    try:
        _, _, F1 = _bert_score_func(
            [str(predicted)],
            [str(ground_truth)],
            lang="en",
            model_type=model_type,
            verbose=False
        )
        return round(float(F1.item()), 4)
    except Exception as e:
        print(f"Warning: BERTScore computation failed ({e}). Returning 0.0")
        return 0.0

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

def match_claims_to_evidence(answer_claims: list, memory_items: list, threshold: float = 0.55) -> dict:
    supported_claims = 0
    unsupported_claims = 0
    
    for ans_claim in answer_claims:
        if not ans_claim or len(ans_claim.strip()) < 4:
            continue

        is_supported = False
        ans_clean = normalize(ans_claim)
        ans_words = set(w for w in ans_clean.split() if len(w) > 3)

        ans_emb = None
        try:
            from app.utils.embeddings import get_cached_embedding
            ans_emb = get_cached_embedding(ans_claim)
        except Exception:
            ans_emb = None
        
        for item in memory_items:
            # Check against claim, content, and raw context
            mem_claim = item.get("claim") or item.get("content") or ""
            mem_text = item.get("text") or item.get("source_text") or ""
            
            for evidence in [mem_claim, mem_text]:
                if not evidence or not str(evidence).strip():
                    continue

                evidence_str = str(evidence)
                # 1. Lexical / substring overlap check
                ev_clean = normalize(evidence_str)
                if ans_clean in ev_clean or ev_clean in ans_clean:
                    is_supported = True
                    break

                ev_words = set(w for w in ev_clean.split() if len(w) > 3)
                if ans_words and ev_words:
                    overlap = len(ans_words.intersection(ev_words)) / max(1, len(ans_words))
                    if overlap >= 0.45:
                        is_supported = True
                        break

                # 2. Dense semantic similarity check
                if ans_emb is not None:
                    try:
                        mem_emb = get_cached_embedding(evidence_str)
                        sim = compute_cosine_similarity(ans_emb, mem_emb)
                        if sim >= threshold:
                            is_supported = True
                            break
                    except Exception:
                        pass
            
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
