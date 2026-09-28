from app.evaluation.metrics import calculate_token_f1, match_claims_to_evidence
from app.evaluation.claim_extractor import extract_claims_rule_based

def evaluate_result(rag_result: dict, ground_truth: str = None) -> dict:
    """
    Evaluates a generated RAG result entirely independent of the execution loop.
    """
    answer = rag_result.get("answer", "")
    memory_items = rag_result.get("memory", [])
    
    evaluation = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "evidence_coverage": 0.0,
        "unsupported_claims": 0,
        "unsupported_claim_ratio": 0.0,
        "total_answer_claims": 0,
        "supported_claims": 0
    }
    
    # 1. F1 Calculation (requires ground truth)
    if ground_truth:
        f1_metrics = calculate_token_f1(answer, ground_truth)
        evaluation["f1"] = f1_metrics["f1"]
        evaluation["precision"] = f1_metrics["precision"]
        evaluation["recall"] = f1_metrics["recall"]
        
    # 2. Evidence Coverage & Unsupported Claims
    if answer:
        answer_claims = extract_claims_rule_based(answer)
        if answer_claims and memory_items:
            evidence_metrics = match_claims_to_evidence(answer_claims, memory_items)
            
            evaluation["total_answer_claims"] = evidence_metrics["total_claims"]
            evaluation["supported_claims"] = evidence_metrics["supported_claims"]
            evaluation["unsupported_claims"] = evidence_metrics["unsupported_claims"]
            evaluation["evidence_coverage"] = evidence_metrics["evidence_coverage"]
            evaluation["unsupported_claim_ratio"] = evidence_metrics["unsupported_ratio"]
            
    return evaluation
