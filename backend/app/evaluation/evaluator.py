from app.evaluation.metrics import (
    calculate_token_f1,
    compute_f1,
    compute_em,
    compute_bertscore,
    match_claims_to_evidence
)
from app.evaluation.claim_extractor import extract_claims_rule_based

def evaluate_result(rag_result: dict, ground_truth: str = None) -> dict:
    """
    Evaluates a generated RAG result using objective primary metrics (F1, EM, BERTScore)
    and evidence coverage, keeping dual reward as a secondary metric.
    """
    answer = rag_result.get("answer", "")
    memory_items = rag_result.get("memory", [])
    dual_reward = rag_result.get("dual_reward", {})
    metrics = rag_result.get("metrics", {})
    
    cumulative_reward = dual_reward.get("cumulative_reward")
    if cumulative_reward is None:
        cumulative_reward = metrics.get("cumulative_dual_reward", 0.0)

    evaluation = {
        # Primary Metrics (Objective, standard, reproducible)
        "f1": 0.0,
        "exact_match": 0.0,
        "bertscore": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        # Secondary Metric (Novel multi-hop reasoning quality)
        "dual_reward": dual_reward,
        "cumulative_dual_reward": cumulative_reward,
        # Groundedness & System Efficiency
        "evidence_coverage": 0.0,
        "unsupported_claims": 0,
        "unsupported_claim_ratio": 0.0,
        "total_answer_claims": 0,
        "supported_claims": 0,
        "hops": metrics.get("hops", 0),
        "retrieval_calls": metrics.get("retrieval_calls", 0),
        "latency_seconds": metrics.get("latency_seconds", 0.0)
    }
    
    # 1. Primary Metrics Calculation (requires ground truth)
    if ground_truth:
        evaluation["f1"] = compute_f1(answer, ground_truth)
        evaluation["exact_match"] = compute_em(answer, ground_truth)
        evaluation["bertscore"] = compute_bertscore(answer, ground_truth)
        f1_metrics = calculate_token_f1(answer, ground_truth)
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

