"""
Adaptive Memory Scoring Engine
Implements parameterized multi-objective memory retention scoring based on Phase 4.
"""

from app.core.config import get_config, MemoryWeightsConfig
from app.core.state import MemoryItem, MemoryStatus
from app.memory.dynamic_weights import get_scheduled_weights
from typing import Dict, Any, Optional


def calculate_memory_retention_score(
    item: MemoryItem,
    current_hop: int,
    weights: MemoryWeightsConfig = None,
    total_hops: int = 4,
    active_items_count: int = 1,
    max_items: int = 6,
    conflict_count: int = 0
) -> float:
    """
    Computes the multi-objective retention score:
    Score = alpha * relevance
          + beta * reliability
          + gamma * novelty
          + delta * future_utility
          + epsilon * evidence_strength
          + zeta * verification_confidence
          - eta * redundancy
          - theta * contradiction_risk
          - lambda * age_decay
    """
    cfg = get_config()
    if weights is None:
        if getattr(cfg.memory, "dynamic_weights", True):
            weights = get_scheduled_weights(
                current_hop=current_hop,
                total_hops=total_hops,
                active_items_count=active_items_count,
                max_items=max_items,
                conflict_count=conflict_count,
                base_weights=cfg.memory.weights
            )
        else:
            weights = cfg.memory.weights

    # Verification boost or penalty
    verification_factor = 1.0 if item.verification_status == "VERIFIED" else (0.5 if item.verification_status == "UNVERIFIED" else -1.0)
    conflict_risk = 1.0 if item.contradiction_status == "DETECTED" else 0.0
    age_decay = max(0, current_hop - item.last_accessed_hop) * weights.lambda_age_cost

    positive_score = (
        weights.alpha_relevance * item.relevance_score +
        weights.beta_reliability * item.reliability +
        weights.gamma_novelty * item.novelty_score +
        weights.delta_future_utility * item.future_utility_score +
        weights.epsilon_evidence_strength * item.evidence_strength +
        weights.zeta_verification * (item.confidence * verification_factor)
    )

    penalty = (
        weights.eta_redundancy_penalty * item.redundancy_score +
        weights.theta_conflict_penalty * conflict_risk +
        age_decay
    )

    score = max(0.0, positive_score - penalty)
    item.retention_score = round(score, 4)
    item.importance_score = round(score, 4)
    return item.retention_score


def explain_scoring_decision(item: MemoryItem, current_hop: int, total_hops: int = 4) -> str:
    """Returns a transparent human-readable explanation for the memory score."""
    reasons = []
    if item.relevance_score >= 0.7:
        reasons.append(f"high relevance ({item.relevance_score:.2f})")
    if item.novelty_score >= 0.8:
        reasons.append(f"high novelty ({item.novelty_score:.2f})")
    if item.redundancy_score >= 0.7:
        reasons.append(f"penalized for redundancy ({item.redundancy_score:.2f})")
    if item.contradiction_status == "DETECTED":
        reasons.append("penalized for active contradiction")
    if item.verification_status == "VERIFIED":
        reasons.append("boosted by factual verification")
    age = current_hop - item.last_accessed_hop
    if age > 1:
        reasons.append(f"decayed by {age} hops")
    if current_hop >= total_hops:
        reasons.append("terminal hop: future utility zeroed")
    return "; ".join(reasons) if reasons else f"base score: {item.retention_score:.2f}"
