"""
State-Conditioned Dynamic Weight Scheduling Engine (Method 1)
Implements dynamic adaptation of memory retention parameters based on reasoning state.
Formulation for ACL/EMNLP research:
    W_t = f(step t, horizon T, memory load M_load, conflict ratio rho_conflict)
"""

import math
from typing import Dict, Any, Optional
from app.core.config import MemoryWeightsConfig, get_config


class DynamicMemoryWeightsScheduler:
    """
    Computes state-conditioned retention weights W(S_t) across reasoning hops.
    Replaces static heuristics with research-grade temporal and contextual adaptation.
    """
    def __init__(self, base_weights: Optional[MemoryWeightsConfig] = None):
        self.base_weights = base_weights or get_config().memory.weights

    def compute_scheduled_weights(
        self,
        current_hop: int,
        total_hops: int = 4,
        active_items_count: int = 1,
        max_items: int = 6,
        conflict_count: int = 0
    ) -> MemoryWeightsConfig:
        """
        Dynamically derives W_t = [alpha_t, beta_t, gamma_t, delta_t, epsilon_t, zeta_t, eta_t, theta_t, lambda_t]
        
        Parameters:
            current_hop: Current reasoning step t (1-indexed)
            total_hops: Maximum reasoning horizon T
            active_items_count: Number of currently active items |M_t|
            max_items: Memory capacity ceiling M_max
            conflict_count: Number of detected contradictory claims in working memory
        """
        t = max(1, current_hop)
        T = max(t, total_hops)
        progress = min(1.0, t / T)                     # p in [0.0, 1.0]
        remaining = max(0.0, (T - t) / T)              # r in [0.0, 1.0]

        # Memory load ratio: |M_t| / M_max
        m_load = min(2.0, max(0.0, active_items_count / max(1, max_items)))

        # Contradiction density: C_t / |M_t|
        rho_conflict = min(1.0, max(0.0, conflict_count / max(1, active_items_count)))

        base = self.base_weights

        # 1. Relevance (alpha_t): High during exploratory phase, stabilizes near end
        # alpha_t = alpha_0 * (1 + 0.5 * (1 - progress))
        alpha = base.alpha_relevance * (1.0 + 0.5 * (1.0 - progress))

        # 2. Reliability (beta_t): Invariant source truthfulness prior
        beta = base.beta_reliability

        # 3. Novelty (gamma_t): Monotonically decays as reasoning concludes to avoid late distractor injection
        # gamma_t = gamma_0 * (1 - 0.75 * progress)
        gamma = base.gamma_novelty * max(0.1, 1.0 - 0.75 * progress)

        # 4. Future Utility (delta_t): Critical for intermediate hops; drops to 0 at terminal hop t == T
        # delta_t = delta_0 * (T - t) / T
        delta = base.delta_future_utility * remaining

        # 5. Evidence Strength (epsilon_t): Direct factual evidence importance increases toward answer generation
        # epsilon_t = epsilon_0 * (1 + 0.3 * progress)
        epsilon = base.epsilon_evidence_strength * (1.0 + 0.3 * progress)

        # 6. Verification (zeta_t): Verification status matters most when crystallizing the final reasoning chain
        # zeta_t = zeta_0 * (1 + 0.8 * progress)
        zeta = base.zeta_verification * (1.0 + 0.8 * progress)

        # 7. Redundancy Penalty (eta_t): Penalizes near-duplicates more aggressively when memory is saturated
        # eta_t = eta_0 * (1 + 0.5 * m_load)
        eta = base.eta_redundancy_penalty * (1.0 + 0.5 * m_load)

        # 8. Contradiction Penalty (theta_t): Scales aggressively when conflicting assertions are detected
        # theta_t = theta_0 * (1 + 1.5 * rho_conflict)
        theta = base.theta_conflict_penalty * (1.0 + 1.5 * rho_conflict)

        # 9. Age Cost Decay (lambda_t): Eviction accelerates exponentially under memory pressure
        # lambda_t = lambda_0 * exp(0.5 * m_load)
        lam = base.lambda_age_cost * math.exp(0.5 * m_load)

        return MemoryWeightsConfig(
            alpha_relevance=round(alpha, 4),
            beta_reliability=round(beta, 4),
            gamma_novelty=round(gamma, 4),
            delta_future_utility=round(delta, 4),
            epsilon_evidence_strength=round(epsilon, 4),
            zeta_verification=round(zeta, 4),
            eta_redundancy_penalty=round(eta, 4),
            theta_conflict_penalty=round(theta, 4),
            lambda_age_cost=round(lam, 4)
        )

    def explain_weight_schedule(
        self,
        current_hop: int,
        total_hops: int = 4,
        active_items_count: int = 1,
        max_items: int = 6,
        conflict_count: int = 0
    ) -> Dict[str, Any]:
        """Provides human-interpretable rationale for the dynamic weight modulation."""
        w = self.compute_scheduled_weights(
            current_hop=current_hop,
            total_hops=total_hops,
            active_items_count=active_items_count,
            max_items=max_items,
            conflict_count=conflict_count
        )
        t = current_hop
        T = total_hops
        return {
            "hop": current_hop,
            "total_hops": total_hops,
            "phase": "exploration" if t == 1 else ("synthesis" if t >= T else "intermediate_reasoning"),
            "weights": w.model_dump() if hasattr(w, "model_dump") else w.dict(),
            "dynamics": {
                "novelty_decay_pct": round((1.0 - (w.gamma_novelty / self.base_weights.gamma_novelty)) * 100, 1),
                "future_utility_active": w.delta_future_utility > 0.0,
                "verification_boost_pct": round(((w.zeta_verification / self.base_weights.zeta_verification) - 1.0) * 100, 1),
                "memory_saturation_ratio": round(active_items_count / max(1, max_items), 2),
                "conflict_escalation_pct": round(((w.theta_conflict_penalty / self.base_weights.theta_conflict_penalty) - 1.0) * 100, 1)
            }
        }


# Global singleton instance
scheduler = DynamicMemoryWeightsScheduler()

def get_scheduled_weights(
    current_hop: int,
    total_hops: int = 4,
    active_items_count: int = 1,
    max_items: int = 6,
    conflict_count: int = 0,
    base_weights: Optional[MemoryWeightsConfig] = None
) -> MemoryWeightsConfig:
    """Helper shortcut function to obtain scheduled weights."""
    if base_weights:
        custom_scheduler = DynamicMemoryWeightsScheduler(base_weights=base_weights)
        return custom_scheduler.compute_scheduled_weights(
            current_hop=current_hop,
            total_hops=total_hops,
            active_items_count=active_items_count,
            max_items=max_items,
            conflict_count=conflict_count
        )
    return scheduler.compute_scheduled_weights(
        current_hop=current_hop,
        total_hops=total_hops,
        active_items_count=active_items_count,
        max_items=max_items,
        conflict_count=conflict_count
    )
