"""
RL-Compatible Environment and Reward Interface for Adaptive State-Aware RAG
Provides Gym-style transition (S_i, a_i, r_i, S_{i+1}) and multi-objective reward calculation.
"""

from typing import Dict, Any, List, Tuple, Optional
from app.core.config import get_config, RewardWeightsConfig
from app.core.state import ReasoningState, ActionType, MemoryOperation


class RLEnvironmentInterface:
    """
    Standardizes state-action-reward step transitions for RL policy optimization.
    """
    def __init__(self, weights: Optional[RewardWeightsConfig] = None):
        self.weights = weights or get_config().reward_weights

    def compute_reward(
        self,
        is_terminal: bool,
        answer_correctness: float,
        evidence_quality: float,
        memory_quality: float,
        retrieval_calls: int,
        reasoning_steps: int,
        unnecessary_retrieval_count: int = 0,
        memory_redundancy: float = 0.0
    ) -> Dict[str, float]:
        """
        Computes the composite RL reward:
        R = w1 * correctness + w2 * evidence_qual + w3 * memory_qual
          + w4 * retrieval_eff + w5 * reasoning_eff
          - w6 * unnecessary_retrievals - w7 * redundancy
        """
        w = self.weights

        retrieval_eff = max(0.0, 1.0 - (retrieval_calls * 0.15))
        reasoning_eff = max(0.0, 1.0 - (reasoning_steps * 0.15))

        reward = (
            (w.w1_answer_correctness * answer_correctness if is_terminal else 0.0) +
            w.w2_evidence_quality * evidence_quality +
            w.w3_memory_quality * memory_quality +
            w.w4_retrieval_efficiency * retrieval_eff +
            w.w5_reasoning_efficiency * reasoning_eff -
            w.w6_unnecessary_retrieval_penalty * (unnecessary_retrieval_count * 0.2) -
            w.w7_memory_redundancy_penalty * memory_redundancy
        )

        return {
            "total_reward": round(reward, 4),
            "correctness_component": round(w.w1_answer_correctness * answer_correctness if is_terminal else 0.0, 4),
            "evidence_quality_component": round(w.w2_evidence_quality * evidence_quality, 4),
            "memory_quality_component": round(w.w3_memory_quality * memory_quality, 4),
            "efficiency_component": round((w.w4_retrieval_efficiency * retrieval_eff) + (w.w5_reasoning_efficiency * reasoning_eff), 4)
        }

    def format_transition_tuple(
        self,
        state_t: ReasoningState,
        action: ActionType,
        reward_dict: Dict[str, float],
        state_t_plus_1: ReasoningState,
        done: bool
    ) -> Dict[str, Any]:
        return {
            "state_t_id": state_t.state_id,
            "step": state_t.reasoning_step,
            "action": action.value if hasattr(action, "value") else str(action),
            "reward": reward_dict.get("total_reward", 0.0),
            "reward_breakdown": reward_dict,
            "next_state_id": state_t_plus_1.state_id,
            "done": done
        }
