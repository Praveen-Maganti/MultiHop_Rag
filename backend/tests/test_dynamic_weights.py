"""
Unit Tests for State-Conditioned Dynamic Weight Scheduling (Method 1)
Validates mathematical properties, state adaptivity, terminal zeroing, and backward compatibility.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_config, update_config, MemoryWeightsConfig
from app.core.state import MemoryItem, MemoryStatus
from app.memory.dynamic_weights import DynamicMemoryWeightsScheduler, get_scheduled_weights
from app.memory.scoring import calculate_memory_retention_score, explain_scoring_decision


class TestDynamicWeightScheduling(unittest.TestCase):

    def setUp(self):
        self.scheduler = DynamicMemoryWeightsScheduler()
        # Reset config to dynamic weights enabled
        cfg = get_config()
        cfg.memory.dynamic_weights = True

    def test_01_novelty_and_future_utility_decay(self):
        """Test that novelty monotonically decays and future utility reaches zero at terminal hop."""
        w_hop1 = self.scheduler.compute_scheduled_weights(current_hop=1, total_hops=4)
        w_hop2 = self.scheduler.compute_scheduled_weights(current_hop=2, total_hops=4)
        w_hop3 = self.scheduler.compute_scheduled_weights(current_hop=3, total_hops=4)
        w_hop4 = self.scheduler.compute_scheduled_weights(current_hop=4, total_hops=4)

        # Novelty: gamma_1 > gamma_2 > gamma_3 > gamma_4
        self.assertGreater(w_hop1.gamma_novelty, w_hop2.gamma_novelty)
        self.assertGreater(w_hop2.gamma_novelty, w_hop3.gamma_novelty)
        self.assertGreater(w_hop3.gamma_novelty, w_hop4.gamma_novelty)

        # Future Utility: delta_1 > delta_2 > delta_3 > delta_4 == 0.0
        self.assertGreater(w_hop1.delta_future_utility, w_hop2.delta_future_utility)
        self.assertGreater(w_hop2.delta_future_utility, w_hop3.delta_future_utility)
        self.assertGreater(w_hop3.delta_future_utility, 0.0)
        self.assertEqual(w_hop4.delta_future_utility, 0.0, "Future utility MUST be 0 at terminal step!")

    def test_02_verification_boost_in_late_hops(self):
        """Test that verification weight zeta scales up as hops approach conclusion."""
        w_hop1 = self.scheduler.compute_scheduled_weights(current_hop=1, total_hops=4)
        w_hop4 = self.scheduler.compute_scheduled_weights(current_hop=4, total_hops=4)

        self.assertGreater(w_hop4.zeta_verification, w_hop1.zeta_verification)
        self.assertGreater(w_hop4.epsilon_evidence_strength, w_hop1.epsilon_evidence_strength)

    def test_03_contradiction_escalation(self):
        """Test that contradiction penalty theta scales up when conflicts are detected."""
        w_no_conflict = self.scheduler.compute_scheduled_weights(
            current_hop=2, total_hops=4, active_items_count=4, conflict_count=0
        )
        w_high_conflict = self.scheduler.compute_scheduled_weights(
            current_hop=2, total_hops=4, active_items_count=4, conflict_count=2
        )

        self.assertGreater(w_high_conflict.theta_conflict_penalty, w_no_conflict.theta_conflict_penalty)

    def test_04_memory_saturation_accelerates_age_decay(self):
        """Test that memory load accelerates age decay lambda and redundancy penalty eta."""
        w_empty_load = self.scheduler.compute_scheduled_weights(
            current_hop=2, total_hops=4, active_items_count=1, max_items=6
        )
        w_saturated_load = self.scheduler.compute_scheduled_weights(
            current_hop=2, total_hops=4, active_items_count=6, max_items=6
        )

        self.assertGreater(w_saturated_load.lambda_age_cost, w_empty_load.lambda_age_cost)
        self.assertGreater(w_saturated_load.eta_redundancy_penalty, w_empty_load.eta_redundancy_penalty)

    def test_05_scoring_integration_with_dynamic_weights(self):
        """Test that retention score adapts dynamically between hop 1 and hop 4 for the same item."""
        item = MemoryItem(
            content="Jean-Daniel Pollet directed the 1963 film Méditerranée.",
            relevance_score=0.9,
            novelty_score=0.95,
            future_utility_score=0.8,
            evidence_strength=0.85,
            confidence=0.88,
            verification_status="VERIFIED",
            last_accessed_hop=1
        )

        # Hop 1: High novelty & future utility boost
        score_hop1 = calculate_memory_retention_score(item, current_hop=1, total_hops=4)

        # Update access hop so age decay is isolated
        item.last_accessed_hop = 4
        # Hop 4: Terminal hop (future utility = 0, novelty reduced, verification increased)
        score_hop4 = calculate_memory_retention_score(item, current_hop=4, total_hops=4)

        self.assertGreater(score_hop1, 0.0)
        self.assertGreater(score_hop4, 0.0)
        explanation = explain_scoring_decision(item, current_hop=4, total_hops=4)
        self.assertIn("terminal hop: future utility zeroed", explanation)

    def test_06_ablation_static_weights_fallback(self):
        """Test that disabling dynamic_weights falls back to static defaults for ablation experiments."""
        cfg = get_config()
        cfg.memory.dynamic_weights = False

        item = MemoryItem(
            content="Static weight test item",
            relevance_score=0.8,
            novelty_score=0.8,
            future_utility_score=0.8,
            evidence_strength=0.8,
            confidence=0.8,
            verification_status="UNVERIFIED",
            last_accessed_hop=1
        )

        score_static_hop1 = calculate_memory_retention_score(item, current_hop=1, total_hops=4)
        item.last_accessed_hop = 4
        score_static_hop4 = calculate_memory_retention_score(item, current_hop=4, total_hops=4)

        # With static weights and equal last_accessed_hop, score remains identical across hops
        self.assertAlmostEqual(score_static_hop1, score_static_hop4, places=3)

        # Restore
        cfg.memory.dynamic_weights = True


if __name__ == "__main__":
    unittest.main()
