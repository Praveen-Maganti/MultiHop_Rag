import unittest
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_config, update_config
from app.core.state import MemoryItem, MemoryStatus, MemoryOperation, Provenance
from app.memory.scoring import calculate_memory_retention_score
from app.memory.adaptive_manager import AdaptiveMemoryManager
from app.memory.global_branch_memory import GlobalBranchMemorySystem
from app.memory.provenance import ProvenanceTracker
from app.retrieval.hybrid_retriever import compute_bm25_score, tokenize, HybridRetriever
from app.evaluation.rl_interface import RLEnvironmentInterface


class TestAdaptiveMemoryManagement(unittest.TestCase):

    def setUp(self):
        self.mgr = AdaptiveMemoryManager(max_items=4)

    def test_01_memory_add(self):
        """Test Memory ADD operation."""
        item = self.mgr.add_item(
            content="Stuart Rosenberg directed the 1970 film Move.",
            source_chunk_ids=["chunk_1"],
            source_text="Move is a 1970 film directed by Stuart Rosenberg.",
            query="Who directed Move (1970)?",
            hop=1,
            relevance=0.95,
            evidence_strength=0.90
        )
        self.assertEqual(len(self.mgr.get_active_items()), 1)
        self.assertEqual(item.status, MemoryStatus.ACTIVE)
        self.assertGreater(item.retention_score, 1.0)

    def test_02_memory_update(self):
        """Test Memory UPDATE operation."""
        item = self.mgr.add_item(
            content="Stuart Rosenberg is from USA.",
            source_chunk_ids=["chunk_2"],
            source_text="Rosenberg was born in Brooklyn, USA.",
            query="Where is Stuart Rosenberg from?",
            hop=1,
            relevance=0.8
        )
        updated = self.mgr.update_item(item.id, confidence_delta=0.1, hop=2)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.last_accessed_hop, 2)
        self.assertEqual(updated.access_count, 2)

    def test_03_memory_merge_and_deduplication(self):
        """Test Memory MERGE operation on semantically duplicate facts."""
        item1 = self.mgr.add_item(
            content="Stuart Rosenberg directed the comedy film Move in 1970.",
            source_chunk_ids=["chunk_a"],
            source_text="Stuart Rosenberg directed Move in 1970.",
            query="Director of Move",
            hop=1
        )
        item2 = self.mgr.add_item(
            content="Stuart Rosenberg directed the comedy film Move in 1970.",
            source_chunk_ids=["chunk_b"],
            source_text="Director of the comedy film Move (1970) is Stuart Rosenberg.",
            query="Who is Move director",
            hop=2
        )
        # Consolidate memory
        res = self.mgr.consolidate_memory(current_hop=2, main_question="Who directed Move?")
        self.assertGreaterEqual(res["duplicates_merged"], 1)
        self.assertEqual(len(self.mgr.get_active_items()), 1)
        self.assertIn("chunk_a", self.mgr.get_active_items()[0].source_chunk_ids)
        self.assertIn("chunk_b", self.mgr.get_active_items()[0].source_chunk_ids)

    def test_04_memory_deletion_and_pruning(self):
        """Test Memory DELETE and capacity limit enforcement."""
        mgr = AdaptiveMemoryManager(max_items=4)
        distinct_facts = [
            ("Paris is the capital and largest city of France.", "France geography"),
            ("Tokyo is the bustling capital city of Japan.", "Japan capital"),
            ("Mount Everest is the highest mountain peak above sea level.", "Everest peak"),
            ("The Pacific Ocean is the largest and deepest ocean on Earth.", "Pacific ocean"),
            ("Python is a high-level interpreted general-purpose programming language.", "Python language"),
            ("Jupiter is the largest planet in the Solar System.", "Jupiter astronomy"),
        ]
        for i, (fact, query) in enumerate(distinct_facts):
            mgr.add_item(
                content=fact,
                source_chunk_ids=[f"chunk_{i}"],
                source_text=fact,
                query=query,
                hop=1,
                relevance=0.7 + (i * 0.04)
            )
        # Consolidate should prune down to max_items=4
        res = mgr.consolidate_memory(current_hop=1, main_question="General knowledge")
        self.assertLessEqual(len(mgr.get_active_items()), 4)
        self.assertGreaterEqual(res["evicted_count"], 2)

    def test_05_contradiction_handling(self):
        """Test Contradiction detection and resolution heuristic."""
        item_a = self.mgr.add_item(
            content="Stuart Rosenberg was born in USA.",
            source_chunk_ids=["c1"],
            source_text="Stuart Rosenberg was born in the USA.",
            query="Origin of Stuart",
            hop=1,
            relevance=0.9,
            evidence_strength=0.95
        )
        item_b = self.mgr.add_item(
            content="Stuart Rosenberg was born in France.",
            source_chunk_ids=["c2"],
            source_text="Stuart Rosenberg was born in France according to early draft.",
            query="Origin of Stuart",
            hop=1,
            relevance=0.8,
            evidence_strength=0.40  # Weaker evidence
        )
        self.mgr.handle_contradiction(item_a, item_b, hop=1)
        self.assertEqual(item_a.status, MemoryStatus.ACTIVE)
        self.assertEqual(item_b.status, MemoryStatus.CONTRADICTED)

    def test_06_multi_objective_scoring_and_age_decay(self):
        """Test that age decay properly reduces retention score over hops."""
        item = self.mgr.add_item(
            content="Méditerranée was directed by Jean-Daniel Pollet.",
            source_chunk_ids=["c3"],
            source_text="Pollet directed Méditerranée (1963).",
            query="Director of Méditerranée",
            hop=1,
            relevance=0.9
        )
        score_hop1 = calculate_memory_retention_score(item, current_hop=1)
        score_hop4 = calculate_memory_retention_score(item, current_hop=4)
        self.assertGreater(score_hop1, score_hop4)

    def test_07_provenance_tracking(self):
        """Test end-to-end provenance creation and citation generation."""
        tracker = ProvenanceTracker()
        tracker.record_provenance(
            item_id="mem_test",
            source_doc="doc_movies.pdf",
            chunk_id="chk_42",
            query="Who directed Move?",
            rank=1,
            hop=1
        )
        active_items = self.mgr.get_active_items()
        citations = tracker.format_citations_for_answer(active_items)
        self.assertIsInstance(citations, list)

    def test_08_branch_memory_isolation_and_promotion(self):
        """Test MCTS Branch Memory isolation and verified global promotion."""
        sys = GlobalBranchMemorySystem(global_max_items=5, branch_max_items=3)
        b1 = sys.create_branch_memory("branch_1")
        b1_item = b1.add_item(
            content="Director Jean-Daniel Pollet is a French filmmaker.",
            source_chunk_ids=["chk_fr"],
            source_text="Pollet was born in France.",
            query="Pollet nationality",
            hop=1,
            relevance=0.95,
            evidence_strength=0.95
        )
        # Global memory should still be empty before promotion
        self.assertEqual(len(sys.global_memory.get_active_items()), 0)
        
        # Promote high-confidence branch discovery to global
        promoted = sys.promote_to_global("branch_1", b1_item.id, confidence_threshold=0.80)
        self.assertIsNotNone(promoted)
        self.assertEqual(len(sys.global_memory.get_active_items()), 1)

    def test_09_rl_reward_interface(self):
        """Test RL reward computation."""
        rl = RLEnvironmentInterface()
        reward = rl.compute_reward(
            is_terminal=True,
            answer_correctness=1.0,
            evidence_quality=0.9,
            memory_quality=0.85,
            retrieval_calls=2,
            reasoning_steps=2,
            unnecessary_retrieval_count=0,
            memory_redundancy=0.0
        )
        self.assertGreater(reward["total_reward"], 3.0)


if __name__ == "__main__":
    unittest.main()
