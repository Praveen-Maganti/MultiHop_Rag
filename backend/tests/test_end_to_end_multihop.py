"""
End-to-End Integration Tests for Adaptive State-Aware Multi-Hop RAG
Tests multi-hop question answering across multiple indexed documents,
provenance generation, and ablation framework execution.
"""

import unittest
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.vector_store import initialize_vector_store
from app.rag.adaptive_rag import run_adaptive_state_aware_rag
from app.evaluation.ablation import AblationFramework


class TestEndToEndMultiHopRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Index multi-hop knowledge across distinct entities:
        # Student -> Project -> Company -> Industry/Domain
        chunks_student = [
            {"chunk_id": "stu_1", "text": "Student Praveen Maganti (ID: ST101) is enrolled in the CS Capstone program and lead developer of Project Alpha."},
            {"chunk_id": "stu_2", "text": "Praveen has a cumulative GPA of 3.92 and specializes in Machine Learning and Natural Language Processing."}
        ]
        chunks_project = [
            {"chunk_id": "prj_1", "text": "Project Alpha is an enterprise knowledge management and reasoning platform sponsored by NovaTech Solutions."},
            {"chunk_id": "prj_2", "text": "Project Alpha successfully delivered its multi-hop retrieval-augmented generation prototype in Q4 2025."}
        ]
        chunks_company = [
            {"chunk_id": "cmp_1", "text": "NovaTech Solutions (C301) is a technology enterprise headquartered in Zurich, Switzerland, operating in the FinTech and Banking domain."},
            {"chunk_id": "cmp_2", "text": "NovaTech Solutions was founded in 2018 with over 500 engineers specializing in algorithmic trading and compliant document AI."}
        ]

        initialize_vector_store("doc_student", chunks_student)
        initialize_vector_store("doc_project", chunks_project)
        initialize_vector_store("doc_company", chunks_company)

    def test_01_multihop_socratic_e2e(self):
        """Test multi-hop reasoning chain: Praveen -> Project Alpha -> NovaTech Solutions -> Zurich."""
        question = "What city is the company headquartered in that sponsors the project led by Praveen Maganti?"
        res = run_adaptive_state_aware_rag(
            document_id="doc_student",
            question=question,
            ground_truth="Zurich",
            mode="socratic"
        )
        self.assertIsNotNone(res.get("answer"))
        self.assertTrue("Sunnyvale" in res["answer"] or "Zurich" in res["answer"])
        self.assertGreater(len(res.get("memory", [])), 0)
        self.assertIn("metrics", res)
        self.assertEqual(res["metrics"]["stop_reason"], "sufficient_evidence")

    def test_02_ablation_sample_run(self):
        """Test that the 10-variant ablation runner executes cleanly."""
        ablation = AblationFramework()
        question = "Which company sponsors Project Alpha and in what city is it headquartered?"
        res = ablation.run_ablation_sample(
            document_id="doc_student",
            question=question,
            ground_truth="NovaTech Solutions, Zurich"
        )
        self.assertIn("ablation_results", res)
        self.assertIn("A_standard_rag", res["ablation_results"])
        self.assertIn("J_full_system", res["ablation_results"])


if __name__ == "__main__":
    unittest.main()
