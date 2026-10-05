# Systematic Ablation Plan (Phase 20)

## 1. The 10 Experimental Variants

To scientifically isolate the contribution of each module, the ablation suite evaluates 10 variant configurations:

| Variant ID | Configuration Name | Description | Key Mechanism Evaluated |
| :--- | :--- | :--- | :--- |
| **A** | **Standard RAG** | Single-turn dense retrieval + direct generation. | Parametric + 1-hop baseline. |
| **B** | **Multi-Hop Flat Memory** | Iterative retrieval with raw chunk concatenation. | Context accumulation baseline. |
| **C** | **Multi-Hop + Reranking** | Iterative retrieval + Cross-Encoder reranking. | Neural reranker impact. |
| **D** | **Multi-Hop + Query Decomp** | Atomic query decomposition without memory curation. | Structured sub-query generation. |
| **E** | **Multi-Hop + Adaptive Retrieval** | Uncertainty-gated retrieval. | Retrieval efficiency & call reduction. |
| **F** | **Multi-Hop + Verification** | Post-hop self-reflection and factual checking. | Hallucination mitigation. |
| **G** | **Multi-Hop + Adaptive Memory** | Dynamic Working Memory with 11 operations & scoring. | Core active memory curation. |
| **H** | **Adaptive Memory + Conflicts** | Dynamic memory + Contradiction detection & resolution. | Robustness against noisy/conflicting data. |
| **I** | **Adaptive Memory + MCTS** | 5-Action MCTS with Branch vs Global memory isolation. | Exhaustive tree search exploration. |
| **J** | **FULL MASTER SYSTEM** | Adaptive State-Aware Multi-Hop RAG (All components). | Synergistic overall performance. |

---

## 2. Running Automated Ablation

Run the ablation suite on any document and question:
```python
from app.evaluation.ablation import AblationFramework

ablation = AblationFramework()
results = ablation.run_ablation_sample(
    document_id="doc_sample",
    question="Where was the director of the project sponsor born?",
    ground_truth="Zurich"
)
print(results["ablation_results"])
```
