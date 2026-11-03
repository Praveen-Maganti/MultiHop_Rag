# Systematic Ablation Plan (Phase 20 & Novelty Proofs)

## 1. The 12 Systematic Experimental Variants (A–L)

To scientifically isolate the contribution of each module, the ablation suite evaluates 12 systematic variant configurations. Variants A–J form the progressive architectural pipeline ladder, while Variants K and L provide targeted negative ablations isolating Novel Contributions 1 and 2:

| Variant ID | Configuration Name | Description | Key Mechanism Evaluated |
| :--- | :--- | :--- | :--- |
| **A** | **Standard RAG** | Single-turn dense retrieval + direct generation. | Parametric + 1-hop baseline. |
| **B** | **Multi-Hop Flat Memory** | Iterative retrieval with raw chunk concatenation. | Context accumulation baseline. |
| **C** | **Multi-Hop + Reranking** | Iterative retrieval + Cross-Encoder reranking. | Neural reranker impact. |
| **D** | **Multi-Hop + Query Decomp** | Atomic query decomposition without memory curation. | Structured sub-query generation. |
| **E** | **Multi-Hop + Adaptive Retrieval** | Uncertainty-gated retrieval. | Retrieval efficiency & call reduction. |
| **F** | **Multi-Hop + Verification** | Post-hop self-reflection and factual checking. | Hallucination mitigation. |
| **G** | **Multi-Hop + Adaptive Memory** | Dynamic Working Memory with 11 operations & static scoring. | Core active memory curation. |
| **H** | **Adaptive Memory + Conflicts** | G + Contradiction detection & resolution heuristic. | Robustness against noisy/conflicting data. |
| **I** | **Adaptive Memory + MCTS** | H + Real UCT MCTS Tree Search (Static Weights). | Isolates MCTS planning search over Socratic. |
| **J** | **FULL MASTER SYSTEM** | I + State-Conditioned Dynamic Weight Scheduling (Full Adaptive). | Synergistic overall system performance. |
| **K** | **Full System - Static Weights** | J with Dynamic Weight Scheduling disabled (`dynamic_weights=False`). | **Ablates Contribution 1** (Dynamic Weight Scheduling). |
| **L** | **MCTS - Shared Branch Memory** | MCTS with shared branch memory instead of clone isolation. | **Ablates Contribution 2** (Branch Memory Isolation). |

---

### Step-by-Step Isolation Guarantee
- **G $\rightarrow$ H**: Isolates Contradiction Resolution (Socratic planner and static weights held constant).
- **H $\rightarrow$ I**: Isolates MCTS Tree Search (Static weights held constant).
- **I $\rightarrow$ J**: Isolates State-Conditioned Dynamic Weight Scheduling (Advanced MCTS planner held constant).
- **J vs K**: Targeted negative proof isolating Dynamic Weight Scheduling (Contribution 1).
- **I vs L**: Targeted negative proof isolating Branch Memory Isolation (Contribution 2).

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
