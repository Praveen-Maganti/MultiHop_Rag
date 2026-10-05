# Evaluation Metrics and Benchmarking Guide

## 1. Evaluation Dimensions & Formulations

The evaluation framework assesses both answer accuracy and internal reasoning / memory quality:

1. **Token F1 & Exact Match (EM)**:
   $$\text{Precision} = \frac{|P \cap GT|}{|P|}, \quad \text{Recall} = \frac{|P \cap GT|}{|GT|}, \quad \text{F1} = \frac{2 \cdot P \cdot R}{P + R}$$
2. **Evidence Coverage**: Proportion of factual claims in the final answer that possess verifiable backing in working memory.
3. **Unsupported Claim Ratio**: Fraction of hallucinated or unsubstantiated claims.
4. **Path Reward ($R_p$)**: Step-level local coherence, sufficiency, and factual fidelity.
5. **Outcome Reward ($R_o$)**: Global trajectory correctness, completeness, and logical soundness.
6. **Cumulative Dual Reward**:
   $$\text{Total Dual Reward} = \frac{1}{n-1} \sum_{i=1}^{n-1} R_p(q_i, r_i) + \lambda R_o(y, x, y^*) \quad (\lambda = 2.0)$$
7. **Efficiency Metrics**: Retrieval call count, LLM call count, token usage, latency (seconds).

---

## 2. Benchmark Datasets Supported

* **HotpotQA**: Multi-hop reasoning with distractor documents.
* **2WikiMultiHopQA**: Multi-hop entity relationship chains with clear provenance.
* **MuSiQue**: Multi-hop compositional reasoning.
* **Bamboogle**: Complex 2-hop search benchmarks.
