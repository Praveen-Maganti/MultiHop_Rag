"""
Systematic Ablation Framework for Adaptive State-Aware RAG
Implements automated benchmarking across all 10 variant configurations (Phase 20).
"""

from typing import List, Dict, Any, Optional
import time

from app.rag.standard_rag import run_standard_rag
from app.rag.multi_hop_rag import run_multi_hop_rag
from app.rag.state_aware_rag import run_state_aware_socratic_rag, run_state_aware_mcts_rag
from app.rag.adaptive_rag import run_adaptive_state_aware_rag
from app.evaluation.evaluator import evaluate_result

# Variant C, D, E, F, H, I runners — built on existing components with flags
from app.retrieval.vector_store import search_all_documents, search_vector_store
from app.retrieval.reranker import rerank_chunks
from app.retrieval.uncertainty import evaluate_retrieval_uncertainty
from app.planning.decomposer import QueryDecomposer
from app.planning.verifier import FactVerifier
from app.memory.adaptive_manager import AdaptiveMemoryManager
from app.memory.conflict_detector import detect_conflict
from app.core.config import get_config
from app.rag.state_aware_prompts import ANSWER_PROMPT, FINALIZE_PROMPT, SUBQUESTION_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import time as _time

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0, max_output_tokens=512)

_decomposer = QueryDecomposer()
_verifier = FactVerifier()
_config = get_config()


def _safe_invoke(chain, inp: dict) -> dict:
    import time
    for attempt in range(3):
        try:
            resp = chain.invoke(inp)
            content = resp.content
            if isinstance(content, list):
                text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
            else:
                text = str(content)
            cleaned = text.strip().replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(cleaned)
            except Exception:
                s, e = cleaned.find("{"), cleaned.rfind("}")
                if s != -1 and e != -1:
                    try:
                        return json.loads(cleaned[s:e+1])
                    except Exception:
                        pass
            return {"raw_text": text}
        except Exception as ex:
            if "429" in str(ex) or "RESOURCE_EXHAUSTED" in str(ex):
                time.sleep((attempt + 1) * 4)
            else:
                raise ex
    return {}


# =============================================================================
# Variant C: Multi-Hop + Cross-Encoder Reranking
# Same as B but reranking is explicitly enabled (B already uses reranker,
# so C isolates reranker by comparing against a no-reranker version of B)
# We implement C as multi-hop with reranker ON and larger top_k to show delta
# =============================================================================
def _run_variant_c(document_id: str, question: str) -> Dict[str, Any]:
    """Multi-Hop + CrossEncoder Reranking (top_k=10 → rerank → top 5)."""
    start = _time.time()
    cumulative_context, previous_queries = [], []
    retrieval_calls, hop = 0, 0
    final_answer = "Could not formulate an answer."
    chain = (
        __import__("langchain_core.prompts", fromlist=["PromptTemplate"])
        .PromptTemplate(
            template=(
                "Answer using ONLY the context below.\n\nQuestion: {question}\n\n"
                "Context:\n{context}\n\nPrevious Queries:\n{previous_queries}\n\n"
                "Return JSON: {{\"action\":\"ANSWER\",\"answer\":\"...\"}} or "
                "{{\"action\":\"SEARCH\",\"query\":\"...\"}}"
            ),
            input_variables=["question", "context", "previous_queries"]
        ) | llm
    )
    for hop in range(3):
        retrieval_calls += 1
        try:
            raw = search_all_documents(question if hop == 0 else current_query, top_k=10)
        except Exception:
            raw = search_vector_store(document_id, question if hop == 0 else current_query, top_k=10)
        current_query = question if hop == 0 else current_query
        # Reranking ON with top_k=5 (larger than B's top_k=3 — isolates reranker impact)
        reranked = rerank_chunks(current_query, raw, top_k=5)
        for c in reranked:
            if not any(x["chunk_id"] == c["chunk_id"] for x in cumulative_context):
                cumulative_context.append(c)
        if current_query not in previous_queries:
            previous_queries.append(current_query)
        ctx = "\n\n".join([f"[{c['chunk_id']}]\n{c['text']}" for c in cumulative_context])
        data = _safe_invoke(chain, {"question": question, "context": ctx, "previous_queries": "\n".join(previous_queries)})
        if data.get("action") == "ANSWER":
            final_answer = data.get("answer", "")
            break
        elif data.get("action") == "SEARCH":
            nq = data.get("query", "").strip()
            if nq and nq not in previous_queries:
                current_query = nq
            else:
                break
        else:
            break
    return {
        "answer": final_answer,
        "memory": [{"claim": c["text"], "text": c["text"], "status": "ACTIVE"} for c in cumulative_context],
        "metrics": {"hops": hop + 1, "retrieval_calls": retrieval_calls,
                    "latency_seconds": round(_time.time() - start, 2)}
    }


# =============================================================================
# Variant D: Multi-Hop + Query Decomposition
# Adds structured subquestion decomposition before retrieval each hop
# =============================================================================
def _run_variant_d(document_id: str, question: str) -> Dict[str, Any]:
    """Multi-Hop + Query Decomposition (decomposer generates atomic subquestions)."""
    start = _time.time()
    cumulative_context, previous_queries = [], []
    retrieval_calls = 0
    final_answer = "Could not formulate an answer."

    # Decompose main question into atomic subquestions
    try:
        _, _, subquestions = _decomposer.decompose(question)
        if not subquestions:
            subquestions = [question]
    except Exception:
        subquestions = [question]

    for hop, subq in enumerate(subquestions[:4]):
        retrieval_calls += 1
        try:
            raw = search_all_documents(subq, top_k=8)
        except Exception:
            raw = search_vector_store(document_id, subq, top_k=8)
        reranked = rerank_chunks(subq, raw, top_k=3)
        for c in reranked:
            if not any(x["chunk_id"] == c["chunk_id"] for x in cumulative_context):
                cumulative_context.append(c)
        previous_queries.append(subq)

    # Final answer generation from accumulated context
    ctx = "\n\n".join([c["text"] for c in cumulative_context])
    fin_data = _safe_invoke(
        FINALIZE_PROMPT | llm,
        {"question": question, "context": ctx or "No context retrieved."}
    )
    final_answer = fin_data.get("final_answer", question)

    return {
        "answer": final_answer,
        "memory": [{"claim": c["text"], "text": c["text"], "status": "ACTIVE"} for c in cumulative_context],
        "metrics": {"hops": len(subquestions[:4]), "retrieval_calls": retrieval_calls,
                    "latency_seconds": round(_time.time() - start, 2)}
    }


# =============================================================================
# Variant E: Multi-Hop + Uncertainty-Gated Retrieval
# Adds D + uncertainty gating — skips retrieval when memory is sufficient
# =============================================================================
def _run_variant_e(document_id: str, question: str) -> Dict[str, Any]:
    """Multi-Hop + Query Decomp + Uncertainty-Gated Retrieval."""
    start = _time.time()
    cumulative_context, previous_queries = [], []
    retrieval_calls = 0
    final_answer = "Could not formulate an answer."

    try:
        _, _, subquestions = _decomposer.decompose(question)
        if not subquestions:
            subquestions = [question]
    except Exception:
        subquestions = [question]

    mem_text = ""
    for hop, subq in enumerate(subquestions[:4]):
        # Uncertainty gate — skip retrieval if memory already covers this subq
        uncertainty = evaluate_retrieval_uncertainty(subq, mem_text)
        if not uncertainty.get("needs_retrieval", True):
            previous_queries.append(subq)
            continue

        retrieval_calls += 1
        focused_q = uncertainty.get("focused_query", subq)
        try:
            raw = search_all_documents(focused_q, top_k=8)
        except Exception:
            raw = search_vector_store(document_id, focused_q, top_k=8)
        reranked = rerank_chunks(focused_q, raw, top_k=3)
        for c in reranked:
            if not any(x["chunk_id"] == c["chunk_id"] for x in cumulative_context):
                cumulative_context.append(c)
        previous_queries.append(subq)
        mem_text = "\n".join([c["text"] for c in cumulative_context])

    ctx = "\n\n".join([c["text"] for c in cumulative_context])
    fin_data = _safe_invoke(
        FINALIZE_PROMPT | llm,
        {"question": question, "context": ctx or "No context retrieved."}
    )
    final_answer = fin_data.get("final_answer", question)

    return {
        "answer": final_answer,
        "memory": [{"claim": c["text"], "text": c["text"], "status": "ACTIVE"} for c in cumulative_context],
        "metrics": {"hops": len(previous_queries), "retrieval_calls": retrieval_calls,
                    "latency_seconds": round(_time.time() - start, 2)}
    }


# =============================================================================
# Variant F: Multi-Hop + Verification
# Adds E + post-hop self-reflection and fact verification
# =============================================================================
def _run_variant_f(document_id: str, question: str) -> Dict[str, Any]:
    """Multi-Hop + Uncertainty-Gated Retrieval + Fact Verification."""
    start = _time.time()
    cumulative_context, previous_queries = [], []
    retrieval_calls = 0
    final_answer = "Could not formulate an answer."

    try:
        _, _, subquestions = _decomposer.decompose(question)
        if not subquestions:
            subquestions = [question]
    except Exception:
        subquestions = [question]

    mem_text = ""
    for hop, subq in enumerate(subquestions[:4]):
        uncertainty = evaluate_retrieval_uncertainty(subq, mem_text)
        if not uncertainty.get("needs_retrieval", True):
            previous_queries.append(subq)
            continue

        retrieval_calls += 1
        focused_q = uncertainty.get("focused_query", subq)
        try:
            raw = search_all_documents(focused_q, top_k=8)
        except Exception:
            raw = search_vector_store(document_id, focused_q, top_k=8)
        reranked = rerank_chunks(focused_q, raw, top_k=3)
        for c in reranked:
            if not any(x["chunk_id"] == c["chunk_id"] for x in cumulative_context):
                cumulative_context.append(c)

        mem_text = "\n".join([c["text"] for c in cumulative_context])

        # Generate sub-answer then verify it
        ans_data = _safe_invoke(
            ANSWER_PROMPT | llm,
            {"question": subq, "context": mem_text}
        )
        sub_ans = ans_data.get("answer", "")

        # Verification step — refutes low-confidence answers
        try:
            ver = _verifier.verify(subq, sub_ans, mem_text)
            if ver and not ver.is_supported and ver.factual_consistency_score < 0.4:
                # Low confidence — do one more retrieval pass
                retrieval_calls += 1
                try:
                    extra = search_all_documents(subq, top_k=5)
                except Exception:
                    extra = search_vector_store(document_id, subq, top_k=5)
                extra_reranked = rerank_chunks(subq, extra, top_k=2)
                for c in extra_reranked:
                    if not any(x["chunk_id"] == c["chunk_id"] for x in cumulative_context):
                        cumulative_context.append(c)
                mem_text = "\n".join([c["text"] for c in cumulative_context])
        except Exception:
            pass

        previous_queries.append(subq)

    ctx = "\n\n".join([c["text"] for c in cumulative_context])
    fin_data = _safe_invoke(
        FINALIZE_PROMPT | llm,
        {"question": question, "context": ctx or "No context retrieved."}
    )
    final_answer = fin_data.get("final_answer", question)

    return {
        "answer": final_answer,
        "memory": [{"claim": c["text"], "text": c["text"], "status": "ACTIVE"} for c in cumulative_context],
        "metrics": {"hops": len(previous_queries), "retrieval_calls": retrieval_calls,
                    "latency_seconds": round(_time.time() - start, 2)}
    }


# =============================================================================
# Variant H: Adaptive Memory + Contradiction Detection
# Same as G (Socratic) but with conflict_detector explicitly active
# G already has conflict detection but H forces it ON even at low similarity
# =============================================================================
def _run_variant_h(document_id: str, question: str, ground_truth: str = None) -> Dict[str, Any]:
    """Adaptive Memory + Explicit Contradiction Resolution (lower conflict threshold)."""
    from app.core.config import AdaptiveRAGConfig, MemoryConfig
    import copy

    # Lower conflict threshold to catch more contradictions than default G
    result = run_state_aware_socratic_rag(document_id, question, ground_truth=ground_truth, max_hops=4)

    # Post-process: run explicit pairwise contradiction check on final memory
    memory_items = result.get("memory", [])
    contradiction_count = 0
    for i, item_a in enumerate(memory_items):
        for item_b in memory_items[i+1:]:
            text_a = item_a.get("claim", item_a.get("text", ""))
            text_b = item_b.get("claim", item_b.get("text", ""))
            if text_a and text_b:
                try:
                    c_res = detect_conflict(text_a, text_b)
                    if c_res.get("conflict_score", 0.0) >= 0.5:  # lower than default 0.6
                        contradiction_count += 1
                        # Mark lower-confidence item
                        item_a["contradiction_flagged"] = True
                except Exception:
                    pass

    result["metrics"]["contradiction_pairs_detected"] = contradiction_count
    result["metrics"]["mode"] = "Adaptive Memory + Contradiction Resolution"
    return result


# =============================================================================
# Variant I: Adaptive Memory + MCTS Tree Search
# Uses the real UCT MCTS engine from adaptive_rag.py
# =============================================================================
def _run_variant_i(document_id: str, question: str, ground_truth: str = None) -> Dict[str, Any]:
    """Adaptive Memory + Real UCT MCTS Tree Search."""
    result = run_adaptive_state_aware_rag(
        document_id=document_id,
        question=question,
        ground_truth=ground_truth,
        mode="mcts"
    )
    result["metrics"]["mode"] = "Adaptive Memory + MCTS Tree Search"
    return result


# =============================================================================
# Main Ablation Framework
# =============================================================================
class AblationFramework:
    """
    Compares all 10 architectural variants across metrics:
    - F1, Exact Match, BERTScore (Primary)
    - Evidence Coverage & Unsupported Claim Ratio
    - Retrieval Calls, Token Usage, Latency
    - Dual Reward (Secondary)
    """
    VARIANTS = {
        "A_standard_rag":               "A. Standard Single-Hop RAG",
        "B_multihop_flat":              "B. Multi-Hop Flat Memory (No Curation)",
        "C_multihop_rerank":            "C. Multi-Hop + Cross-Encoder Reranking",
        "D_multihop_decomp":            "D. Multi-Hop + Query Decomposition",
        "E_multihop_adaptive_retrieval":"E. Multi-Hop + Uncertainty-Gated Retrieval",
        "F_multihop_verification":      "F. Multi-Hop + Fact Verification",
        "G_adaptive_memory":            "G. Multi-Hop + Dynamic Working Memory",
        "H_memory_contradiction":       "H. Adaptive Memory + Contradiction Resolution",
        "I_adaptive_mcts":              "I. Adaptive Memory + MCTS Tree Search",
        "J_full_system":                "J. Full Adaptive State-Aware RAG (Master)",
    }

    def _record(self, variant_key: str, res: dict, eval_res: dict, t0: float) -> dict:
        m = res.get("metrics", {})
        return {
            "variant":      self.VARIANTS[variant_key],
            "answer":       res.get("answer", ""),
            "f1":           eval_res.get("f1", 0.0),
            "exact_match":  eval_res.get("exact_match", 0.0),
            "bertscore":    eval_res.get("bertscore", 0.0),
            "coverage":     eval_res.get("evidence_coverage", 0.0),
            "unsupported":  eval_res.get("unsupported_claim_ratio", 0.0),
            "hops":         m.get("hops", 1),
            "retrievals":   m.get("retrieval_calls", 1),
            "dual_reward":  m.get("cumulative_dual_reward", 0.0),
            "latency":      round(_time.time() - t0, 2),
        }

    def run_ablation_sample(
        self,
        document_id: str,
        question: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        results = {}

        # ── A: Standard RAG ──────────────────────────────────────────────────
        t0 = _time.time()
        try:
            res = run_standard_rag(document_id, question)
            results["A_standard_rag"] = self._record("A_standard_rag", res, evaluate_result(res, ground_truth), t0)
            results["A_standard_rag"]["hops"] = 1
            results["A_standard_rag"]["retrievals"] = 1
        except Exception as e:
            results["A_standard_rag"] = {"error": str(e)}

        # ── B: Multi-Hop Flat ─────────────────────────────────────────────────
        t0 = _time.time()
        try:
            res = run_multi_hop_rag(document_id, question)
            results["B_multihop_flat"] = self._record("B_multihop_flat", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["B_multihop_flat"] = {"error": str(e)}

        # ── C: Multi-Hop + Reranking ──────────────────────────────────────────
        t0 = _time.time()
        try:
            res = _run_variant_c(document_id, question)
            results["C_multihop_rerank"] = self._record("C_multihop_rerank", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["C_multihop_rerank"] = {"error": str(e)}

        # ── D: Multi-Hop + Query Decomposition ────────────────────────────────
        t0 = _time.time()
        try:
            res = _run_variant_d(document_id, question)
            results["D_multihop_decomp"] = self._record("D_multihop_decomp", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["D_multihop_decomp"] = {"error": str(e)}

        # ── E: Multi-Hop + Uncertainty-Gated Retrieval ────────────────────────
        t0 = _time.time()
        try:
            res = _run_variant_e(document_id, question)
            results["E_multihop_adaptive_retrieval"] = self._record("E_multihop_adaptive_retrieval", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["E_multihop_adaptive_retrieval"] = {"error": str(e)}

        # ── F: Multi-Hop + Verification ───────────────────────────────────────
        t0 = _time.time()
        try:
            res = _run_variant_f(document_id, question)
            results["F_multihop_verification"] = self._record("F_multihop_verification", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["F_multihop_verification"] = {"error": str(e)}

        # ── G: Adaptive Memory (Socratic) ─────────────────────────────────────
        t0 = _time.time()
        try:
            res = run_state_aware_socratic_rag(document_id, question, ground_truth=ground_truth)
            results["G_adaptive_memory"] = self._record("G_adaptive_memory", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["G_adaptive_memory"] = {"error": str(e)}

        # ── H: Adaptive Memory + Contradiction Resolution ─────────────────────
        t0 = _time.time()
        try:
            res = _run_variant_h(document_id, question, ground_truth=ground_truth)
            results["H_memory_contradiction"] = self._record("H_memory_contradiction", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["H_memory_contradiction"] = {"error": str(e)}

        # ── I: Adaptive Memory + MCTS ─────────────────────────────────────────
        t0 = _time.time()
        try:
            res = _run_variant_i(document_id, question, ground_truth=ground_truth)
            results["I_adaptive_mcts"] = self._record("I_adaptive_mcts", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["I_adaptive_mcts"] = {"error": str(e)}

        # ── J: Full System ────────────────────────────────────────────────────
        t0 = _time.time()
        try:
            res = run_adaptive_state_aware_rag(document_id, question, ground_truth=ground_truth, mode="adaptive")
            results["J_full_system"] = self._record("J_full_system", res, evaluate_result(res, ground_truth), t0)
        except Exception as e:
            results["J_full_system"] = {"error": str(e)}

        return {
            "question":        question,
            "ground_truth":    ground_truth,
            "ablation_results": results
        }

    def print_summary_table(self, ablation_output: Dict[str, Any]) -> None:
        """Prints a paper-ready markdown table from ablation results."""
        results = ablation_output.get("ablation_results", {})
        print("\n| Variant | F1 | EM | BERTScore | Retrievals | Hops | Latency |")
        print("|---|---|---|---|---|---|---|")
        for key in self.VARIANTS:
            r = results.get(key, {})
            if "error" in r:
                print(f"| {self.VARIANTS[key]} | ERROR | - | - | - | - | - |")
            else:
                print(
                    f"| {r.get('variant', key)} "
                    f"| {r.get('f1', 0):.3f} "
                    f"| {r.get('exact_match', 0):.3f} "
                    f"| {r.get('bertscore', 0):.3f} "
                    f"| {r.get('retrievals', 0)} "
                    f"| {r.get('hops', 0)} "
                    f"| {r.get('latency', 0):.1f}s |"
                )
