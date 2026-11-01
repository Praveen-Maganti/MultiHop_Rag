"""
AblationFramework
=================
Runs the Proposed RAG 6 times per question — once with all components ON (full system)
and once with each component disabled individually.

Variants:
  full              — all components ON  (baseline)
  no_reranker       — CrossEncoder disabled, use raw FAISS order
  no_filter         — SemanticFilter disabled, all reranked chunks pass
  no_claim_extractor— whole chunk as one claim, no sentence splitting
  no_conflict       — conflict detection disabled
  no_compressor     — memory never evicted/compressed
"""

import asyncio
import uuid
import time
from typing import List, Optional

from app.rag.proposed_rag import run_proposed_rag
from app.evaluation.evaluator import evaluate_result
from app.models.database import save_ablation_run

# ── 6 ablation variants ────────────────────────────────────────────────────
VARIANTS = {
    "full": {
        "reranker": True, "filter": True,
        "claim_extractor": True, "conflict": True, "compressor": True
    },
    "no_reranker": {
        "reranker": False, "filter": True,
        "claim_extractor": True, "conflict": True, "compressor": True
    },
    "no_filter": {
        "reranker": True, "filter": False,
        "claim_extractor": True, "conflict": True, "compressor": True
    },
    "no_claim_extractor": {
        "reranker": True, "filter": True,
        "claim_extractor": False, "conflict": True, "compressor": True
    },
    "no_conflict": {
        "reranker": True, "filter": True,
        "claim_extractor": True, "conflict": False, "compressor": True
    },
    "no_compressor": {
        "reranker": True, "filter": True,
        "claim_extractor": True, "conflict": True, "compressor": False
    },
}

VARIANT_LABELS = {
    "full":               "Full System (all ON)",
    "no_reranker":        "No CrossEncoder Reranker",
    "no_filter":          "No Semantic Filter",
    "no_claim_extractor": "No Claim Extractor",
    "no_conflict":        "No Conflict Detector",
    "no_compressor":      "No Memory Compressor",
}


def _run_single_variant(document_id: str, question: str, ground_truth: Optional[str],
                         variant_name: str, config: dict) -> dict:
    """Run one variant for one question and return all metrics."""
    result = run_proposed_rag(document_id, question, config=config)

    # Evaluation — only ACTIVE/VERIFIED memory items
    eval_adapter = {
        "answer": result.get("answer", ""),
        "memory": [
            {"claim": m.get("claim", ""), "text": m.get("text", m.get("claim", ""))}
            for m in result.get("memory", [])
            if m.get("status", "ACTIVE") in ["ACTIVE", "VERIFIED"]
        ]
    }
    ev = evaluate_result(eval_adapter, ground_truth)
    m  = result.get("metrics", {})

    return {
        "variant":               variant_name,
        "answer":                result.get("answer", ""),
        "f1":                    ev.get("f1", 0.0),
        "precision":             ev.get("precision", 0.0),
        "recall":                ev.get("recall", 0.0),
        "evidence_coverage":     ev.get("evidence_coverage", 0.0),
        "unsupported_claim_rate":ev.get("unsupported_claim_ratio", 0.0),
        "hops":                  m.get("hops", 0),
        "retrieval_calls":       m.get("retrieval_calls", 0),
        "memory_items":          m.get("memory_items", 0),
        "tokens":                m.get("estimated_tokens", 0),
        "latency":               m.get("latency_seconds", 0.0),
        "conflict_llm_calls":    m.get("conflict_llm_calls", 0),
        "conflict_calls_saved":  m.get("conflict_calls_saved", 0),
        "duplicates_removed":    m.get("duplicates_removed", 0),
        "low_utility_removed":   m.get("low_utility_removed", 0),
    }


async def run_ablation(document_id: str,
                       questions: List[str],
                       ground_truths: List[Optional[str]],
                       sleep_between: float = 3.0) -> dict:
    """
    Run all 6 variants across all questions.
    Returns:
      ablation_id  — unique run ID
      per_question — list of {question, variants: {variant_name: metrics}}
      summary      — averaged metrics per variant across all questions
    """
    ablation_id = f"abl_{uuid.uuid4().hex[:8]}"
    per_question = []

    for q, gt in zip(questions, ground_truths):
        q_result = {"question": q, "ground_truth": gt, "variants": {}}

        for variant_name, config in VARIANTS.items():
            print(f"\n[Ablation] variant={variant_name} | q={q[:60]}")
            metrics = _run_single_variant(document_id, q, gt, variant_name, config)
            q_result["variants"][variant_name] = metrics

            # Persist to DB
            save_ablation_run({
                "ablation_id":            ablation_id,
                "document_id":            document_id,
                "question":               q,
                "ground_truth":           gt,
                "variant":                variant_name,
                "answer":                 metrics["answer"],
                "f1":                     metrics["f1"],
                "precision":              metrics["precision"],
                "recall":                 metrics["recall"],
                "evidence_coverage":      metrics["evidence_coverage"],
                "unsupported_claim_rate": metrics["unsupported_claim_rate"],
                "hops":                   metrics["hops"],
                "retrieval_calls":        metrics["retrieval_calls"],
                "memory_items":           metrics["memory_items"],
                "tokens":                 metrics["tokens"],
                "latency":                metrics["latency"],
                "conflict_llm_calls":     metrics["conflict_llm_calls"],
                "conflict_calls_saved":   metrics["conflict_calls_saved"],
                "duplicates_removed":     metrics["duplicates_removed"],
                "low_utility_removed":    metrics["low_utility_removed"],
            })

            await asyncio.sleep(sleep_between)

        per_question.append(q_result)

    # ── Build summary: average each metric per variant ────────────────────
    summary = {}
    for variant_name in VARIANTS:
        rows = [pq["variants"][variant_name] for pq in per_question]
        n = len(rows)

        def avg(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        summary[variant_name] = {
            "label":                 VARIANT_LABELS[variant_name],
            "f1":                    avg("f1"),
            "precision":             avg("precision"),
            "recall":                avg("recall"),
            "evidence_coverage":     avg("evidence_coverage"),
            "unsupported_claim_rate":avg("unsupported_claim_rate"),
            "hops":                  avg("hops"),
            "retrieval_calls":       avg("retrieval_calls"),
            "memory_items":          avg("memory_items"),
            "tokens":                avg("tokens"),
            "latency":               avg("latency"),
            "conflict_llm_calls":    avg("conflict_llm_calls"),
            "conflict_calls_saved":  avg("conflict_calls_saved"),
            "duplicates_removed":    avg("duplicates_removed"),
            "low_utility_removed":   avg("low_utility_removed"),
            # Delta vs full system (positive = improvement from full)
            "f1_delta":              round(avg("f1") - summary["full"]["f1"], 4) if "full" in summary else 0.0,
        }

    # Compute deltas now that all variants are in summary
    full_f1 = summary["full"]["f1"]
    for v in summary:
        summary[v]["f1_delta"] = round(summary[v]["f1"] - full_f1, 4)

    return {
        "ablation_id":  ablation_id,
        "per_question": per_question,
        "summary":      summary,
    }
