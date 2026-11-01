#!/usr/bin/env python
"""
Benchmark script for evaluating the Adaptive State-Aware RAG system on
HotpotQA (distractor) and MuSiQue datasets.

Metrics Hierarchy:
------------------
PRIMARY METRICS (Standard, objective, reproducible across research literature):
  1. Token F1 (Token overlap multiset F1)
  2. Exact Match (EM, strict string equivalence after normalization)
  3. BERTScore (Semantic similarity via contextual embeddings, DistilBERT)

SECONDARY METRICS (System novelty & intermediate reasoning quality):
  - Cumulative Dual Reward (local step path rewards + global outcome reward)
  - Average Path Reward
  - Outcome Reward

SYSTEM EFFICIENCY METRICS:
  - Average Hops Used
  - Average Retrieval Calls
  - Execution Latency (seconds)

Usage:
    python benchmark.py --dataset hotpot --samples 1
    python benchmark.py --dataset hotpot --samples 100 --mode adaptive
    python benchmark.py --dataset musique --samples 50 --output musique_eval.json
"""

import argparse
import random
import sys
import os
import json
import time
from typing import List, Dict, Any
from collections import Counter
import re

# Safeguard against incompatible torchvision installations in environment
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root and backend dir are in sys.path
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
for p in [BACKEND_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

# Dataset loading
from datasets import load_dataset

# Core system imports
from app.retrieval.vector_store import initialize_vector_store
from app.rag.adaptive_rag import run_adaptive_state_aware_rag

# Optional import for BERTScore
_BERTSCORE_AVAILABLE = False
try:
    from bert_score import score as bert_score
    _BERTSCORE_AVAILABLE = True
except Exception as e:
    print(f"Notice: bert-score not loaded directly ({e}). Trying fallback...")

# =============================================================================
# Primary Objective Metrics
# =============================================================================

def _normalize(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and normalizing whitespace."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())

def compute_f1(predicted: str, ground_truth: str) -> float:
    """
    Computes token-level F1 score (standard multiset overlap as used in SQuAD / HotpotQA).
    Measures precision and recall of individual words between prediction and ground truth.
    """
    pred_tokens = _normalize(predicted).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return round(2 * precision * recall / (precision + recall), 4)

def compute_em(predicted: str, ground_truth: str) -> float:
    """
    Exact Match (EM) binary score (1.0 if identical after normalization, 0.0 otherwise).
    Strict precision measure showing whether the system gives exact answers.
    """
    return float(_normalize(predicted) == _normalize(ground_truth))

def compute_bertscore(
    predicted: str,
    ground_truth: str,
    model_type: str = "distilbert-base-uncased"
) -> float:
    """
    Computes BERTScore F1 metric measuring semantic similarity via contextual neural embeddings.
    Catches paraphrased or synonymous answers that token overlap F1 punishes.
    """
    if not predicted or not ground_truth:
        return 0.0
    if not _BERTSCORE_AVAILABLE:
        return 0.0
    try:
        _, _, F1 = bert_score(
            [str(predicted)],
            [str(ground_truth)],
            lang="en",
            model_type=model_type,
            verbose=False
        )
        return round(float(F1.item()), 4)
    except Exception as e:
        print(f"Warning: BERTScore computation failed ({e}). Returning 0.0")
        return 0.0


# =============================================================================
# Context Ingestion
# =============================================================================

def _ingest_context(sample: Dict) -> str:
    """Convert a dataset sample into context passages and index into FAISS."""
    if "context" in sample:
        titles = sample["context"]["title"]
        sentences = sample["context"]["sentences"]
        parts = []
        for title, sents in zip(titles, sentences):
            parts.append(f"{title}: {' '.join(sents)}")
        context_text = "\n\n".join(parts)
    elif "paragraphs" in sample:
        texts = []
        for p in sample["paragraphs"]:
            txt = p.get("paragraph_text") or p.get("text") or ""
            texts.append(txt)
        context_text = "\n\n".join(texts)
    else:
        raise ValueError("Unsupported sample format for context extraction")

    doc_id = f"doc_{random.randint(0, 1_000_000)}"
    chunks = []
    for idx, part in enumerate(context_text.split("\n\n")):
        if part.strip():
            chunks.append({
                "chunk_id": f"{doc_id}_{idx}",
                "document_id": doc_id,
                "text": part.strip(),
                "page": 1,
                "chunk_index": idx,
            })
    initialize_vector_store(doc_id, chunks)
    return doc_id


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark_sample(sample: Dict, mode: str = "adaptive") -> Dict[str, Any]:
    """
    Run a single sample through the RAG system and return:
    - Primary Metrics: F1, Exact Match (EM), BERTScore
    - Secondary Metrics: Dual Reward (Path, Outcome, Cumulative)
    - System Efficiency: Hops, Retrieval Calls, Latency
    """
    doc_id = _ingest_context(sample)
    question = sample["question"]
    ground_truth = sample.get("answer") or sample.get("answer_aliases")
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""
    ground_truth = str(ground_truth or "")

    start_time = time.time()
    result = run_adaptive_state_aware_rag(
        document_id=doc_id,
        question=question,
        ground_truth=ground_truth,
        mode=mode,
    )
    latency = round(time.time() - start_time, 2)
    pred = result.get("answer", "")

    # 1. Primary Metrics (Standard, Objective, Reproducible)
    f1 = compute_f1(pred, ground_truth) if ground_truth else 0.0
    em = compute_em(pred, ground_truth) if ground_truth else 0.0
    b_score = compute_bertscore(pred, ground_truth) if ground_truth else 0.0

    # 2. Secondary Metrics (Dual Reward System for Reasoning Quality)
    dual_reward_data = result.get("dual_reward", {})
    metrics = result.get("metrics", {})
    cumulative_dual = dual_reward_data.get("cumulative_reward")
    if cumulative_dual is None:
        cumulative_dual = metrics.get("cumulative_dual_reward", 0.0)

    path_rewards = dual_reward_data.get("path_rewards", [])
    avg_path = round(sum(path_rewards) / len(path_rewards), 4) if path_rewards else metrics.get("avg_path_reward", 0.0)
    outcome_eval = dual_reward_data.get("outcome_reward")
    if isinstance(outcome_eval, dict):
        outcome_score = outcome_eval.get("outcome_reward", 0.0)
    else:
        outcome_score = metrics.get("outcome_reward", 0.0)

    # Clean up FAISS vector store
    from app.retrieval.vector_store import vector_db
    if doc_id in vector_db:
        del vector_db[doc_id]

    return {
        "question": question,
        "predicted": pred,
        "ground_truth": ground_truth,
        # PRIMARY METRICS
        "f1": f1,
        "exact_match": em,
        "bertscore": b_score,
        # SECONDARY REASONING METRICS
        "dual_reward": dual_reward_data,
        "cumulative_dual_reward": float(cumulative_dual or 0.0),
        "avg_path_reward": float(avg_path or 0.0),
        "outcome_reward": float(outcome_score or 0.0),
        # SYSTEM EFFICIENCY
        "hops_used": metrics.get("hops", 0),
        "retrieval_calls": metrics.get("retrieval_calls", 0),
        "latency_seconds": latency,
    }


def evaluate_dataset(
    name: str,
    sample_count: int = 300,
    mode: str = "adaptive",
    output_path: str = None
) -> Dict[str, Any]:
    print(f"\nLoading '{name}' dataset for evaluation...", flush=True)
    if name.lower() == "hotpot":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=False)
    elif name.lower() == "musique":
        ds = load_dataset("dgslibisey/MuSiQue", split="validation", streaming=False)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    total_avail = len(ds)
    sample_count = min(sample_count, total_avail)
    print(f"Randomly selecting {sample_count} samples from {total_avail} available (Mode: {mode})...\n", flush=True)

    samples = random.sample(list(ds), sample_count)
    results = []

    for i, s in enumerate(samples, 1):
        print(f"[{i}/{sample_count}] Evaluating Question: {s.get('question')[:80]}...", flush=True)
        r = run_benchmark_sample(s, mode=mode)
        results.append(r)
        print(f"    --> Pred: {r['predicted'][:70]}...", flush=True)
        print(f"    --> GT:   {r['ground_truth'][:70]}", flush=True)
        print(f"    --> [PRIMARY]   F1: {r['f1']:.3f} | EM: {r['exact_match']:.1f} | BERTScore: {r['bertscore']:.3f}", flush=True)
        print(f"    --> [SECONDARY] Dual Reward: {r['cumulative_dual_reward']:.3f} | Hops: {r['hops_used']}\n", flush=True)

    # Aggregate Primary Metrics
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_em = sum(r["exact_match"] for r in results) / len(results)
    avg_bertscore = sum(r["bertscore"] for r in results) / len(results)

    # Aggregate Secondary Metrics
    avg_dual_reward = sum(r["cumulative_dual_reward"] for r in results) / len(results)
    avg_path_reward = sum(r["avg_path_reward"] for r in results) / len(results)
    avg_outcome_reward = sum(r["outcome_reward"] for r in results) / len(results)

    # Aggregate Efficiency Metrics
    avg_hops = sum(r["hops_used"] for r in results) / len(results)
    avg_ret_calls = sum(r["retrieval_calls"] for r in results) / len(results)
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)

    # Print Formatted Results Table (Paper-Ready)
    summary_banner = f"""
========================================================================================
                      BENCHMARK EVALUATION RESULTS
========================================================================================
Dataset        : {name.upper()} (Validation)
Sample Count   : {sample_count}
Execution Mode : {mode.upper()}
Timestamp      : {time.strftime('%Y-%m-%d %H:%M:%S')}
========================================================================================

[PRIMARY METRICS] - Standard, Objective & Reproducible
  * Token F1 Score        : {avg_f1:.4f}  ({avg_f1 * 100:.2f}%)
  * Exact Match (EM)      : {avg_em:.4f}  ({avg_em * 100:.2f}%)
  * BERTScore (F1)        : {avg_bertscore:.4f}

[SECONDARY METRICS] - Novel Contribution (Multi-Hop Reasoning Quality)
  * Cumulative Dual Reward: {avg_dual_reward:.4f}
  * Avg Path Reward (Rp)  : {avg_path_reward:.4f}
  * Avg Outcome Reward(Ro): {avg_outcome_reward:.4f}

[SYSTEM EFFICIENCY]
  * Avg Reasoning Hops    : {avg_hops:.2f}
  * Avg Retrieval Calls   : {avg_ret_calls:.2f}
  * Avg Latency per Query : {avg_latency:.2f} seconds

========================================================================================
LaTeX / Markdown Summary Table for Paper:
| System / Configuration | F1 (^) | EM (^) | BERTScore (^) | Dual Reward (^) |
| {mode.title()} (Ours) | {avg_f1*100:.1f} | {avg_em*100:.1f} | {avg_bertscore:.3f} | {avg_dual_reward:.3f} |
========================================================================================
"""
    print(summary_banner)

    summary_data = {
        "dataset": name,
        "sample_count": sample_count,
        "mode": mode,
        "primary_metrics": {
            "token_f1": round(avg_f1, 4),
            "exact_match": round(avg_em, 4),
            "bertscore_f1": round(avg_bertscore, 4),
        },
        "secondary_metrics": {
            "cumulative_dual_reward": round(avg_dual_reward, 4),
            "avg_path_reward": round(avg_path_reward, 4),
            "avg_outcome_reward": round(avg_outcome_reward, 4),
        },
        "efficiency_metrics": {
            "avg_hops": round(avg_hops, 2),
            "avg_retrieval_calls": round(avg_ret_calls, 2),
            "avg_latency_seconds": round(avg_latency, 2),
        },
        "detailed_results": results
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Results successfully saved to: {output_path}")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA / MuSiQue Multi-Hop Benchmark")
    parser.add_argument("--dataset", choices=["hotpot", "musique"], required=True, help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=300, help="Number of random samples to run (e.g. 1, 100, 300)")
    parser.add_argument("--mode", default="adaptive", help="RAG mode (adaptive, socratic, mcts)")
    parser.add_argument("--output", default=None, help="Optional output JSON path for benchmark results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    evaluate_dataset(args.dataset, sample_count=args.samples, mode=args.mode, output_path=args.output)


if __name__ == "__main__":
    main()
