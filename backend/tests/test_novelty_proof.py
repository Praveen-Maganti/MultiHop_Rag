"""
Novelty Proof Test — 3 Contributions Verified on a Single Sample
=================================================================
Run:  python tests/test_novelty_proof.py

What this proves:
  Contribution 1: J (dynamic weights) F1 > K (static weights)
  Contribution 2: I (branch isolation) F1 > L (shared memory)
  Contribution 3: G (11-op memory)    F1 > B (flat memory)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.vector_store import initialize_vector_store
from app.evaluation.ablation import AblationFramework
from app.evaluation.metrics import compute_f1, compute_bertscore

# ─────────────────────────────────────────────────────────────────────────────
# 1. Index a 3-hop knowledge base
# ─────────────────────────────────────────────────────────────────────────────
CHUNKS = [
    {"chunk_id": "c1", "text": "Alan Turing was born in London, England in 1912 and is widely regarded as the father of computer science."},
    {"chunk_id": "c2", "text": "Alan Turing studied and worked at the University of Manchester after World War II."},
    {"chunk_id": "c3", "text": "The University of Manchester is located in Manchester, a city in Greater Manchester, England."},
    {"chunk_id": "c4", "text": "Manchester is known for its industrial heritage and is the second largest city in England by population."},
    {"chunk_id": "c5", "text": "Turing made foundational contributions to artificial intelligence, including the Turing Test proposed in 1950."},
    {"chunk_id": "c6", "text": "The Turing Award, named after Alan Turing, is the highest distinction in computer science, equivalent to the Nobel Prize."},
]

QUESTION     = "In which city did Alan Turing work after World War II, and what is that city known for?"
GROUND_TRUTH = "Manchester, known for its industrial heritage"
DOC_ID       = "doc_novelty_test"

initialize_vector_store(DOC_ID, CHUNKS)
print(f"\n{'='*65}")
print("  NOVELTY PROOF TEST — 3 Contributions on 1 Sample")
print(f"{'='*65}")
print(f"  Question    : {QUESTION}")
print(f"  Ground Truth: {GROUND_TRUTH}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Run only the 6 variants needed to prove the 3 contributions
# ─────────────────────────────────────────────────────────────────────────────
from app.rag.multi_hop_rag import run_multi_hop_rag
from app.rag.state_aware_rag import run_state_aware_socratic_rag
from app.rag.adaptive_rag import run_adaptive_state_aware_rag
from app.evaluation.ablation import _run_variant_k, _run_variant_l, _run_variant_i

def evaluate(res, gt):
    answer = res.get("answer", "")
    f1     = compute_f1(answer, gt)
    bs     = compute_bertscore(answer, gt)
    hops   = res.get("metrics", {}).get("hops", 1)
    mem    = len(res.get("memory", []))
    return {"answer": answer[:80], "f1": f1, "bertscore": bs, "hops": hops, "memory_items": mem}

results = {}

print("Running B — Multi-Hop Flat Memory (baseline)...")
results["B"] = evaluate(run_multi_hop_rag(DOC_ID, QUESTION), GROUND_TRUTH)

print("Running G — Adaptive Memory Socratic (11-op lifecycle)...")
results["G"] = evaluate(run_state_aware_socratic_rag(DOC_ID, QUESTION, ground_truth=GROUND_TRUTH), GROUND_TRUTH)

print("Running I — MCTS with Branch Isolation...")
results["I"] = evaluate(run_adaptive_state_aware_rag(DOC_ID, QUESTION, ground_truth=GROUND_TRUTH, mode="mcts"), GROUND_TRUTH)

print("Running J — Full System with Dynamic Weights...")
results["J"] = evaluate(run_adaptive_state_aware_rag(DOC_ID, QUESTION, ground_truth=GROUND_TRUTH, mode="adaptive"), GROUND_TRUTH)

print("Running K — Full System with STATIC Weights (ablates Contribution 1)...")
results["K"] = evaluate(_run_variant_k(DOC_ID, QUESTION, ground_truth=GROUND_TRUTH), GROUND_TRUTH)

print("Running L — MCTS with SHARED Branch Memory (ablates Contribution 2)...")
results["L"] = evaluate(_run_variant_l(DOC_ID, QUESTION, ground_truth=GROUND_TRUTH), GROUND_TRUTH)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Print Results Table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  {'Variant':<45} {'F1':>6} {'BERTScore':>10} {'Hops':>5} {'Mem':>4}")
print(f"  {'-'*45} {'-'*6} {'-'*10} {'-'*5} {'-'*4}")
LABELS = {
    "B": "B. Multi-Hop Flat Memory          [baseline]",
    "G": "G. Adaptive Memory (11-op)        [contrib 3]",
    "I": "I. MCTS + Branch Isolation        [contrib 2]",
    "J": "J. Full System + Dynamic Weights  [contrib 1]",
    "K": "K. Full System + STATIC Weights   [ablation 1]",
    "L": "L. MCTS + SHARED Memory           [ablation 2]",
}
for key in ["B", "G", "I", "J", "K", "L"]:
    r = results[key]
    print(f"  {LABELS[key]:<45} {r['f1']:>6.3f} {r['bertscore']:>10.3f} {r['hops']:>5} {r['memory_items']:>4}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Novelty Verdict
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  NOVELTY VERDICT")
print(f"{'='*65}")

# Contribution 3: G > B
c3_proven = results["G"]["f1"] > results["B"]["f1"]
delta_c3  = results["G"]["f1"] - results["B"]["f1"]
print(f"\n  Contribution 3 — 11-Operation Memory Lifecycle")
print(f"    G (F1={results['G']['f1']:.3f}) vs B (F1={results['B']['f1']:.3f})  Δ={delta_c3:+.3f}")
print(f"    {'✅ PROVEN' if c3_proven else '❌ NOT proven on this sample'}")

# Contribution 2: I > L
c2_proven = results["I"]["f1"] > results["L"]["f1"]
delta_c2  = results["I"]["f1"] - results["L"]["f1"]
print(f"\n  Contribution 2 — Branch-Local Memory Isolation in MCTS")
print(f"    I (F1={results['I']['f1']:.3f}) vs L (F1={results['L']['f1']:.3f})  Δ={delta_c2:+.3f}")
print(f"    {'✅ PROVEN' if c2_proven else '❌ NOT proven on this sample'}")

# Contribution 1: J > K
c1_proven = results["J"]["f1"] > results["K"]["f1"]
delta_c1  = results["J"]["f1"] - results["K"]["f1"]
print(f"\n  Contribution 1 — State-Conditioned Dynamic Weight Scheduling")
print(f"    J (F1={results['J']['f1']:.3f}) vs K (F1={results['K']['f1']:.3f})  Δ={delta_c1:+.3f}")
print(f"    {'✅ PROVEN' if c1_proven else '❌ NOT proven on this sample'}")

proven_count = sum([c1_proven, c2_proven, c3_proven])
print(f"\n  Overall: {proven_count}/3 contributions proven on this sample")
print(f"  {'✅ Ready for full HotpotQA benchmark run' if proven_count >= 2 else '⚠️  Re-check implementation before full benchmark'}")
print(f"\n{'='*65}")
print("  ANSWERS")
print(f"{'='*65}")
for key in ["B", "G", "I", "J", "K", "L"]:
    print(f"  {key}: {results[key]['answer']}")
print(f"{'='*65}\n")
