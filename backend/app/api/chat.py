from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.rag.multi_hop_rag import run_multi_hop_rag
from app.rag.proposed_rag import run_proposed_rag
from app.rag.state_aware_rag import run_state_aware_socratic_rag, run_state_aware_mcts_rag
from app.rag.adaptive_rag import run_adaptive_state_aware_rag
from app.evaluation.evaluator import evaluate_result
import traceback

router = APIRouter()

class ChatRequest(BaseModel):
    document_id: str
    question: str
    ground_truth: Optional[str] = None
    mode: Optional[str] = "adaptive"  # "adaptive", "socratic", "mcts", "proposed"

@router.post("")
async def chat(request: ChatRequest):
    doc_id = request.document_id
    question = request.question
    ground_truth = request.ground_truth
    mode = (request.mode or "adaptive").lower()

    try:
        # 1. Baseline: Multi-Hop RAG (constant flat memory)
        multihop_results = run_multi_hop_rag(doc_id, question)
        mh_adapter = {
            "answer": multihop_results.get("answer", ""),
            "memory": [{"claim": e["text"], "text": e["text"]} for e in multihop_results.get("evidence", [])]
        }
        mh_eval = evaluate_result(mh_adapter, ground_truth)
        multihop_results["evaluation"] = {
            "f1": mh_eval.get("f1", 0.0),
            "coverage": mh_eval.get("evidence_coverage", 0.0),
            "unsupported": mh_eval.get("unsupported_claim_ratio", 0.0)
        }

        # 2. Selected Mode Execution
        if mode == "mcts":
            state_aware_results = run_state_aware_mcts_rag(doc_id, question, ground_truth=ground_truth, max_depth=4, num_rollouts=3)
        elif mode == "proposed":
            state_aware_results = run_proposed_rag(doc_id, question)
        elif mode == "socratic":
            state_aware_results = run_state_aware_socratic_rag(doc_id, question, ground_truth=ground_truth, max_hops=4)
        else:
            # Default to Master Adaptive State-Aware RAG Engine
            state_aware_results = run_adaptive_state_aware_rag(doc_id, question, ground_truth=ground_truth, mode="adaptive")

        # Evaluate selected method
        eval_adapter = {
            "answer": state_aware_results.get("answer", ""),
            "memory": [
                {"claim": m.get("claim", ""), "text": m.get("text", m.get("claim", ""))}
                for m in state_aware_results.get("memory", [])
                if m.get("status", "ACTIVE") in ["ACTIVE", "VERIFIED"]
            ]
        }
        selected_eval = evaluate_result(eval_adapter, ground_truth)
        state_aware_results["evaluation"] = {
            "f1": selected_eval.get("f1", 0.0),
            "coverage": selected_eval.get("evidence_coverage", 0.0),
            "unsupported": selected_eval.get("unsupported_claim_ratio", 0.0)
        }

        return {
            "standard": multihop_results,
            "proposed": {
                **state_aware_results,
                "hop_trace": state_aware_results.get("hop_trace", []),
                "citations": state_aware_results.get("citations", []),
                "mode": mode
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
