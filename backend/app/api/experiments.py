from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.rag.multi_hop_rag import run_multi_hop_rag
from app.rag.proposed_rag import run_proposed_rag
from app.evaluation.evaluator import evaluate_result
from app.models.database import save_experiment_run, init_db
import uuid
import time
import asyncio

router = APIRouter()
init_db()

class QuestionRequest(BaseModel):
    question: str
    ground_truth: str = None

class ExperimentRequest(BaseModel):
    document_id: str
    questions: List[QuestionRequest]

@router.post("/run")
async def run_experiment(request: ExperimentRequest):
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    results = []
    
    for q_item in request.questions:
        q = q_item.question
        gt = q_item.ground_truth
        
        try:
            mh_res = run_multi_hop_rag(request.document_id, q)
            mh_adapter = {
                "answer": mh_res.get("answer", ""),
                "memory": [{"claim": e["text"], "text": e["text"]} for e in mh_res.get("evidence", [])]
            }
            std_eval = evaluate_result(mh_adapter, ground_truth=gt)

            save_experiment_run({
                "experiment_id": experiment_id,
                "document_id": request.document_id,
                "question": q,
                "ground_truth": gt,
                "system_type": "multihop",
                "answer": mh_res.get("answer", ""),
                "retrieved_chunks": len(mh_res.get("evidence", [])),
                "memory_items": 0,
                "hops": mh_res.get("metrics", {}).get("hops", 1),
                "retrieval_calls": mh_res.get("metrics", {}).get("retrieval_calls", 0),
                "tokens": mh_res.get("metrics", {}).get("estimated_tokens", 0),
                "latency": mh_res.get("metrics", {}).get("latency_seconds", 0),
                "evidence_coverage": std_eval.get("evidence_coverage", 0.0),
                "unsupported_claim_rate": std_eval.get("unsupported_claim_ratio", 0.0),
                "f1": std_eval.get("f1", 0.0),
                "precision": std_eval.get("precision", 0.0),
                "recall": std_eval.get("recall", 0.0)
            })
            
            await asyncio.sleep(4)
            
            prop_res = run_proposed_rag(request.document_id, q)
            prop_eval = evaluate_result(prop_res, ground_truth=gt)
            
            save_experiment_run({
                "experiment_id": experiment_id,
                "document_id": request.document_id,
                "question": q,
                "ground_truth": gt,
                "system_type": "proposed",
                "answer": prop_res.get("answer", ""),
                "retrieved_chunks": 0,
                "memory_items": len(prop_res.get("memory", [])),
                "hops": prop_res.get("metrics", {}).get("hops", 1),
                "retrieval_calls": prop_res.get("metrics", {}).get("retrieval_calls", 0),
                "tokens": prop_res.get("metrics", {}).get("tokens", prop_res.get("metrics", {}).get("estimated_tokens", 0)),
                "latency": prop_res.get("metrics", {}).get("latency", prop_res.get("metrics", {}).get("latency_seconds", 0)),
                "evidence_coverage": prop_eval.get("evidence_coverage", 0.0),
                "unsupported_claim_rate": prop_eval.get("unsupported_claim_ratio", 0.0),
                "f1": prop_eval.get("f1", 0.0),
                "precision": prop_eval.get("precision", 0.0),
                "recall": prop_eval.get("recall", 0.0)
            })
            
            results.append({"question": q, "status": "completed"})
            await asyncio.sleep(4)
        except Exception as e:
            results.append({"question": q, "status": f"failed: {str(e)}"})
            
    return {
        "experiment_id": experiment_id,
        "results": results,
        "message": "Experiment run complete."
    }
