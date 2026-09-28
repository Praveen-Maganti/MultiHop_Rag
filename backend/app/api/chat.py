from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag.standard_rag import run_standard_rag
from app.rag.proposed_rag import run_proposed_rag
from app.evaluation.evaluator import evaluate_result
import os
import traceback

router = APIRouter()

class ChatRequest(BaseModel):
    document_id: str
    question: str
    ground_truth: str = None

@router.post("")
async def chat(request: ChatRequest):
    doc_id = request.document_id
    question = request.question
    ground_truth = request.ground_truth
    
    try:
        standard_results = run_standard_rag(doc_id, question)
        std_adapter = {
            "answer": standard_results.get("answer", ""),
            "memory": [{"claim": e["text"], "text": e["text"]} for e in standard_results.get("evidence", [])]
        }
        std_eval = evaluate_result(std_adapter, ground_truth)
        standard_results["evaluation"] = {
            "f1": std_eval.get("f1", 0.0),
            "coverage": std_eval.get("evidence_coverage", 0.0),
            "unsupported": std_eval.get('unsupported_claim_ratio', 0.0)
        }
        
        proposed_results = run_proposed_rag(doc_id, question)
        prop_eval = evaluate_result(proposed_results, ground_truth)
        proposed_results["evaluation"] = {
            "f1": prop_eval.get("f1", 0.0),
            "coverage": prop_eval.get("evidence_coverage", 0.0),
            "unsupported": prop_eval.get('unsupported_claim_ratio', 0.0)
        }
        
        return {
            "standard": standard_results,
            "proposed": {
                **proposed_results,
                "hop_trace": proposed_results.get("hop_trace", [])
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

