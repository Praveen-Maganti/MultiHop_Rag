from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.rag.ablation_framework import run_ablation, VARIANT_LABELS
from app.models.database import DB_PATH
import sqlite3

router = APIRouter()


class AblationQuestion(BaseModel):
    question: str
    ground_truth: Optional[str] = None


class AblationRequest(BaseModel):
    document_id: str
    questions: List[AblationQuestion]


@router.post("/run")
async def run_ablation_endpoint(request: AblationRequest):
    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one question required.")
    try:
        questions    = [q.question for q in request.questions]
        ground_truths = [q.ground_truth for q in request.questions]
        result = await run_ablation(request.document_id, questions, ground_truths)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ablation_id}")
async def get_ablation_results(ablation_id: str):
    """Fetch a previously run ablation from the DB and rebuild summary."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = lambda c, r: {col[0]: r[i] for i, col in enumerate(c.description)}
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ablation_runs WHERE ablation_id = ?", (ablation_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Ablation run not found.")

    # Group by variant and average
    from collections import defaultdict
    variant_rows = defaultdict(list)
    for row in rows:
        variant_rows[row["variant"]].append(row)

    def avg(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {}
    for variant, vrows in variant_rows.items():
        summary[variant] = {
            "label":                  VARIANT_LABELS.get(variant, variant),
            "f1":                     avg(vrows, "f1"),
            "precision":              avg(vrows, "precision"),
            "recall":                 avg(vrows, "recall"),
            "evidence_coverage":      avg(vrows, "evidence_coverage"),
            "unsupported_claim_rate": avg(vrows, "unsupported_claim_rate"),
            "hops":                   avg(vrows, "hops"),
            "tokens":                 avg(vrows, "tokens"),
            "latency":                avg(vrows, "latency"),
            "memory_items":           avg(vrows, "memory_items"),
            "conflict_llm_calls":     avg(vrows, "conflict_llm_calls"),
            "conflict_calls_saved":   avg(vrows, "conflict_calls_saved"),
            "duplicates_removed":     avg(vrows, "duplicates_removed"),
            "low_utility_removed":    avg(vrows, "low_utility_removed"),
        }

    full_f1 = summary.get("full", {}).get("f1", 0.0)
    for v in summary:
        summary[v]["f1_delta"] = round(summary[v]["f1"] - full_f1, 4)

    return {"ablation_id": ablation_id, "summary": summary}
