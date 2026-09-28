from fastapi import APIRouter
import sqlite3

router = APIRouter()
DB_PATH = "experiments.db"

def dict_factory(cursor, row):
    return {
        col[0]: row[idx]
        for idx, col in enumerate(cursor.description)
    }

@router.get("/{experiment_id}")
async def get_metrics(experiment_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?",
        (experiment_id,)
    )
    runs = cursor.fetchall()
    conn.close()
    
    if not runs:
        return {"error": "Experiment not found"}
        
    std_runs = [r for r in runs if r["system_type"] == "standard"]
    prop_runs = [r for r in runs if r["system_type"] == "proposed"]
    
    def avg(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0
        
    def avg_latency(lst):
        vals = [float(r["latency"]) for r in lst if r["latency"] is not None]
        return f"{sum(vals) / len(vals):.2f}s" if vals else "0.00s"
    
    metrics = {
        "standard": {
            "evidence_coverage": round(avg(std_runs, "evidence_coverage"), 4),
            "unsupported_claim_rate": round(avg(std_runs, "unsupported_claim_rate"), 4),
            "f1": round(avg(std_runs, "f1"), 4),
            "precision": round(avg(std_runs, "precision"), 4),
            "recall": round(avg(std_runs, "recall"), 4),
            "tokens": round(avg(std_runs, "tokens")),
            "retrieval_calls": round(avg(std_runs, "retrieval_calls"), 1),
            "latency": avg_latency(std_runs),
            "memory_items_or_chunks": round(avg(std_runs, "retrieved_chunks"), 1)
        },
        "proposed": {
            "evidence_coverage": round(avg(prop_runs, "evidence_coverage"), 4),
            "unsupported_claim_rate": round(avg(prop_runs, "unsupported_claim_rate"), 4),
            "f1": round(avg(prop_runs, "f1"), 4),
            "precision": round(avg(prop_runs, "precision"), 4),
            "recall": round(avg(prop_runs, "recall"), 4),
            "tokens": round(avg(prop_runs, "tokens")),
            "retrieval_calls": round(avg(prop_runs, "retrieval_calls"), 1),
            "latency": avg_latency(prop_runs),
            "memory_items_or_chunks": round(avg(prop_runs, "memory_items"), 1)
        }
    }
    return metrics
