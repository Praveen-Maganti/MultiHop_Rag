import json
import uuid
from typing import List, Dict, Any

class ReasoningTrace:
    def __init__(self):
        self.trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        self.hops: List[Dict[str, Any]] = []
        self.current_hop: Dict[str, Any] = None

    def start_hop(self, hop: int, query: str):
        self.current_hop = {
            "hop": hop,
            "query": query,
            "retrieval": {},
            "filtering": {},
            "memory": {},
            "conflicts": {},
            "compression": {},
            "decision": {}
        }
        
    def _check_active(self):
        if self.current_hop is None:
            raise RuntimeError("No active reasoning hop")

    def record_retrieval(self, raw_count: int, reranked_count: int, filtered_count: int, chunks: list):
        self._check_active()
        compact_chunks = []
        for c in chunks:
            compact_chunks.append({
                "chunk_id": c.get("chunk_id"),
                "score": round(c.get("score", 0.0), 4),
                "selected": True
            })
            
        self.current_hop["retrieval"] = {
            "raw_count": raw_count,
            "reranked_count": reranked_count,
            "filtered_count": filtered_count,
            "chunks": compact_chunks
        }
        
    def record_filtering(self, input_count: int, filtered_count: int, removed_count: int):
        self._check_active()
        self.current_hop["filtering"] = {
            "input": input_count,
            "retained": filtered_count,
            "removed": removed_count
        }

    def record_memory_update(self, items_before: int, items_added: int, items_after: int):
        self._check_active()
        self.current_hop["memory"] = {
            "before": items_before,
            "added": items_added,
            "after": items_after
        }

    def record_conflicts(self, conflicts_detected: bool, metrics: dict):
        self._check_active()
        self.current_hop["conflicts"] = {
            "detected": bool(conflicts_detected),
            "possible_pairs": metrics.get("possible_conflict_pairs", 0),
            "candidate_pairs": metrics.get("candidate_conflict_pairs", 0),
            "conflict_llm_calls": metrics.get("conflict_llm_calls", 0),
            "conflict_calls_saved": metrics.get("conflict_calls_saved", 0)
        }

    def record_compression(self, metrics: dict):
        self._check_active()
        if not metrics:
            self.current_hop["compression"] = {"performed": False}
            return
            
        performed = metrics.get("before_compression", 0) > metrics.get("after_compression", 0) or metrics.get("compression_performed", False)
        
        self.current_hop["compression"] = {
            "performed": performed,
            "before": metrics.get("before_compression", 0),
            "after": metrics.get("after_compression", 0),
            "duplicates_removed": metrics.get("duplicates_removed", 0),
            "claims_merged": metrics.get("claims_merged", 0),
            "conflicts_preserved": metrics.get("conflicts_preserved", 0),
            "low_utility_removed": metrics.get("low_utility_removed", 0),
            "compression_ratio": metrics.get("compression_ratio", "0%")
        }

    def record_decision(self, action: str, next_query: str = None, stop_reason: str = None, answer: str = None):
        self._check_active()
        
        if action not in {"SEARCH", "ANSWER", "STOP"}:
            raise ValueError(f"Invalid action: {action}")
            
        decision = {"action": action}
        if next_query is not None:
            decision["next_query"] = next_query
        if stop_reason is not None:
            decision["stop_reason"] = stop_reason
        if answer is not None:
            decision["answer"] = answer
            
        self.current_hop["decision"] = decision
        
    def record_error(self, error_type: str, message: str):
        self._check_active()
        self.current_hop["error"] = {
            "type": error_type,
            "message": str(message)[:1000]
        }
        
    def record_timing(self, latency_seconds: float):
        self._check_active()
        self.current_hop["latency_seconds"] = round(latency_seconds, 3)

    def finalize_hop(self):
        if self.current_hop is None:
            return
        self.hops.append(self.current_hop)
        self.current_hop = None

    def to_list(self) -> List[Dict]:
        return self.hops
        
    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "hops": self.hops
        }
        
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
