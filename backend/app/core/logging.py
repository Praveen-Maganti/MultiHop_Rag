"""
Observability & Reasoning Trace Logging for Adaptive State-Aware RAG
Provides structured JSON logging and human-readable trajectory logs.
"""

import json
import time
from typing import Dict, Any, List
from app.core.state import ReasoningState, ReasoningStep, MemoryOperation


class TrajectoryLogger:
    """
    Records full structured reasoning trajectories for inspection,
    ablation analysis, and reinforcement learning state-action logs.
    """
    def __init__(self, query: str):
        self.query = query
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "total_hops": 0,
            "total_retrievals": 0,
            "total_llm_calls": 0,
            "memory_additions": 0,
            "memory_merges": 0,
            "memory_deletions": 0,
            "contradictions_handled": 0,
            "total_latency": 0.0
        }

    def log_step(self, step: ReasoningStep, memory_before_count: int, memory_after_count: int):
        self.steps.append({
            "step": step.step_number,
            "action": step.action.value if hasattr(step.action, "value") else str(step.action),
            "subquestion": step.subquestion,
            "subanswer": step.subanswer,
            "retrieval_query": step.retrieval_query,
            "retrieved_count": step.retrieved_documents_count,
            "memory_before": memory_before_count,
            "memory_after": memory_after_count,
            "operations": [
                {
                    "op": op.operation.value if hasattr(op.operation, "value") else str(op.operation),
                    "target": op.target_item_id,
                    "reason": op.reason
                }
                for op in step.memory_operations
            ],
            "path_reward": step.path_reward,
            "duration": step.duration_seconds
        })

    def format_human_readable_trace(self) -> str:
        lines = [f"=== REASONING TRAJECTORY FOR QUERY: '{self.query}' ==="]
        for s in self.steps:
            lines.append(f"\n[STEP {s['step']}] Action: {s['action']}")
            if s.get("subquestion"):
                lines.append(f"  Subquestion: {s['subquestion']}")
            if s.get("retrieval_query"):
                lines.append(f"  Retrieval Query: {s['retrieval_query']} (Retrieved: {s['retrieved_count']} docs)")
            if s.get("operations"):
                op_strs = [f"{o['op']}({o['reason']})" for o in s["operations"]]
                lines.append(f"  Memory Ops: {', '.join(op_strs)}")
            if s.get("subanswer"):
                lines.append(f"  Intermediate Answer: {s['subanswer']}")
            lines.append(f"  Memory State: {s['memory_before']} -> {s['memory_after']} items | Path Reward: {s['path_reward']:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        self.metrics["total_latency"] = round(time.time() - self.start_time, 2)
        return {
            "query": self.query,
            "steps": self.steps,
            "metrics": self.metrics,
            "total_latency_seconds": self.metrics["total_latency"]
        }
