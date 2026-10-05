"""
Adaptive Dynamic Planner
Selects the optimal action at each reasoning step conditioned on working memory state Si.
"""

from typing import Dict, Any, Optional
from app.core.state import ReasoningState, PlanningDecision, ActionType
from app.core.config import get_config
from app.rag.state_aware_prompts import (
    SUBQUESTION_PROMPT,
    CONSOLIDATE_PROMPT,
    REFINE_PROMPT,
    REDIRECT_PROMPT,
    FINALIZE_PROMPT
)
from langchain_google_genai import ChatGoogleGenerativeAI
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

class AdaptivePlanner:
    """
    Decides the next operational action based on working memory completeness,
    contradiction status, remaining step budget, and subquestion dependencies.
    """
    def __init__(self):
        self.config = get_config().planning

    def decide_next_action(self, state: ReasoningState) -> PlanningDecision:
        current_memory_text = state.get_memory_context_text()
        active_items = state.get_active_memory()

        # 1. Terminal condition: reached max steps
        if state.reasoning_step >= state.max_steps:
            return PlanningDecision(
                action=ActionType.A5_CONCLUDE,
                rationale="Reached maximum reasoning step budget; synthesizing conclusion from memory.",
                requires_retrieval=False,
                stop_reason="max_steps_reached"
            )

        # 2. Check for active contradictions in memory
        has_active_conflict = any(i.contradiction_status == "DETECTED" for i in active_items)
        if has_active_conflict:
            return PlanningDecision(
                action=ActionType.RESOLVE_CONFLICT,
                rationale="Unresolved contradiction detected in working memory; resolving before proceeding.",
                requires_retrieval=True
            )

        # 3. Check question answerability with LLM prompt
        subq_chain = SUBQUESTION_PROMPT | llm
        try:
            resp = subq_chain.invoke({
                "question": state.query,
                "context": current_memory_text
            })
            content = resp.content
            if isinstance(content, list):
                text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
            else:
                text = str(content)
            cleaned = text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            
            is_answerable = bool(data.get("answerable", False))
            target_subq = data.get("subquestion")
            
            if is_answerable:
                return PlanningDecision(
                    action=ActionType.A5_CONCLUDE,
                    rationale="Working memory contains complete evidence to answer the main question.",
                    requires_retrieval=False,
                    stop_reason="sufficient_evidence"
                )
            
            if not target_subq or not str(target_subq).strip():
                target_subq = state.query

            return PlanningDecision(
                action=ActionType.A1_DECOMPOSE_AND_ANSWER,
                subquestion=target_subq,
                rationale=data.get("reasoning", "Knowledge gap identified in working memory."),
                requires_retrieval=True
            )
        except Exception as e:
            # Fallback
            return PlanningDecision(
                action=ActionType.A1_DECOMPOSE_AND_ANSWER,
                subquestion=state.query,
                rationale=f"Planner heuristic fallback: {str(e)}",
                requires_retrieval=True
            )
