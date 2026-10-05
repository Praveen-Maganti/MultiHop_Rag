"""
Uncertainty and Retrieval Gating Engine
Evaluates whether working memory can already resolve a subquestion,
preventing unnecessary search/LLM overhead.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any
from app.core.state import ReasoningState
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

UNCERTAINTY_PROMPT = PromptTemplate(
    template="""You are a Meta-Cognitive Uncertainty Evaluator.
Analyze whether the current Working Memory has sufficient factual evidence to answer the Target Subquestion without needing external document retrieval.

Target Subquestion:
{subquestion}

Current Working Memory:
{memory_text}

INSTRUCTIONS:
1. If the memory explicitly contains all facts needed to answer: "needs_retrieval": false, "confidence": 0.95+
2. If critical information is missing or uncertain: "needs_retrieval": true, "confidence": 0.40
3. If memory has conflicting information: "needs_retrieval": true, "confidence": 0.30

OUTPUT FORMAT (return ONLY valid JSON):
{{
  "needs_retrieval": true,
  "confidence": 0.45,
  "rationale": "<explanation of knowledge presence or gap>",
  "focused_query": "<refined search phrase if retrieval is needed>"
}}
""",
    input_variables=["subquestion", "memory_text"]
)

def evaluate_retrieval_uncertainty(subquestion: str, memory_text: str) -> Dict[str, Any]:
    if not memory_text or "No previous memory" in memory_text or "empty" in memory_text.lower():
        return {
            "needs_retrieval": True,
            "confidence": 0.2,
            "rationale": "Memory is empty; initial retrieval required.",
            "focused_query": subquestion
        }

    chain = UNCERTAINTY_PROMPT | llm
    try:
        resp = chain.invoke({"subquestion": subquestion, "memory_text": memory_text})
        text = resp.content
        if isinstance(text, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return {
            "needs_retrieval": bool(data.get("needs_retrieval", True)),
            "confidence": float(data.get("confidence", 0.5)),
            "rationale": data.get("rationale", ""),
            "focused_query": data.get("focused_query") or subquestion
        }
    except Exception as e:
        # Safe fallback: retrieve if uncertain
        return {
            "needs_retrieval": True,
            "confidence": 0.5,
            "rationale": f"Uncertainty evaluation fallback: {str(e)}",
            "focused_query": subquestion
        }
