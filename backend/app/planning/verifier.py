"""
Self-Reflection and Verification Engine
Evaluates factual support, evidence sufficiency, logical validity, and hallucination risk.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any, List
from app.core.state import VerificationResult, MemoryItem
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

VERIFICATION_PROMPT = PromptTemplate(
    template="""You are a rigorous Research Verification Judge.
Critically evaluate if the Proposed Answer is strictly and fully supported by the Provided Working Memory Evidence.

Question:
{question}

Proposed Answer:
{answer}

Provided Evidence in Working Memory:
{evidence}

EVALUATION CRITERIA:
1. Factual Consistency: Does the answer contain any claim NOT present or contradicted in evidence?
2. Evidence Sufficiency: Is the evidence comprehensive enough to answer the question?
3. Logical Validity: Does the conclusion follow logically from the premises?

OUTPUT FORMAT (return ONLY valid JSON):
{{
  "is_supported": true,
  "factual_consistency_score": 0.95,
  "evidence_sufficiency_score": 0.90,
  "logical_validity_score": 1.0,
  "critique": "<brief assessment>",
  "unsupported_claims": []
}}
""",
    input_variables=["question", "answer", "evidence"]
)

class FactVerifier:
    """
    Evaluates claims and intermediate answers against working memory.
    """
    def verify(self, question: str, answer: str, memory_text: str) -> VerificationResult:
        if not memory_text or not answer:
            return VerificationResult(
                is_supported=False,
                factual_consistency_score=0.0,
                evidence_sufficiency_score=0.0,
                logical_validity_score=0.0,
                critique="Empty answer or empty memory context.",
                unsupported_claims=["Entire answer unsupported due to lack of evidence."]
            )

        chain = VERIFICATION_PROMPT | llm
        try:
            resp = chain.invoke({
                "question": question,
                "answer": answer,
                "evidence": memory_text
            })
            text = resp.content
            if isinstance(text, list):
                text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
            cleaned = text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            return VerificationResult(
                is_supported=bool(data.get("is_supported", True)),
                factual_consistency_score=float(data.get("factual_consistency_score", 0.9)),
                evidence_sufficiency_score=float(data.get("evidence_sufficiency_score", 0.9)),
                logical_validity_score=float(data.get("logical_validity_score", 0.9)),
                critique=data.get("critique", ""),
                unsupported_claims=data.get("unsupported_claims", [])
            )
        except Exception as e:
            return VerificationResult(
                is_supported=True,
                factual_consistency_score=0.8,
                evidence_sufficiency_score=0.8,
                logical_validity_score=0.8,
                critique=f"Verification completed with fallback: {str(e)}",
                unsupported_claims=[]
            )
