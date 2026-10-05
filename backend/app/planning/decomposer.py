"""
Adaptive Query Decomposer and Dependency DAG Engine
Analyzes questions, estimates complexity, and creates dependency-tracked subquestions.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Tuple
from app.core.state import SubQuestion
import json
import uuid

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

DECOMPOSER_PROMPT = PromptTemplate(
    template="""You are an expert Query Decomposer and Planning Architect.
Analyze the user's Question. Determine if it is a single-hop lookup or a multi-hop reasoning problem.
If multi-hop, construct an atomic sequence of subquestions with explicit dependencies.

Question:
{question}

GUIDING PRINCIPLES:
1. Do NOT over-decompose simple questions. If a single retrieval can answer it: is_multihop=false.
2. For multi-hop questions, each subquestion must be atomic (one entity/fact).
3. Specify dependencies: a subquestion that depends on the answer of an earlier subquestion must list that subquestion's ID.

OUTPUT FORMAT (return ONLY valid JSON):
{{
  "is_multihop": true,
  "estimated_hops": 2,
  "complexity_reason": "<why single or multi-hop>",
  "subquestions": [
    {{
      "id": "sq_1",
      "question": "<atomic subquestion 1>",
      "dependencies": [],
      "expected_output_type": "entity"
    }},
    {{
      "id": "sq_2",
      "question": "<atomic subquestion 2 referencing sq_1 result>",
      "dependencies": ["sq_1"],
      "expected_output_type": "fact"
    }}
  ]
}}
""",
    input_variables=["question"]
)

class QueryDecomposer:
    """
    Decomposes questions into structured DAGs of SubQuestions.
    """
    def decompose(self, question: str) -> Tuple[bool, int, List[SubQuestion]]:
        chain = DECOMPOSER_PROMPT | llm
        try:
            resp = chain.invoke({"question": question})
            text = resp.content
            if isinstance(text, list):
                text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
            cleaned = text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            
            is_multihop = bool(data.get("is_multihop", True))
            estimated_hops = int(data.get("estimated_hops", 2 if is_multihop else 1))
            
            subq_list: List[SubQuestion] = []
            for sq_dict in data.get("subquestions", []):
                subq_list.append(SubQuestion(
                    id=sq_dict.get("id", f"sq_{uuid.uuid4().hex[:6]}"),
                    question=sq_dict.get("question", question),
                    dependencies=sq_dict.get("dependencies", []),
                    expected_output_type=sq_dict.get("expected_output_type", "entity"),
                    status="PENDING"
                ))
            if not subq_list:
                subq_list = [SubQuestion(question=question, status="PENDING")]

            return is_multihop, estimated_hops, subq_list
        except Exception as e:
            # Fallback
            return False, 1, [SubQuestion(question=question, status="PENDING")]
