from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

conflict_prompt = PromptTemplate(
    template="""
Compare these two claims for factual contradictions.

Claim 1:
{evidence1}

Claim 2:
{evidence2}

Provide a JSON object with:
- "conflict_score": 0.0 (no conflict) to 1.0 (direct contradiction)
- "conflict_importance": 0.0 (trivial detail) to 1.0 (critical contradiction that must be resolved)

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown.
Example:
{{"conflict_score": 0.8, "conflict_importance": 0.9}}
""",
    input_variables=["evidence1", "evidence2"]
)

def detect_conflict(evidence1: str, evidence2: str) -> dict:
    chain = conflict_prompt | llm
    try:
        response = chain.invoke({"evidence1": evidence1, "evidence2": evidence2})
        content_obj = response.content
        if isinstance(content_obj, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content_obj])
        else:
            text = str(content_obj)
        text = text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        print(f"Error in detect_conflict: {e}")
        print(f"Raw response: {text if 'text' in locals() else 'N/A'}")
        return {"conflict_score": 0.0, "conflict_importance": 0.0}
