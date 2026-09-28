from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

scorer_prompt = PromptTemplate(
    template="""
Evaluate the following piece of retrieved evidence against the user's question.
Provide scores from 0.0 to 1.0 for:
1. relevance: How relevant is the evidence to the question?
2. support: Does the evidence support a potential answer?
3. reliability: How clear and unambiguous is this text?

Question: {question}
Evidence: {evidence}

Return ONLY a valid JSON object:
{{"relevance": 0.0, "support": 0.0, "reliability": 0.0}}
""",
    input_variables=["question", "evidence"]
)

def score_evidence(question: str, evidence: str) -> dict:
    chain = scorer_prompt | llm
    try:
        response = chain.invoke({"question": question, "evidence": evidence})
        content_obj = response.content
        if isinstance(content_obj, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content_obj])
        else:
            text = str(content_obj)
        text = text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        print(f"Error in score_evidence: {e}")
        print(f"Raw response: {text if 'text' in locals() else 'N/A'}")
        return {"relevance": 0.0, "support": 0.0, "reliability": 0.0}
