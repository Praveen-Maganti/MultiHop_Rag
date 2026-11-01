from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json
import time

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

def safe_invoke(chain, input_data: dict, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (attempt + 1) * 3
                time.sleep(wait_time)
            else:
                raise e
    return chain.invoke(input_data)

scorer_prompt = PromptTemplate(
    template="""
Evaluate the following retrieved evidence chunk against the user's question.
Provide scores from 0.0 to 1.0 for:
1. relevance: How relevant is this evidence to answering the question? (0=unrelated, 1=directly answers it)
2. support: How strongly does this evidence support or contribute to a correct answer? (0=no support, 1=fully supports)
3. reliability: How clear, specific and unambiguous is this text? (0=vague/noisy, 1=clear/precise)

Question: {question}
Evidence: {evidence}

Be strict. Only give high scores (>0.6) if the evidence clearly and directly helps answer the question.
Return ONLY a valid JSON object:
{{"relevance": 0.0, "support": 0.0, "reliability": 0.0}}
""",
    input_variables=["question", "evidence"]
)

def score_evidence(question: str, evidence: str, full_text: str = None) -> dict:
    """Score evidence against the question. Uses full_text if provided for better context."""
    chain = scorer_prompt | llm
    # Score against full chunk text when available — short claims lose context
    text_to_score = full_text if full_text and len(full_text) > len(evidence) else evidence
    try:
        response = safe_invoke(chain, {"question": question, "evidence": text_to_score})
        content_obj = response.content
        if isinstance(content_obj, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content_obj])
        else:
            text = str(content_obj)
        text = text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        print(f"Error in score_evidence: {e}")
        return {"relevance": 0.0, "support": 0.0, "reliability": 0.0}

