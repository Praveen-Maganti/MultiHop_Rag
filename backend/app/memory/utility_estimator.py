from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

utility_prompt = PromptTemplate(
    template="""
You are evaluating how useful a specific claim is for the current reasoning task.

--- REASONING STATE ---
Original Question: {question}
Current Hop: {current_hop} / {max_hops}
Current Query: {current_query}
Previous Queries: {previous_queries}
Remaining LLM Budget / Hops: {remaining_hops}

--- CLAIM TO EVALUATE ---
Claim: {claim}
Relevance Score: {relevance_score}
Evidence Score: {evidence_score}
Novelty Score: {novelty_score}
Conflict Importance: {conflict_importance}

1. Does this claim help answer the question?
2. Does it provide information not already in memory?
3. Does it support an unresolved part of the question?
4. Does it resolve or contribute to a conflict?
5. Is it redundant?
6. Will it likely be useful in later hops?
7. Is retaining it worth the remaining budget?

OUTPUT FORMAT:
Return a valid JSON object matching exactly:
{{"utility_score": 0.95, "reason": "short explanation", "keep": true}}
""",
    input_variables=[
        "question", "current_hop", "max_hops", "current_query", 
        "previous_queries", "remaining_hops", "claim", 
        "relevance_score", "evidence_score", "novelty_score", 
        "conflict_importance"
    ]
)

def estimate_utility(reasoning_state: dict, claim_data: dict) -> dict:
    chain = utility_prompt | llm
    try:
        response = chain.invoke({
            "question": reasoning_state.get("question", ""),
            "current_hop": reasoning_state.get("current_hop", 1),
            "max_hops": reasoning_state.get("max_hops", 3),
            "current_query": reasoning_state.get("current_query", ""),
            "previous_queries": reasoning_state.get("previous_queries", ""),
            "remaining_hops": reasoning_state.get("remaining_hops", 1),
            "claim": claim_data.get("claim", ""),
            "relevance_score": claim_data.get("relevance_score", 0.0),
            "evidence_score": claim_data.get("evidence_score", 0.0),
            "novelty_score": claim_data.get("novelty_score", 0.0),
            "conflict_importance": claim_data.get("conflict_importance", 0.0),
        })
        content_obj = response.content
        if isinstance(content_obj, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content_obj])
        else:
            text = str(content_obj)
            
        text = text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        print(f"Error in estimate_utility: {e}")
        return {"utility_score": 0.0, "reason": "Error estimating utility", "keep": False}
