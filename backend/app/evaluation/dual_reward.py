"""
Path-Outcome Dual Reward System for State-Aware RAG
Directly implements Section 3.2 & Appendix C.4.1 from the ACL 2026 paper.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from app.rag.state_aware_prompts import PATH_REWARD_PROMPT, OUTCOME_REWARD_PROMPT, JUDGE_ANSWER_PROMPT
import json
import time
import os

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
                print(f"Rate limit hit in dual_reward, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    return chain.invoke(input_data)

def evaluate_path_reward(
    main_question: str,
    reasoning_trace: str,
    sub_question: str,
    selected_information: str,
    generated_answer: str
) -> dict:
    """
    Computes local path reward Rp(ri, qi) evaluating:
    1. Relevance of extracted information to sub-question
    2. Sufficiency of information for answering
    3. Logical coherence with reasoning trace
    4. Factuality relative to selected info
    """
    chain = PATH_REWARD_PROMPT | llm
    try:
        response = safe_invoke(chain, {
            "main_question": main_question,
            "reasoning_trace": reasoning_trace if reasoning_trace else "None (Initial Step)",
            "sub_question": sub_question,
            "selected_information": selected_information if selected_information else "None",
            "generated_answer": generated_answer
        })
        text = response.content
        if isinstance(text, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
        text = str(text).strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return {
            "relevance_score": float(data.get("relevance_score", 0.8)),
            "sufficiency_score": float(data.get("sufficiency_score", 0.8)),
            "logical_coherence_score": float(data.get("logical_coherence_score", 0.8)),
            "factuality_score": float(data.get("factuality_score", 0.8)),
            "overall_path_reward": float(data.get("overall_path_reward", 0.8)),
            "critique": data.get("critique", "")
        }
    except Exception as e:
        print(f"Error computing path reward: {e}")
        return {
            "relevance_score": 0.5,
            "sufficiency_score": 0.5,
            "logical_coherence_score": 0.5,
            "factuality_score": 0.5,
            "overall_path_reward": 0.5,
            "critique": f"Evaluation fallback: {str(e)}"
        }


def evaluate_outcome_reward(
    original_question: str,
    reasoning_path: str,
    predicted_answer: str,
    correct_answer: str = None
) -> dict:
    """
    Computes global outcome reward Ro(y, x, y*) evaluating:
    1. Overall Coherence of the reasoning path
    2. Completeness & Sufficiency of intermediate steps
    3. Consistency (no internal contradictions)
    4. Accuracy against ground truth or logical soundness
    """
    chain = OUTCOME_REWARD_PROMPT | llm
    try:
        response = safe_invoke(chain, {
            "original_question": original_question,
            "reasoning_path": reasoning_path,
            "correct_answer": correct_answer if correct_answer else "Not provided (evaluate logical soundness)",
            "predicted_answer": predicted_answer
        })
        text = response.content
        if isinstance(text, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
        text = str(text).strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return {
            "coherence_score": float(data.get("coherence_score", 0.85)),
            "completeness_score": float(data.get("completeness_score", 0.85)),
            "consistency_score": float(data.get("consistency_score", 0.90)),
            "outcome_accuracy_score": float(data.get("outcome_accuracy_score", 0.85)),
            "outcome_reward": float(data.get("outcome_reward", 0.85)),
            "verdict": data.get("verdict", "")
        }
    except Exception as e:
        print(f"Error computing outcome reward: {e}")
        return {
            "coherence_score": 0.5,
            "completeness_score": 0.5,
            "consistency_score": 0.5,
            "outcome_accuracy_score": 0.5,
            "outcome_reward": 0.5,
            "verdict": f"Evaluation fallback: {str(e)}"
        }


def compute_cumulative_dual_reward(path_rewards: list, outcome_reward: float, lambda_param: float = 2.0) -> float:
    """
    Paper Formula (Equation 5):
    Reward = (1 / (n - 1)) * sum_{i=1}^{n-1} Rp(qi, ri) + lambda * Ro(y, x, y*)
    """
    if not path_rewards:
        return round(lambda_param * outcome_reward, 4)
    avg_path = sum(path_rewards) / len(path_rewards)
    total = avg_path + (lambda_param * outcome_reward)
    return round(total, 4)


def judge_mcts_answer(user_question: str, system_answer: str, correct_answer: str = None) -> float:
    """
    Paper Appendix C.4.2: Judge Answer Prompt (0-10 scale for MCTS rollout reward)
    """
    chain = JUDGE_ANSWER_PROMPT | llm
    try:
        response = safe_invoke(chain, {
            "user_question": user_question,
            "system_answer": system_answer,
            "correct_answer": correct_answer if correct_answer else "Not provided"
        })
        text = response.content
        if isinstance(text, list):
            text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
        text = str(text).strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        rating = float(data.get("rating", 7.0))
        return max(0.0, min(1.0, rating / 10.0))
    except Exception as e:
        print(f"Error in judge_mcts_answer: {e}")
        return 0.7
