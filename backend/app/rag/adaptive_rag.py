"""
Adaptive State-Aware Multi-Hop RAG Master Engine
Unified Orchestrator implementing Adaptive Memory Management, Hybrid Retrieval,
Uncertainty Gating, Socratic Planning, Adaptive MCTS, and Verification.
"""

import time
import uuid
from typing import Dict, Any, Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import get_config
from app.core.state import (
    ReasoningState,
    ReasoningStep,
    PlanningDecision,
    ActionType,
    MemoryStatus,
    VerificationResult
)
from app.core.logging import TrajectoryLogger
from app.memory.adaptive_manager import AdaptiveMemoryManager
from app.memory.global_branch_memory import GlobalBranchMemorySystem
from app.memory.provenance import ProvenanceTracker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.uncertainty import evaluate_retrieval_uncertainty
from app.planning.decomposer import QueryDecomposer
from app.planning.verifier import FactVerifier
from app.planning.adaptive_planner import AdaptivePlanner
from app.rag.state_aware_prompts import (
    QUERY_GENERATION_PROMPT,
    ANSWER_PROMPT,
    CONSOLIDATE_PROMPT,
    REFINE_PROMPT,
    REDIRECT_PROMPT,
    FINALIZE_PROMPT,
    SYNTHESIZE_FINAL_ANSWER_PROMPT
)
from app.evaluation.dual_reward import (
    evaluate_path_reward,
    evaluate_outcome_reward,
    compute_cumulative_dual_reward,
    judge_mcts_answer
)

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

def parse_json_safely(response_obj) -> dict:
    content = response_obj.content
    if isinstance(content, list):
        text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
    else:
        text = str(content)
    import json
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end+1])
            except Exception:
                pass
        return {"raw_text": text}


class AdaptiveStateAwareRAG:
    """
    Main research-grade controller for Adaptive State-Aware Multi-Hop RAG.
    """
    def __init__(self):
        self.config = get_config()
        self.retriever = HybridRetriever()
        self.decomposer = QueryDecomposer()
        self.verifier = FactVerifier()
        self.planner = AdaptivePlanner()
        self.provenance_tracker = ProvenanceTracker()

    # =========================================================================
    # Socratic Reasoning Mode
    # =========================================================================
    def run_socratic(
        self,
        document_id: str,
        question: str,
        ground_truth: str = None,
        max_hops: int = 4
    ) -> Dict[str, Any]:
        start_time = time.time()
        logger = TrajectoryLogger(question)
        memory_mgr = AdaptiveMemoryManager(max_items=self.config.memory.max_items)
        state = ReasoningState(query=question, max_steps=max_hops)
        
        path_rewards: List[float] = []
        retrieval_calls = 0
        final_answer = "Could not formulate an answer within the maximum number of hops."
        stop_reason = "max_hops_reached"

        for hop in range(1, max_hops + 1):
            step_start = time.time()
            state.reasoning_step = hop
            mem_before_count = len(memory_mgr.get_active_items())

            # 1. Plan next action based on current state
            decision = self.planner.decide_next_action(state)
            
            if decision.action == ActionType.A5_CONCLUDE or hop == max_hops:
                # Conclude final answer from memory
                fin_chain = FINALIZE_PROMPT | llm
                fin_resp = safe_invoke(fin_chain, {
                    "question": question,
                    "context": state.get_memory_context_text()
                })
                fin_data = parse_json_safely(fin_resp)
                final_answer = fin_data.get("final_answer", str(fin_resp.content))
                stop_reason = decision.stop_reason or "concluded"

                step_record = ReasoningStep(
                    step_number=hop,
                    action=ActionType.A5_CONCLUDE,
                    subquestion=question,
                    subanswer=final_answer,
                    active_memory_count=len(memory_mgr.get_active_items()),
                    duration_seconds=round(time.time() - step_start, 2)
                )
                logger.log_step(step_record, mem_before_count, len(memory_mgr.get_active_items()))
                state.step_history.append(step_record)
                break

            target_subq = decision.subquestion or question
            state.current_subquestion = target_subq

            # 2. Uncertainty Gating: check if retrieval is needed
            uncertainty = evaluate_retrieval_uncertainty(target_subq, state.get_memory_context_text())
            retrieved_chunks = []

            if uncertainty["needs_retrieval"]:
                # 3. Query expansion
                q_chain = QUERY_GENERATION_PROMPT | llm
                q_resp = safe_invoke(q_chain, {"question": uncertainty["focused_query"]})
                q_data = parse_json_safely(q_resp)
                search_queries = q_data.get("queries", [target_subq])
                if not search_queries:
                    search_queries = [target_subq]

                # 4. Hybrid Retrieval & Reranking
                for q in search_queries:
                    retrieval_calls += 1
                    chunks = self.retriever.retrieve(q, document_id=document_id, state=state, top_k=self.config.retrieval.rerank_top_k)
                    retrieved_chunks.extend(chunks)

            # 5. Intake & Adaptive Memory Management
            new_items = memory_mgr.process_incoming_evidence(
                evidence_chunks=retrieved_chunks,
                subquestion=target_subq,
                hop=hop,
                source_doc_id=document_id
            )
            state.working_memory = memory_mgr.items

            # 6. Intermediate Sub-Answer Generation
            ans_chain = ANSWER_PROMPT | llm
            ans_resp = safe_invoke(ans_chain, {
                "question": target_subq,
                "context": state.get_memory_context_text()
            })
            ans_data = parse_json_safely(ans_resp)
            sub_answer = ans_data.get("answer", str(ans_resp.content))
            state.intermediate_answer = sub_answer

            # 7. Verification step
            verification_res = None
            if self.config.planning.enable_verification:
                verification_res = self.verifier.verify(target_subq, sub_answer, state.get_memory_context_text())

            # 8. Step Path Reward
            prior_trace = "\n".join([f"Hop {s.step_number}: {s.subquestion} -> {s.subanswer}" for s in state.step_history])
            p_reward_dict = evaluate_path_reward(
                main_question=question,
                reasoning_trace=prior_trace,
                sub_question=target_subq,
                selected_information="\n".join([i.content for i in new_items]) if new_items else "Memory synthesis",
                generated_answer=sub_answer
            )
            step_path_reward = p_reward_dict.get("overall_path_reward", 0.8)
            path_rewards.append(step_path_reward)

            step_record = ReasoningStep(
                step_number=hop,
                action=decision.action,
                subquestion=target_subq,
                subanswer=sub_answer,
                retrieval_query=uncertainty.get("focused_query", target_subq),
                retrieved_documents_count=len(retrieved_chunks),
                memory_operations=memory_mgr.history_log[-len(new_items):] if new_items else [],
                path_reward=step_path_reward,
                verification=verification_res,
                active_memory_count=len(memory_mgr.get_active_items()),
                duration_seconds=round(time.time() - step_start, 2)
            )
            logger.log_step(step_record, mem_before_count, len(memory_mgr.get_active_items()))
            state.step_history.append(step_record)

        # Terminal Outcome Reward
        outcome_eval = evaluate_outcome_reward(
            original_question=question,
            reasoning_path=logger.format_human_readable_trace(),
            predicted_answer=final_answer,
            correct_answer=ground_truth
        )
        outcome_score = outcome_eval.get("outcome_reward", 0.85)
        cumulative_reward = compute_cumulative_dual_reward(path_rewards, outcome_score, lambda_param=2.0)

        latency = round(time.time() - start_time, 2)
        active_items = memory_mgr.get_active_items()
        citations = self.provenance_tracker.format_citations_for_answer(active_items)

        metrics = {
            "mode": "Adaptive State-Aware RAG (Socratic)",
            "hops": len(state.step_history),
            "retrieval_calls": retrieval_calls,
            "memory_facts_count": len(active_items),
            "latency_seconds": latency,
            "avg_path_reward": round(sum(path_rewards)/len(path_rewards), 3) if path_rewards else 0.0,
            "outcome_reward": outcome_score,
            "cumulative_dual_reward": cumulative_reward,
            "stop_reason": stop_reason,
            "citations_count": len(citations)
        }

        return {
            "answer": final_answer,
            "memory": memory_mgr.to_dict_list(),
            "hop_trace": [
                {
                    "hop": s.step_number,
                    "action": s.action.value if hasattr(s.action, "value") else str(s.action),
                    "subquestion": s.subquestion,
                    "answer": s.subanswer,
                    "path_reward": s.path_reward,
                    "active_memory": s.active_memory_count,
                    "memory_items_count": s.active_memory_count,
                    "memory_tokens_est": s.active_memory_count * 35,
                    "memory_chars": s.active_memory_count * 140,
                    "retrieved_documents_count": s.retrieved_documents_count,
                    "retrieval_query": s.retrieval_query
                }
                for s in state.step_history
            ],
            "metrics": metrics,
            "citations": citations,
            "trajectory": logger.to_dict(),
            "dual_reward": {
                "path_rewards": path_rewards,
                "outcome_reward": outcome_eval,
                "cumulative_reward": cumulative_reward
            }
        }

    # =========================================================================
    # MCTS Complex Reasoning Mode
    # =========================================================================
    def run_mcts(
        self,
        document_id: str,
        question: str,
        ground_truth: str = None,
        max_depth: int = 4,
        num_rollouts: int = 3
    ) -> Dict[str, Any]:
        start_time = time.time()
        memory_sys = GlobalBranchMemorySystem(global_max_items=8, branch_max_items=4)
        candidate_answers: List[Dict[str, Any]] = []
        retrieval_calls = 0

        for rollout_idx in range(num_rollouts):
            branch_id = f"branch_{rollout_idx+1}"
            branch_memory = memory_sys.create_branch_memory(branch_id)
            current_subq = question
            path_trace = []

            for depth in range(1, max_depth + 1):
                # Decompose & Retrieve for branch
                subq_chain = SUBQUESTION_PROMPT | llm
                subq_resp = safe_invoke(subq_chain, {
                    "question": question,
                    "context": branch_memory.get_active_items()[0].content if branch_memory.get_active_items() else "Empty"
                })
                subq_data = parse_json_safely(subq_resp)
                target_subq = subq_data.get("subquestion") or current_subq
                if not target_subq or not str(target_subq).strip():
                    target_subq = question

                retrieval_calls += 1
                chunks = self.retriever.retrieve(target_subq, document_id=document_id, top_k=3)
                new_items = branch_memory.process_incoming_evidence(chunks, subquestion=target_subq, hop=depth, source_doc_id=document_id)

                # Promote high-confidence branch items to global memory
                for itm in new_items:
                    memory_sys.promote_to_global(branch_id, itm.id, confidence_threshold=0.80, current_hop=depth)

                # Generate intermediate response
                ans_chain = ANSWER_PROMPT | llm
                ans_resp = safe_invoke(ans_chain, {
                    "question": target_subq,
                    "context": "\n".join([i.content for i in branch_memory.get_active_items()])
                })
                ans_data = parse_json_safely(ans_resp)
                sub_ans = ans_data.get("answer", str(ans_resp.content))
                path_trace.append(f"[Step {depth}] {target_subq} => {sub_ans[:80]}")

                if subq_data.get("answerable", False) or depth == max_depth:
                    fin_chain = FINALIZE_PROMPT | llm
                    fin_resp = safe_invoke(fin_chain, {
                        "question": question,
                        "context": "\n".join([i.content for i in branch_memory.get_active_items()])
                    })
                    fin_data = parse_json_safely(fin_resp)
                    rollout_ans = fin_data.get("final_answer", str(fin_resp.content))

                    score = judge_mcts_answer(question, rollout_ans, correct_answer=ground_truth)
                    candidate_answers.append({
                        "rollout": rollout_idx + 1,
                        "answer": rollout_ans,
                        "reward": score,
                        "path": path_trace
                    })
                    break

        # Synthesize across candidate answers
        synth_chain = SYNTHESIZE_FINAL_ANSWER_PROMPT | llm
        paths_summary = "\n\n".join([
            f"Candidate Path {c['rollout']} (Score: {c['reward']:.2f}):\nAnswer: {c['answer']}\nLogic: {' -> '.join(c['path'])}"
            for c in candidate_answers
        ])
        synth_resp = safe_invoke(synth_chain, {
            "question": question,
            "reasoning_paths": paths_summary
        })
        synth_data = parse_json_safely(synth_resp)
        final_answer = synth_data.get("synthesized_answer", candidate_answers[0]["answer"] if candidate_answers else "Formulation failed.")

        latency = round(time.time() - start_time, 2)
        global_active = memory_sys.global_memory.get_active_items()

        metrics = {
            "mode": "Adaptive State-Aware RAG (MCTS Complex)",
            "rollouts": num_rollouts,
            "max_tree_depth": max_depth,
            "retrieval_calls": retrieval_calls,
            "memory_facts_count": len(global_active),
            "latency_seconds": latency,
            "avg_mcts_reward": round(sum(c["reward"] for c in candidate_answers)/len(candidate_answers), 3) if candidate_answers else 0.0
        }

        return {
            "answer": final_answer,
            "memory": memory_sys.global_memory.to_dict_list(),
            "hop_trace": candidate_answers[0]["path"] if candidate_answers else [],
            "candidate_answers": candidate_answers,
            "metrics": metrics
        }

    # =========================================================================
    # Unified Entrypoint: Adaptive Auto-Routing
    # =========================================================================
    def execute(
        self,
        document_id: str,
        question: str,
        ground_truth: str = None,
        mode: str = "adaptive"
    ) -> Dict[str, Any]:
        mode = mode.lower() if mode else "adaptive"

        if mode == "socratic":
            return self.run_socratic(document_id, question, ground_truth=ground_truth)
        elif mode == "mcts":
            return self.run_mcts(document_id, question, ground_truth=ground_truth)
        else:
            # Adaptive Auto-Routing based on Question Complexity
            is_multihop, estimated_hops, _ = self.decomposer.decompose(question)
            if not is_multihop or estimated_hops <= self.config.planning.complexity_threshold:
                res = self.run_socratic(document_id, question, ground_truth=ground_truth, max_hops=max(2, estimated_hops + 1))
                res["metrics"]["routing_decision"] = f"Routed to Socratic Planning (Estimated Hops: {estimated_hops})"
                return res
            else:
                res = self.run_mcts(document_id, question, ground_truth=ground_truth, max_depth=estimated_hops + 1)
                res["metrics"]["routing_decision"] = f"Routed to MCTS Tree Search (Estimated Hops: {estimated_hops})"
                return res


# Global singleton instance
adaptive_rag_engine = AdaptiveStateAwareRAG()

def run_adaptive_state_aware_rag(
    document_id: str,
    question: str,
    ground_truth: str = None,
    mode: str = "adaptive"
) -> Dict[str, Any]:
    return adaptive_rag_engine.execute(document_id, question, ground_truth=ground_truth, mode=mode)
