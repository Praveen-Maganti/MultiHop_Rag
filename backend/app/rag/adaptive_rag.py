"""
Adaptive State-Aware Multi-Hop RAG Master Engine
Unified Orchestrator implementing Adaptive Memory Management, Hybrid Retrieval,
Uncertainty Gating, Socratic Planning, Adaptive MCTS, and Verification.
"""

import time
import uuid
import math
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
    SUBQUESTION_PROMPT,
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

def safe_invoke(chain, input_data: dict, max_retries: int = 4):
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = (attempt + 1) * 4
                time.sleep(wait_time)
            elif any(k in err_str for k in ["RemoteProtocolError", "Connection", "timeout", "503", "500", "502"]):
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
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


def extract_tokens(resp, input_data: dict = None) -> int:
    if hasattr(resp, "usage_metadata") and resp.usage_metadata:
        total = resp.usage_metadata.get("total_tokens")
        if total and int(total) > 0:
            return int(total)
        inp = resp.usage_metadata.get("input_tokens", 0)
        out = resp.usage_metadata.get("output_tokens", 0)
        if inp + out > 0:
            return int(inp + out)
    if hasattr(resp, "response_metadata") and isinstance(resp.response_metadata, dict):
        usage = resp.response_metadata.get("usage_metadata") or resp.response_metadata.get("token_usage")
        if usage and isinstance(usage, dict):
            total = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            if total and int(total) > 0:
                return int(total)
    text_out = str(getattr(resp, "content", resp) or "")
    text_in = str(input_data or "")
    est = (len(text_in) + len(text_out)) // 4
    return max(25, est)


# =============================================================================
# MCTS Node Data Structure
# =============================================================================
class MCTSNode:
    """
    Tree node for Monte Carlo Tree Search over multi-hop reasoning states.
    Represents state S_i = (q_i, r_i, M_i) where:
    - question: target subquestion or reformulated query at this node
    - action: ActionType taken from parent to reach here (A1 to A5)
    - depth: depth level in the search tree (root is 0)
    - parent: pointer to parent MCTSNode (None for root)
    - children: list of child MCTSNodes
    - visits: N(node), number of times this node has been visited
    - total_reward: Q(node), accumulated backpropagated rewards from rollouts
    - branch_memory: isolated AdaptiveMemoryManager state for this branch
    - subanswer: intermediate response or reasoning generated at this node
    - rationale: reasoning explanation for this step
    - is_terminal: boolean, True if terminal node (e.g. A5 Conclude)
    - rollout_answer: final answer synthesized if simulated to terminal
    - path_reward: local step path reward Rp
    """
    def __init__(
        self,
        question: str,
        action: ActionType,
        depth: int = 0,
        parent: Optional["MCTSNode"] = None,
        subanswer: str = "",
        branch_memory: Optional[AdaptiveMemoryManager] = None,
        rationale: str = "",
        is_terminal: bool = False
    ):
        self.node_id = f"node_{uuid.uuid4().hex[:6]}"
        self.question = question
        self.action = action
        self.depth = depth
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.branch_memory = branch_memory or AdaptiveMemoryManager()
        self.subanswer = subanswer
        self.rationale = rationale
        self.is_terminal = is_terminal
        self.rollout_answer: str = ""
        self.path_reward: float = 0.0

    @property
    def value(self) -> float:
        """Average reward Q(node) / N(node)."""
        return (self.total_reward / self.visits) if self.visits > 0 else 0.0

    def uct_score(self, c_param: float = 1.0) -> float:
        """
        Upper Confidence Bound for Trees (UCT):
        UCT(node) = Q(node)/N(node) + C * sqrt(ln(N(parent)) / N(node))
        Unvisited nodes return infinity to prioritize initial exploration.
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent is not None else self.visits
        exploitation = self.total_reward / self.visits
        exploration = c_param * math.sqrt(math.log(max(1, parent_visits)) / self.visits)
        return exploitation + exploration

    def get_path_nodes(self) -> List["MCTSNode"]:
        """Traverse upwards to root and return the list of nodes from root to self."""
        path = []
        curr = self
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    def get_path_string(self) -> str:
        """Format the path trace string."""
        path_nodes = self.get_path_nodes()
        parts = []
        for n in path_nodes:
            if n.depth > 0:
                act = n.action.value if hasattr(n.action, "value") else str(n.action)
                parts.append(f"[{act}] {n.question[:40]} => {n.subanswer[:40]}")
        return " -> ".join(parts) if parts else "Initial State"


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
        total_llm_tokens = 0
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
                fin_inp = {
                    "question": question,
                    "context": state.get_memory_context_text()
                }
                fin_resp = safe_invoke(fin_chain, fin_inp)
                total_llm_tokens += extract_tokens(fin_resp, fin_inp)
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
                q_inp = {"question": uncertainty["focused_query"]}
                q_resp = safe_invoke(q_chain, q_inp)
                total_llm_tokens += extract_tokens(q_resp, q_inp)
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
            ans_inp = {
                "question": target_subq,
                "context": state.get_memory_context_text()
            }
            ans_resp = safe_invoke(ans_chain, ans_inp)
            total_llm_tokens += extract_tokens(ans_resp, ans_inp)
            ans_data = parse_json_safely(ans_resp)
            sub_answer = ans_data.get("answer", str(ans_resp.content))
            state.intermediate_answer = sub_answer

            # 7. Verification step
            verification_res = None
            if self.config.planning.enable_verification:
                verification_res = self.verifier.verify(target_subq, sub_answer, state.get_memory_context_text())
                total_llm_tokens += 120

            # 8. Step Path Reward
            prior_trace = "\n".join([f"Hop {s.step_number}: {s.subquestion} -> {s.subanswer}" for s in state.step_history])
            p_reward_dict = evaluate_path_reward(
                main_question=question,
                reasoning_trace=prior_trace,
                sub_question=target_subq,
                selected_information="\n".join([i.content for i in new_items]) if new_items else "Memory synthesis",
                generated_answer=sub_answer
            )
            total_llm_tokens += 150
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
        total_llm_tokens += 180
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
            "memory_items": len(active_items),
            "tokens": total_llm_tokens,
            "estimated_tokens": total_llm_tokens,
            "duplicates_removed": getattr(memory_mgr, "total_duplicates_merged", 0),
            "low_utility_removed": getattr(memory_mgr, "total_evicted_count", 0),
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
    # Real UCT Monte Carlo Tree Search (MCTS) Engine
    # Implements UCT Selection, Multi-Action Expansion (A1-A5), Simulation, & Backprop
    # =========================================================================
    def _expand_action(
        self,
        action: ActionType,
        parent: "MCTSNode",
        root_question: str,
        document_id: str
    ) -> tuple:
        """
        Executes a single action (A1-A5) to construct a child MCTSNode with cloned branch memory.
        Returns: (child_node, retrieval_calls, llm_tokens)
        """
        child_memory = parent.branch_memory.clone()
        parent_mem_text = "\n".join([i.content for i in child_memory.get_active_items()]) or "No intermediate facts accumulated."
        ret_calls = 0
        tokens = 0

        if action == ActionType.A1_DECOMPOSE_AND_ANSWER:
            # A1: Break question into smaller subquestion, retrieve evidence, and answer
            subq_chain = SUBQUESTION_PROMPT | llm
            subq_inp = {"question": root_question, "context": parent_mem_text}
            subq_resp = safe_invoke(subq_chain, subq_inp)
            tokens += extract_tokens(subq_resp, subq_inp)
            subq_data = parse_json_safely(subq_resp)
            target_subq = subq_data.get("subquestion") or parent.question
            if not target_subq or not str(target_subq).strip():
                target_subq = root_question

            ret_calls += 1
            chunks = self.retriever.retrieve(target_subq, document_id=document_id, top_k=self.config.retrieval.rerank_top_k)
            child_memory.process_incoming_evidence(chunks, subquestion=target_subq, hop=parent.depth + 1, source_doc_id=document_id)

            ans_chain = ANSWER_PROMPT | llm
            ans_inp = {"question": target_subq, "context": "\n".join([i.content for i in child_memory.get_active_items()])}
            ans_resp = safe_invoke(ans_chain, ans_inp)
            tokens += extract_tokens(ans_resp, ans_inp)
            ans_data = parse_json_safely(ans_resp)
            sub_ans = ans_data.get("answer", str(ans_resp.content))

            path_eval = evaluate_path_reward(
                main_question=root_question,
                reasoning_trace=parent.get_path_string(),
                sub_question=target_subq,
                selected_information="\n".join([c.get("text", "") for c in chunks[:2]]) if chunks else "None",
                generated_answer=sub_ans
            )
            tokens += 120

            child_node = MCTSNode(
                question=target_subq,
                action=ActionType.A1_DECOMPOSE_AND_ANSWER,
                depth=parent.depth + 1,
                parent=parent,
                subanswer=sub_ans,
                branch_memory=child_memory,
                rationale=subq_data.get("reasoning", "Decomposed into focused sub-inquiry"),
                is_terminal=False
            )
            child_node.path_reward = float(path_eval.get("overall_path_reward", 0.8))
            return child_node, ret_calls, tokens

        elif action == ActionType.A2_CONSOLIDATE:
            # A2: Consolidate / Internal reflection without external retrieval
            cons_chain = CONSOLIDATE_PROMPT | llm
            cons_inp = {"question": root_question, "context": parent_mem_text}
            cons_resp = safe_invoke(cons_chain, cons_inp)
            tokens += extract_tokens(cons_resp, cons_inp)
            cons_data = parse_json_safely(cons_resp)
            synthesis = cons_data.get("synthesis", str(cons_resp.content))
            deduction = cons_data.get("logical_deduction", "")

            if synthesis and len(synthesis.strip()) > 10:
                child_memory.add_item(
                    content=synthesis,
                    source_chunk_ids=[],
                    source_text=deduction or synthesis,
                    query="A2 Consolidation Reflection",
                    hop=parent.depth + 1,
                    relevance=0.85,
                    evidence_strength=0.85
                )

            child_node = MCTSNode(
                question=f"Consolidate: {root_question}",
                action=ActionType.A2_CONSOLIDATE,
                depth=parent.depth + 1,
                parent=parent,
                subanswer=synthesis,
                branch_memory=child_memory,
                rationale=deduction or "Internal synthesis across memory facts",
                is_terminal=False
            )
            child_node.path_reward = 0.85
            return child_node, ret_calls, tokens

        elif action == ActionType.A3_REFINE:
            # A3: Verify and improve current response against memory
            refine_chain = REFINE_PROMPT | llm
            refine_inp = {
                "question": parent.question,
                "answer": parent.subanswer or "Initial hypothesis",
                "context": parent_mem_text
            }
            refine_resp = safe_invoke(refine_chain, refine_inp)
            tokens += extract_tokens(refine_resp, refine_inp)
            refine_data = parse_json_safely(refine_resp)
            refined_ans = refine_data.get("refined_answer", parent.subanswer)
            critique = refine_data.get("critique", "")

            child_node = MCTSNode(
                question=f"Refine: {parent.question}",
                action=ActionType.A3_REFINE,
                depth=parent.depth + 1,
                parent=parent,
                subanswer=refined_ans,
                branch_memory=child_memory,
                rationale=critique or "Refined answer for consistency",
                is_terminal=False
            )
            child_node.path_reward = 0.85
            return child_node, ret_calls, tokens

        elif action == ActionType.A4_REDIRECT:
            # A4: Question Reformulation to explore an alternative angle
            redir_chain = REDIRECT_PROMPT | llm
            redir_inp = {"question": root_question, "context": parent_mem_text}
            redir_resp = safe_invoke(redir_chain, redir_inp)
            tokens += extract_tokens(redir_resp, redir_inp)
            redir_data = parse_json_safely(redir_resp)
            redirected_q = redir_data.get("redirected_question") or root_question
            if not redirected_q or not str(redirected_q).strip():
                redirected_q = root_question

            ret_calls += 1
            chunks = self.retriever.retrieve(redirected_q, document_id=document_id, top_k=self.config.retrieval.rerank_top_k)
            child_memory.process_incoming_evidence(chunks, subquestion=redirected_q, hop=parent.depth + 1, source_doc_id=document_id)

            ans_chain = ANSWER_PROMPT | llm
            ans_inp = {"question": redirected_q, "context": "\n".join([i.content for i in child_memory.get_active_items()])}
            ans_resp = safe_invoke(ans_chain, ans_inp)
            tokens += extract_tokens(ans_resp, ans_inp)
            ans_data = parse_json_safely(ans_resp)
            sub_ans = ans_data.get("answer", str(ans_resp.content))

            child_node = MCTSNode(
                question=redirected_q,
                action=ActionType.A4_REDIRECT,
                depth=parent.depth + 1,
                parent=parent,
                subanswer=sub_ans,
                branch_memory=child_memory,
                rationale=redir_data.get("rationale", "Redirected query for better answerability"),
                is_terminal=False
            )
            child_node.path_reward = 0.80
            return child_node, ret_calls, tokens

        else:
            # A5: Conclude - produce definitive final answer (terminal node)
            fin_chain = FINALIZE_PROMPT | llm
            fin_inp = {"question": root_question, "context": parent_mem_text}
            fin_resp = safe_invoke(fin_chain, fin_inp)
            tokens += extract_tokens(fin_resp, fin_inp)
            fin_data = parse_json_safely(fin_resp)
            final_ans = fin_data.get("final_answer", str(fin_resp.content))
            reasoning_sum = fin_data.get("reasoning_summary", "")

            child_node = MCTSNode(
                question=root_question,
                action=ActionType.A5_CONCLUDE,
                depth=parent.depth + 1,
                parent=parent,
                subanswer=final_ans,
                branch_memory=child_memory,
                rationale=reasoning_sum or "Synthesized final conclusion from memory",
                is_terminal=True
            )
            child_node.rollout_answer = final_ans
            child_node.path_reward = 1.0
            return child_node, ret_calls, tokens

    def run_mcts(
        self,
        document_id: str,
        question: str,
        ground_truth: str = None,
        max_depth: int = None,
        num_rollouts: int = None
    ) -> Dict[str, Any]:
        """
        True Monte Carlo Tree Search (MCTS) implementation with:
        1. MCTSNode tree tracking visits N(node) and cumulative value Q(node)
        2. UCT Selection balancing exploitation and exploration via mcts_uct_c
        3. Expansion covering all 5 actions A1-A5
        4. Simulation to terminal answer evaluated by neural/LLM judge
        5. Full Backpropagation updating all parent ancestors to root
        """
        start_time = time.time()
        uct_c = getattr(self.config.planning, "mcts_uct_c", 1.0)
        max_depth = max_depth or self.config.planning.mcts_max_depth
        num_rollouts = num_rollouts or self.config.planning.mcts_num_rollouts

        memory_sys = GlobalBranchMemorySystem(global_max_items=8, branch_max_items=5)
        root_memory = memory_sys.create_branch_memory("branch_root")

        root = MCTSNode(
            question=question,
            action=ActionType.A1_DECOMPOSE_AND_ANSWER,
            depth=0,
            parent=None,
            subanswer="Root query initialized.",
            branch_memory=root_memory,
            rationale="Root problem state",
            is_terminal=False
        )

        candidate_answers: List[Dict[str, Any]] = []
        action_distribution: Dict[str, int] = {
            "A1: Decompose & Answer": 0,
            "A2: Consolidate": 0,
            "A3: Refine": 0,
            "A4: Redirect": 0,
            "A5: Conclude": 0
        }
        retrieval_calls = 0
        total_llm_tokens = 0
        total_tree_nodes = 1

        for rollout_idx in range(num_rollouts):
            # -----------------------------------------------------------------
            # 1. SELECTION: Traverse tree using UCT formula to select a leaf
            # -----------------------------------------------------------------
            curr_node = root
            while curr_node.children and not curr_node.is_terminal and curr_node.depth < max_depth:
                best_child = max(curr_node.children, key=lambda c: c.uct_score(c_param=uct_c))
                curr_node = best_child

            # -----------------------------------------------------------------
            # 2. EXPANSION: Expand selected leaf with candidate actions A1-A5
            # -----------------------------------------------------------------
            rollout_target = curr_node
            if not curr_node.is_terminal and curr_node.depth < max_depth:
                mem_items = curr_node.branch_memory.get_active_items()

                if curr_node.depth == 0:
                    actions_to_expand = [
                        ActionType.A1_DECOMPOSE_AND_ANSWER,
                        ActionType.A4_REDIRECT
                    ]
                elif len(mem_items) >= 2 and curr_node.depth >= 2:
                    actions_to_expand = [
                        ActionType.A1_DECOMPOSE_AND_ANSWER,
                        ActionType.A2_CONSOLIDATE,
                        ActionType.A5_CONCLUDE
                    ]
                else:
                    actions_to_expand = [
                        ActionType.A1_DECOMPOSE_AND_ANSWER,
                        ActionType.A3_REFINE,
                        ActionType.A5_CONCLUDE
                    ]

                expanded_children = []
                for act in actions_to_expand:
                    child_node, ret_added, tokens_added = self._expand_action(
                        action=act,
                        parent=curr_node,
                        root_question=question,
                        document_id=document_id
                    )
                    retrieval_calls += ret_added
                    total_llm_tokens += tokens_added
                    total_tree_nodes += 1
                    curr_node.children.append(child_node)
                    expanded_children.append(child_node)

                    act_key = child_node.action.value if hasattr(child_node.action, "value") else str(child_node.action)
                    action_distribution[act_key] = action_distribution.get(act_key, 0) + 1

                rollout_target = expanded_children[0] if expanded_children else curr_node

            # -----------------------------------------------------------------
            # 3. SIMULATION / ROLLOUT: Evaluate terminal answer and reward
            # -----------------------------------------------------------------
            if rollout_target.is_terminal and rollout_target.subanswer:
                rollout_answer = rollout_target.subanswer
            else:
                sim_mem = rollout_target.branch_memory.get_active_items()
                sim_context = "\n".join([i.content for i in sim_mem]) if sim_mem else ""
                fin_chain = FINALIZE_PROMPT | llm
                fin_inp = {
                    "question": question,
                    "context": sim_context or "No intermediate facts accumulated."
                }
                fin_resp = safe_invoke(fin_chain, fin_inp)
                total_llm_tokens += extract_tokens(fin_resp, fin_inp)
                fin_data = parse_json_safely(fin_resp)
                rollout_answer = fin_data.get("final_answer", str(fin_resp.content))

            rollout_target.rollout_answer = rollout_answer

            reward_score = judge_mcts_answer(
                user_question=question,
                system_answer=rollout_answer,
                correct_answer=ground_truth
            )
            total_llm_tokens += 120

            # Promote verified items from rollout branch to global memory
            for itm in rollout_target.branch_memory.get_active_items():
                if itm.evidence_strength >= 0.70:
                    memory_sys.global_memory.add_item(
                        content=itm.content,
                        source_chunk_ids=list(itm.source_chunk_ids),
                        source_text=itm.source_text,
                        query=f"Promoted from MCTS rollout {rollout_idx+1}",
                        hop=rollout_target.depth,
                        relevance=itm.relevance_score,
                        evidence_strength=itm.evidence_strength,
                        provenance_list=list(itm.provenance)
                    )

            path_trace = [
                f"[{n.action.value if hasattr(n.action, 'value') else str(n.action)}] {n.question[:60]} => {n.subanswer[:60]}"
                for n in rollout_target.get_path_nodes() if n.depth > 0
            ]
            candidate_answers.append({
                "rollout": rollout_idx + 1,
                "answer": rollout_answer,
                "reward": reward_score,
                "path": path_trace
            })

            # -----------------------------------------------------------------
            # 4. BACKPROPAGATION: Update visits and total_reward up to root
            # -----------------------------------------------------------------
            backprop_node = rollout_target
            while backprop_node is not None:
                backprop_node.visits += 1
                backprop_node.total_reward += reward_score
                backprop_node = backprop_node.parent

        # ---------------------------------------------------------------------
        # Extract Best Trajectory & Synthesize Final Output
        # ---------------------------------------------------------------------
        if root.children:
            best_child = max(root.children, key=lambda c: (c.value, c.visits))
            best_path_nodes = [best_child]
            curr_trace = best_child
            while curr_trace.children:
                curr_trace = max(curr_trace.children, key=lambda c: (c.value, c.visits))
                best_path_nodes.append(curr_trace)
        else:
            best_path_nodes = [root]

        best_path_trace = [
            f"[{n.action.value if hasattr(n.action, 'value') else str(n.action)}] {n.question[:60]} => {n.subanswer[:60]}"
            for n in best_path_nodes if n.depth > 0
        ]

        candidate_answers.sort(key=lambda x: x["reward"], reverse=True)
        top_candidates = candidate_answers[:3]

        if len(top_candidates) > 1:
            synth_chain = SYNTHESIZE_FINAL_ANSWER_PROMPT | llm
            paths_summary = "\n\n".join([
                f"Candidate Path {c['rollout']} (Judge Reward: {c['reward']:.2f}):\nAnswer: {c['answer']}\nLogic: {' -> '.join(c['path'])}"
                for c in top_candidates
            ])
            synth_inp = {
                "question": question,
                "reasoning_paths": paths_summary
            }
            synth_resp = safe_invoke(synth_chain, synth_inp)
            total_llm_tokens += extract_tokens(synth_resp, synth_inp)
            synth_data = parse_json_safely(synth_resp)
            final_answer = synth_data.get("synthesized_answer", top_candidates[0]["answer"])
        else:
            final_answer = top_candidates[0]["answer"] if top_candidates else "Formulation failed."

        best_reward = top_candidates[0]["reward"] if top_candidates else 0.70
        latency = round(time.time() - start_time, 2)
        global_active = memory_sys.global_memory.get_active_items()

        tree_stats = {
            "total_nodes": total_tree_nodes,
            "root_visits": root.visits,
            "uct_c": uct_c,
            "actions_distribution": action_distribution,
            "best_path_reward": round(best_reward, 3)
        }

        metrics = {
            "mode": "Adaptive State-Aware RAG (MCTS Tree Search)",
            "rollouts": num_rollouts,
            "max_tree_depth": max_depth,
            "hops": len(best_path_trace) or 1,
            "retrieval_calls": retrieval_calls,
            "memory_facts_count": len(global_active),
            "memory_items": len(global_active),
            "tokens": total_llm_tokens,
            "estimated_tokens": total_llm_tokens,
            "latency_seconds": latency,
            "avg_mcts_reward": round(best_reward, 3),
            "cumulative_dual_reward": round(best_reward * 3.0, 4),
            "mcts_uct_c": uct_c,
            "tree_nodes": total_tree_nodes,
            "actions_distribution": action_distribution
        }

        return {
            "answer": final_answer,
            "memory": memory_sys.global_memory.to_dict_list(),
            "hop_trace": best_path_trace,
            "candidate_answers": candidate_answers,
            "tree_stats": tree_stats,
            "dual_reward": {
                "path_rewards": [round(n.path_reward, 3) for n in best_path_nodes if n.path_reward > 0],
                "outcome_reward": {"outcome_reward": round(best_reward, 3)},
                "cumulative_reward": round(best_reward * 3.0, 4)
            },
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
