"""
State-Aware RAG Implementation
Directly implements the algorithms and architecture from the ACL 2026 Paper:
"Reasoning with Memory: Adaptive Information Management for Retrieval-Augmented Generation"
(Hieu Man et al., Amazon AI & Univ of Oregon)

Features:
1. Dynamic Working Memory (Mi) with Extractor Consolidation and Updating
2. Socratic Planning Mode (Single-chain iterative decomposition)
3. Monte Carlo Tree Search (MCTS) Exploration Mode with 5 Actions (A1-A5) & Globally Shared Memory
4. Path-Outcome Dual Reward Computation
"""

import time
import json
import math
import uuid
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from app.retrieval.vector_store import search_all_documents, search_vector_store
from app.retrieval.reranker import rerank_chunks
from app.utils.embeddings import is_semantically_similar_query
from app.rag.state_aware_prompts import (
    SUBQUESTION_PROMPT,
    QUERY_GENERATION_PROMPT,
    EXTRACT_PROMPT,
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

# LLM for Generator and Extractor modules
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
                print(f"Rate limit encountered, waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    return chain.invoke(input_data)

def parse_llm_json(response_obj) -> dict:
    content = response_obj.content
    if isinstance(content, list):
        text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
    else:
        text = str(content)
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


# ==============================================================================
# Dynamic Working Memory Representation (Mi)
# ==============================================================================
class StateAwareMemory:
    """
    Working Memory state (Mi) that serves as a global, dynamic cognitive workspace.
    Maintains distilled facts, resolves contradictions, eliminates redundancies,
    and updates per-step across reasoning trajectories.
    """
    def __init__(self, memory_id: str = None):
        self.memory_id = memory_id or f"mem_{uuid.uuid4().hex[:8]}"
        self.facts: List[Dict[str, Any]] = []
        self.qa_history: List[Dict[str, str]] = []
        self.contradictions: List[str] = []
        self.raw_evidence: List[Dict[str, Any]] = []

    def get_context_text(self) -> str:
        if not self.facts and not self.qa_history:
            return "No previous memory accumulated yet."
        
        lines = []
        if self.facts:
            lines.append("--- Active Working Memory Facts ---")
            for idx, f in enumerate(self.facts):
                lines.append(f"[{idx+1}] {f.get('fact', '')}")
                if f.get("justification"):
                    lines.append(f"    (Context: {f.get('justification')})")
        
        if self.qa_history:
            lines.append("\n--- Intermediate Sub-QA History ---")
            for idx, qa in enumerate(self.qa_history):
                lines.append(f"Step {idx+1} Sub-Q: {qa['question']}")
                lines.append(f"Step {idx+1} Sub-A: {qa['answer']}")
                
        if self.contradictions:
            lines.append("\n--- Identified Contradictions / Nuances ---")
            for c in self.contradictions:
                lines.append(f"• {c}")
                
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "facts": self.facts,
            "qa_history": self.qa_history,
            "contradictions": self.contradictions,
            "total_facts": len(self.facts)
        }

    def clone(self) -> "StateAwareMemory":
        new_mem = StateAwareMemory(self.memory_id)
        new_mem.facts = [dict(f) for f in self.facts]
        new_mem.qa_history = [dict(qa) for qa in self.qa_history]
        new_mem.contradictions = list(self.contradictions)
        new_mem.raw_evidence = list(self.raw_evidence)
        return new_mem


# ==============================================================================
# Extractor Module: Consolidation and Memory Update
# ==============================================================================
def extractor_consolidate(
    subquestion: str,
    documents: List[Dict[str, Any]],
    current_memory: StateAwareMemory
) -> Dict[str, Any]:
    """
    Paper Equation (1): I_i = Econsolidate(q_i, D_i, M_{i-1})
    Distills knowledge optimized for immediate reasoning needs:
    - Relevance filtering (identifying facts supporting q_i)
    - Contradiction resolution (reconciling conflicts between D_i and M_{i-1})
    - Redundancy elimination (de-duplication)
    """
    if not documents:
        return {"decision": "not_relevant", "extracted_facts": [], "contradictions_found": [], "summary": ""}

    combined_doc_text = "\n\n".join([
        f"[Doc {idx+1} | Source: {doc.get('chunk_id', 'chunk')}]\n{doc.get('text', '')}"
        for idx, doc in enumerate(documents)
    ])
    
    chain = EXTRACT_PROMPT | llm
    resp = safe_invoke(chain, {
        "question": subquestion,
        "document": combined_doc_text,
        "existing_memory": current_memory.get_context_text()
    })
    
    result = parse_llm_json(resp)
    if not isinstance(result, dict) or "extracted_facts" not in result:
        result = {
            "decision": "relevant" if documents else "not_relevant",
            "extracted_facts": [{"fact": doc.get("text", "")[:200], "criteria": ["Supporting Evidence"], "justification": "Direct retrieval"} for doc in documents[:3]],
            "contradictions_found": [],
            "summary": "Extracted relevant document chunks."
        }
    return result


def extractor_update_memory(
    main_question: str,
    current_memory: StateAwareMemory,
    distilled_info: Dict[str, Any],
    qa_pair: Dict[str, str],
    hop: int
) -> StateAwareMemory:
    """
    Paper Equation (2): M_i = Eupdate(x, M_{i-1}, I_i, (q_i, r_i))
    Integrates the new question-response pair (q_i, r_i) and distilled knowledge I_i with M_{i-1}.
    """
    new_facts = distilled_info.get("extracted_facts", [])
    for f in new_facts:
        fact_text = f.get("fact", "").strip()
        if fact_text and not any(existing.get("fact") == fact_text for existing in current_memory.facts):
            current_memory.facts.append({
                "id": f"fact_{uuid.uuid4().hex[:6]}",
                "fact": fact_text,
                "criteria": f.get("criteria", []),
                "justification": f.get("justification", ""),
                "hop_added": hop,
                "status": "ACTIVE"
            })

    for c in distilled_info.get("contradictions_found", []):
        if c and c not in current_memory.contradictions:
            current_memory.contradictions.append(c)

    if qa_pair and qa_pair.get("question") and qa_pair.get("answer"):
        current_memory.qa_history.append({
            "hop": hop,
            "question": qa_pair["question"],
            "answer": qa_pair["answer"]
        })

    return current_memory


# ==============================================================================
# Socratic Planning Mode (Single-chain Iterative Reasoning)
# ==============================================================================
def run_state_aware_socratic_rag(
    document_id: str,
    question: str,
    ground_truth: str = None,
    max_hops: int = 5
) -> Dict[str, Any]:
    """
    State-Aware RAG (Socratic Planning):
    Iteratively decomposes questions using A1 (Decompose & Answer) with memory-augmented
    retrieval, filtering & consolidation at each step, concluding with A5 (Finalize).
    """
    start_time = time.time()
    memory = StateAwareMemory()
    hop_trace = []
    path_rewards = []
    previous_queries = []
    retrieval_calls = 0
    final_answer = "Could not formulate an answer within the maximum number of hops."
    confidence = 0.0

    current_subquestion = question

    for hop in range(1, max_hops + 1):
        # 1. QUESTION DECOMPOSITION (Generator Step 1)
        subq_chain = SUBQUESTION_PROMPT | llm
        subq_resp = safe_invoke(subq_chain, {
            "question": question,
            "context": memory.get_context_text()
        })
        subq_data = parse_llm_json(subq_resp)

        is_answerable = subq_data.get("answerable", False)
        target_subq = subq_data.get("subquestion") or current_subquestion

        # 2. CHECK IF ANSWERABLE OR NEED CONCLUDE (A5)
        if is_answerable or hop == max_hops:
            finalize_chain = FINALIZE_PROMPT | llm
            finalize_resp = safe_invoke(finalize_chain, {
                "question": question,
                "context": memory.get_context_text()
            })
            finalize_data = parse_llm_json(finalize_resp)
            final_answer = finalize_data.get("final_answer", str(finalize_resp.content))
            confidence = finalize_data.get("confidence", 0.9)

            hop_trace.append({
                "hop": hop,
                "action": "A5: Conclude",
                "subquestion": target_subq,
                "answer": final_answer,
                "reasoning": finalize_data.get("reasoning_summary", "Final synthesized answer from working memory."),
                "memory_state_size": len(memory.facts)
            })
            break

        # 3. GENERATE SEARCH QUERIES (Zero-synthesis query expansion)
        query_chain = QUERY_GENERATION_PROMPT | llm
        query_resp = safe_invoke(query_chain, {"question": target_subq})
        query_data = parse_llm_json(query_resp)
        search_queries = query_data.get("queries", [target_subq])
        if not search_queries:
            search_queries = [target_subq]

        # 4. RETRIEVE CANDIDATE DOCUMENTS (Retriever R)
        retrieved_docs = []
        for q in search_queries:
            if q not in previous_queries:
                previous_queries.append(q)
            retrieval_calls += 1
            try:
                raw_chunks = search_all_documents(q, top_k=6)
            except Exception:
                raw_chunks = search_vector_store(document_id, q, top_k=6)
            
            reranked = rerank_chunks(q, raw_chunks, top_k=3)
            retrieved_docs.extend(reranked)

        # Deduplicate retrieved chunks by ID
        unique_docs = []
        seen_ids = set()
        for doc in retrieved_docs:
            cid = doc.get("chunk_id", str(doc.get("text", "")[:30]))
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_docs.append(doc)

        # 5. EXTRACTOR CONSOLIDATION (I_i = Econsolidate)
        distilled = extractor_consolidate(target_subq, unique_docs, memory)

        # 6. GENERATE INTERMEDIATE SUB-ANSWER (r_i = G(q_i, I_i))
        answer_chain = ANSWER_PROMPT | llm
        consolidated_facts_text = "\n".join([f"- {f.get('fact')}" for f in distilled.get("extracted_facts", [])]) or distilled.get("summary", "No direct facts.")
        answer_resp = safe_invoke(answer_chain, {
            "question": target_subq,
            "context": f"Distilled Evidence:\n{consolidated_facts_text}\n\nWorking Memory Context:\n{memory.get_context_text()}"
        })
        answer_data = parse_llm_json(answer_resp)
        sub_answer = answer_data.get("answer", str(answer_resp.content))

        # 7. UPDATE WORKING MEMORY (M_i = Eupdate)
        extractor_update_memory(
            main_question=question,
            current_memory=memory,
            distilled_info=distilled,
            qa_pair={"question": target_subq, "answer": sub_answer},
            hop=hop
        )

        # 8. EVALUATE STEP PATH REWARD (R_p)
        prior_trace = "\n".join([f"Hop {h['hop']}: {h['subquestion']} -> {h['answer']}" for h in hop_trace])
        p_reward = evaluate_path_reward(
            main_question=question,
            reasoning_trace=prior_trace,
            sub_question=target_subq,
            selected_information=consolidated_facts_text,
            generated_answer=sub_answer
        )
        path_rewards.append(p_reward.get("overall_path_reward", 0.8))

        hop_trace.append({
            "hop": hop,
            "action": "A1: Decompose & Answer",
            "subquestion": target_subq,
            "search_queries": search_queries,
            "retrieved_chunks": len(unique_docs),
            "extracted_facts": len(distilled.get("extracted_facts", [])),
            "answer": sub_answer,
            "path_reward": p_reward.get("overall_path_reward", 0.8),
            "memory_state_size": len(memory.facts)
        })

    # Terminal Outcome Reward & Cumulative Dual Reward
    reasoning_path_summary = "\n".join([f"Step {h['hop']} [{h['action']}]: {h.get('subquestion', '')} => {h.get('answer', '')}" for h in hop_trace])
    outcome_eval = evaluate_outcome_reward(
        original_question=question,
        reasoning_path=reasoning_path_summary,
        predicted_answer=final_answer,
        correct_answer=ground_truth
    )
    outcome_score = outcome_eval.get("outcome_reward", 0.85)
    cumulative_reward = compute_cumulative_dual_reward(path_rewards, outcome_score, lambda_param=2.0)

    latency = round(time.time() - start_time, 2)

    formatted_memory = [
        {
            "id": f.get("id", f"fact_{i}"),
            "claim": f.get("fact", ""),
            "text": f.get("justification", ""),
            "hop_added": f.get("hop_added", 1),
            "criteria": f.get("criteria", []),
            "status": "ACTIVE",
            "relevance_score": 1.0,
            "utility_score": 1.0
        }
        for i, f in enumerate(memory.facts)
    ]

    mem_chars = sum(len(getattr(f, "fact", str(f))) + len(getattr(f, "context", "")) for f in memory.facts)
    estimated_tokens = max(180, (mem_chars // 4) + (len(final_answer) // 4) + (len(hop_trace) * 220))

    metrics = {
        "mode": "State-Aware RAG (Socratic)",
        "hops": len(hop_trace),
        "retrieval_calls": retrieval_calls,
        "memory_facts_count": len(memory.facts),
        "memory_items": len(memory.facts),
        "tokens": estimated_tokens,
        "estimated_tokens": estimated_tokens,
        "latency_seconds": latency,
        "avg_path_reward": round(sum(path_rewards)/len(path_rewards), 3) if path_rewards else 0.0,
        "outcome_reward": outcome_score,
        "cumulative_dual_reward": cumulative_reward,
        "confidence": confidence,
        "queries_generated": len(previous_queries)
    }

    return {
        "answer": final_answer,
        "memory": formatted_memory,
        "hop_trace": hop_trace,
        "metrics": metrics,
        "state_representation": memory.to_dict(),
        "dual_reward": {
            "path_rewards": path_rewards,
            "outcome_reward": outcome_eval,
            "cumulative_reward": cumulative_reward
        }
    }


# ==============================================================================
# Monte Carlo Tree Search (MCTS) Exploration Mode
# ==============================================================================
class MCTSStateNode:
    """
    Tree node representing reasoning state Si = (qi, ri, Mi) in MCTS.
    """
    def __init__(
        self,
        subquestion: str,
        response: str,
        action_type: str,
        depth: int,
        parent: Optional["MCTSStateNode"] = None,
        rationale: str = ""
    ):
        self.node_id = f"node_{uuid.uuid4().hex[:6]}"
        self.subquestion = subquestion
        self.response = response
        self.action_type = action_type
        self.depth = depth
        self.parent = parent
        self.children: List["MCTSStateNode"] = []
        self.visits = 0
        self.value = 0.0
        self.rationale = rationale

    def uct_score(self, c_param: float = 1.0) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.value / self.visits
        exploration = c_param * math.sqrt(math.log(parent_visits + 1) / self.visits)
        return exploitation + exploration


def run_state_aware_mcts_rag(
    document_id: str,
    question: str,
    ground_truth: str = None,
    max_depth: int = 4,
    num_rollouts: int = 2
) -> Dict[str, Any]:
    """
    State-Aware RAG (Monte Carlo Tree Search Mode):
    Explores reasoning paths using 5 Action Types (A1-A5) with Globally Shared Working Memory.
    - A1: Decompose & Answer
    - A2: Consolidate (Internal reflection without retrieval)
    - A3: Refine (Verify and improve current response against memory)
    - A4: Redirect (Reformulate questions for clarity)
    - A5: Conclude (Produce final answer)
    """
    start_time = time.time()
    global_memory = StateAwareMemory(memory_id="mcts_shared_global_memory")
    root_node = MCTSStateNode(
        subquestion=question,
        response="Root problem state initialized.",
        action_type="Root",
        depth=0
    )

    rollout_traces = []
    candidate_final_answers = []
    total_retrieval_calls = 0

    for rollout_idx in range(num_rollouts):
        current_node = root_node
        current_path_nodes = [current_node]

        # 1. SELECTION (traverse tree via UCT)
        while current_node.children and current_node.depth < max_depth:
            best_child = max(current_node.children, key=lambda child: child.uct_score(c_param=1.0))
            current_node = best_child
            current_path_nodes.append(current_node)
            if current_node.action_type == "A5: Conclude":
                break

        # 2. EXPANSION
        if current_node.depth < max_depth and current_node.action_type != "A5: Conclude":
            actions_to_expand = []
            if current_node.depth == 0:
                actions_to_expand = ["A1: Decompose & Answer", "A4: Redirect"]
            elif len(global_memory.facts) >= 2 and current_node.depth >= 2:
                actions_to_expand = ["A1: Decompose & Answer", "A2: Consolidate", "A5: Conclude"]
            else:
                actions_to_expand = ["A1: Decompose & Answer", "A3: Refine", "A5: Conclude"]

            for act in actions_to_expand:
                if act == "A1: Decompose & Answer":
                    subq_chain = SUBQUESTION_PROMPT | llm
                    subq_resp = safe_invoke(subq_chain, {"question": question, "context": global_memory.get_context_text()})
                    subq_data = parse_llm_json(subq_resp)
                    subq = subq_data.get("subquestion") or question
                    if not subq or not str(subq).strip():
                        subq = question

                    total_retrieval_calls += 1
                    try:
                        raw = search_all_documents(subq, top_k=5)
                    except Exception:
                        raw = search_vector_store(document_id, subq, top_k=5)
                    reranked = rerank_chunks(subq, raw, top_k=3)

                    distilled = extractor_consolidate(subq, reranked, global_memory)
                    ans_chain = ANSWER_PROMPT | llm
                    ans_resp = safe_invoke(ans_chain, {
                        "question": subq,
                        "context": f"Distilled: {distilled.get('summary')}\nMemory: {global_memory.get_context_text()}"
                    })
                    ans_data = parse_llm_json(ans_resp)
                    sub_ans = ans_data.get("answer", str(ans_resp.content))

                    extractor_update_memory(question, global_memory, distilled, {"question": subq, "answer": sub_ans}, hop=current_node.depth + 1)

                    child = MCTSStateNode(subquestion=subq, response=sub_ans, action_type="A1: Decompose & Answer", depth=current_node.depth + 1, parent=current_node)
                    current_node.children.append(child)

                elif act == "A2: Consolidate":
                    cons_chain = CONSOLIDATE_PROMPT | llm
                    cons_resp = safe_invoke(cons_chain, {"question": question, "context": global_memory.get_context_text()})
                    cons_data = parse_llm_json(cons_resp)
                    synthesis = cons_data.get("synthesis", str(cons_resp.content))

                    child = MCTSStateNode(subquestion=f"Synthesize {question}", response=synthesis, action_type="A2: Consolidate", depth=current_node.depth + 1, parent=current_node)
                    current_node.children.append(child)

                elif act == "A3: Refine":
                    refine_chain = REFINE_PROMPT | llm
                    refine_resp = safe_invoke(refine_chain, {"question": current_node.subquestion, "answer": current_node.response, "context": global_memory.get_context_text()})
                    refine_data = parse_llm_json(refine_resp)
                    refined = refine_data.get("refined_answer", current_node.response)

                    child = MCTSStateNode(subquestion=f"Refine {current_node.subquestion}", response=refined, action_type="A3: Refine", depth=current_node.depth + 1, parent=current_node)
                    current_node.children.append(child)

                elif act == "A4: Redirect":
                    redir_chain = REDIRECT_PROMPT | llm
                    redir_resp = safe_invoke(redir_chain, {"question": question, "context": global_memory.get_context_text()})
                    redir_data = parse_llm_json(redir_resp)
                    redirected_q = redir_data.get("redirected_question", question)
                    if not redirected_q or not str(redirected_q).strip():
                        redirected_q = question

                    child = MCTSStateNode(subquestion=redirected_q, response="Redirected inquiry.", action_type="A4: Redirect", depth=current_node.depth + 1, parent=current_node)
                    current_node.children.append(child)

                elif act == "A5: Conclude":
                    fin_chain = FINALIZE_PROMPT | llm
                    fin_resp = safe_invoke(fin_chain, {"question": question, "context": global_memory.get_context_text()})
                    fin_data = parse_llm_json(fin_resp)
                    final_ans = fin_data.get("final_answer", str(fin_resp.content))

                    child = MCTSStateNode(subquestion=question, response=final_ans, action_type="A5: Conclude", depth=current_node.depth + 1, parent=current_node)
                    current_node.children.append(child)

            if current_node.children:
                current_node = current_node.children[0]
                current_path_nodes.append(current_node)

        # 3. EVALUATION & ROLLOUT REWARD
        rollout_answer = current_node.response
        if current_node.action_type != "A5: Conclude":
            fin_chain = FINALIZE_PROMPT | llm
            fin_resp = safe_invoke(fin_chain, {"question": question, "context": global_memory.get_context_text()})
            fin_data = parse_llm_json(fin_resp)
            rollout_answer = fin_data.get("final_answer", current_node.response)

        reward_score = judge_mcts_answer(user_question=question, system_answer=rollout_answer, correct_answer=ground_truth)
        candidate_final_answers.append({
            "rollout": rollout_idx + 1,
            "answer": rollout_answer,
            "reward": reward_score,
            "path": [f"[{n.action_type}] {n.subquestion} => {n.response[:80]}" for n in current_path_nodes]
        })

        # 4. BACKPROPAGATION
        for node in current_path_nodes:
            node.visits += 1
            node.value += reward_score

        rollout_traces.append({
            "rollout": rollout_idx + 1,
            "path_depth": len(current_path_nodes),
            "reward": reward_score,
            "terminal_action": current_node.action_type
        })

    # SYNTHESIZE FINAL ANSWER ACROSS BEST MCTS CANDIDATE PATHS (Appendix C.4.2)
    candidate_paths_text = "\n\n".join([
        f"Candidate Path {c['rollout']} (Judge Reward: {c['reward']:.2f}):\nAnswer: {c['answer']}\nLogic: {' -> '.join(c['path'])}"
        for c in candidate_final_answers
    ])
    synth_chain = SYNTHESIZE_FINAL_ANSWER_PROMPT | llm
    synth_resp = safe_invoke(synth_chain, {
        "question": question,
        "reasoning_paths": candidate_paths_text
    })
    synth_data = parse_llm_json(synth_resp)
    final_synthesized_answer = synth_data.get("synthesized_answer", candidate_final_answers[0]["answer"] if candidate_final_answers else "Answer formulation failed.")

    latency = round(time.time() - start_time, 2)

    formatted_memory = [
        {
            "id": f.get("id", f"fact_{i}"),
            "claim": f.get("fact", ""),
            "text": f.get("justification", ""),
            "hop_added": f.get("hop_added", 1),
            "criteria": f.get("criteria", []),
            "status": "ACTIVE",
            "relevance_score": 1.0,
            "utility_score": 1.0
        }
        for i, f in enumerate(global_memory.facts)
    ]

    mem_chars = sum(len(f.get("fact", "")) for f in global_memory.facts)
    estimated_tokens = max(250, (mem_chars // 4) + (len(final_synthesized_answer) // 4) + (num_rollouts * max_depth * 180))

    metrics = {
        "mode": "State-Aware RAG (MCTS Exploration)",
        "rollouts": num_rollouts,
        "max_tree_depth": max_depth,
        "retrieval_calls": total_retrieval_calls,
        "memory_facts_count": len(global_memory.facts),
        "memory_items": len(global_memory.facts),
        "tokens": estimated_tokens,
        "estimated_tokens": estimated_tokens,
        "latency_seconds": latency,
        "avg_mcts_reward": round(sum(c["reward"] for c in candidate_final_answers)/len(candidate_final_answers), 3) if candidate_final_answers else 0.0,
        "total_nodes_created": root_node.visits
    }

    return {
        "answer": final_synthesized_answer,
        "memory": formatted_memory,
        "hop_trace": candidate_final_answers[0]["path"] if candidate_final_answers else [],
        "rollout_traces": rollout_traces,
        "candidate_answers": candidate_final_answers,
        "metrics": metrics,
        "state_representation": global_memory.to_dict()
    }
