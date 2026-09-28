from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.retrieval.vector_store import search_all_documents, search_vector_store
from app.retrieval.reranker import rerank_chunks
from app.utils.embeddings import is_semantically_similar_query
import time
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

reasoning_prompt = PromptTemplate(
    template="""
You are an advanced reasoning agent equipped with an Evidence- and Utility-Aware Working Memory.
Your goal is to answer the question using ONLY the provided memory.

Question:
{question}

Working Memory:
{memory_text}

Previous Search Queries:
{previous_queries}

INSTRUCTIONS:
1. Identify which parts of the question are already answered by the Working Memory.
2. Identify any missing information required to fully answer the question.
3. If you have sufficient evidence to answer the question, output a JSON object with action "ANSWER" and provide the final answer. Answer concisely using only facts from memory.
4. If you need more information, generate exactly ONE specific query for the missing information. Output a JSON object with action "SEARCH" and the new query.
5. If the memory contains conflicting information, explicitly address the conflict in your answer.
6. Do not search for information already present. Do not repeat previous searches. Do not use outside knowledge.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching one of these two structures, with no markdown formatting:
{{"action": "SEARCH", "query": "..."}}
OR
{{"action": "ANSWER", "answer": "..."}}
""",
    input_variables=["question", "memory_text", "previous_queries"]
)


def run_proposed_rag(document_id: str, question: str, max_hops: int = 3, max_memory_items: int = 15):
    from app.memory.working_memory import WorkingMemory
    from app.memory.memory_selector import select_evidence
    from app.memory.memory_compressor import compress_memory
    from app.retrieval.semantic_filter import filter_chunks

    start_time = time.time()

    memory = WorkingMemory()
    from app.tracing.reasoning_trace import ReasoningTrace
    trace = ReasoningTrace()

    previous_queries = []
    retrieval_calls = 0
    current_query = question
    final_answer = "Could not formulate an answer within the maximum number of hops."
    chain = reasoning_prompt | llm

    metrics_dict = {
        "input_chunks": 0,
        "filtered_chunks": 0,
        "removed_chunks": 0,
        "evidence_llm_calls": 0,
        "total_claims": 0,
        "possible_conflict_pairs": 0,
        "candidate_conflict_pairs": 0,
        "conflict_llm_calls": 0,
        "conflict_calls_saved": 0,
        "duplicates_removed": 0,
        "claims_merged": 0,
        "low_utility_removed": 0
    }

    hop_trace = []
    total_input_tokens = 0
    total_output_tokens = 0
    decision = {"action": "STOP"}

    for hop in range(max_hops):
        trace.start_hop(hop + 1, current_query)
        items_before_retrieval = len(memory.items)

        # 1. RETRIEVE across all documents (multi-hop requires cross-document search)
        retrieval_calls += 1
        try:
            raw_chunks = search_all_documents(current_query, top_k=10)
        except Exception:
            # Fallback to single-document search
            raw_chunks = search_vector_store(document_id, current_query, top_k=10)

        reranked_chunks = rerank_chunks(current_query, raw_chunks, top_k=3)

        # 2. SEMANTIC FILTER
        metrics_dict["input_chunks"] += len(reranked_chunks)
        new_chunks = filter_chunks(current_query, reranked_chunks)

        print(f"\n=== HOP {hop + 1} ===")
        print(f"Query: {current_query}")
        print(f"RAW: {len(raw_chunks)} | RERANKED: {len(reranked_chunks)} | FILTERED: {len(new_chunks)}")

        metrics_dict["filtered_chunks"] += len(new_chunks)
        metrics_dict["removed_chunks"] += (len(reranked_chunks) - len(new_chunks))

        trace.record_retrieval(len(raw_chunks), len(reranked_chunks), len(new_chunks), new_chunks)
        trace.record_filtering(len(reranked_chunks), len(new_chunks), len(reranked_chunks) - len(new_chunks))

        hop_evidence_summary = []
        if new_chunks:
            for c in new_chunks:
                hop_evidence_summary.append(c.get("text", "")[:120])

        # 3. NO CHUNKS → STOP
        if not new_chunks:
            trace.record_decision("STOP", stop_reason="no_useful_chunks")
            trace.record_timing(time.time() - start_time)
            trace.finalize_hop()
            if not hop_trace:
                final_answer = "Stopped early: No relevant chunks found for query."
            break

        # 4. SELECT EVIDENCE → MEMORY
        select_evidence(new_chunks, current_query, memory, current_hop=hop + 1, metrics=metrics_dict)

        active_items = [i for i in memory.items if i.status in ["ACTIVE", "VERIFIED"]]
        trace.record_memory_update(items_before_retrieval, len(new_chunks), len(active_items))

        print("Memory after hop:")
        for item in active_items:
            print(f"  - {item.claim}")

        has_conflict = any(item.conflict_score > 0.5 for item in active_items)
        trace.record_conflicts(has_conflict, metrics_dict)

        reasoning_state = {
            "question": question,
            "current_hop": hop + 1,
            "max_hops": max_hops,
            "current_query": current_query,
            "previous_queries": previous_queries,
            "remaining_hops": max_hops - hop
        }

        compression_metrics = compress_memory(memory, reasoning_state, max_memory_items)
        trace.record_compression(compression_metrics)

        if "compression_ratio" in compression_metrics:
            metrics_dict["duplicates_removed"] += compression_metrics.get("duplicates_removed", 0)
            metrics_dict["claims_merged"] += compression_metrics.get("claims_merged", 0)
            metrics_dict["low_utility_removed"] += compression_metrics.get("low_utility_removed", 0)

        # 5. UPDATE PREVIOUS QUERIES
        if current_query not in previous_queries:
            previous_queries.append(current_query)

        queries_text = "\n".join(previous_queries) if previous_queries else "None"

        # 6. LLM REASONING
        response = chain.invoke({
            "question": question,
            "memory_text": memory.get_all_text(),
            "previous_queries": queries_text
        })

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_input_tokens += response.usage_metadata.get('input_tokens', 0)
            total_output_tokens += response.usage_metadata.get('output_tokens', 0)

        content_obj = response.content
        if isinstance(content_obj, list):
            output_text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content_obj])
        else:
            output_text = str(content_obj)

        text = output_text.strip().replace("```json", "").replace("```", "").strip()
        try:
            decision = json.loads(text)
        except Exception as e:
            trace.record_error("JSON_PARSE_ERROR", f"LLM returned invalid JSON: {str(e)}")
            decision = {"action": "ANSWER", "answer": text}

        # Record this hop in the hop_trace
        hop_trace.append({
            "hop": hop + 1,
            "query": current_query,
            "evidence": hop_evidence_summary,
            "action": decision.get("action", "UNKNOWN"),
            "next_query": decision.get("query") if decision.get("action") == "SEARCH" else None
        })

        # 7. TERMINATION
        if decision.get("action", "ANSWER") == "ANSWER":
            final_answer = decision.get("answer", "")
            trace.record_decision("ANSWER", stop_reason="sufficient_evidence", answer=final_answer)
            trace.record_timing(time.time() - start_time)
            trace.finalize_hop()
            break
        elif decision.get("action") == "SEARCH":
            next_query = decision.get("query", "").strip()
            is_duplicate = is_semantically_similar_query(next_query, previous_queries, threshold=0.90)

            if next_query in previous_queries or is_duplicate:
                trace.record_decision("STOP", stop_reason="repeated_query")
                trace.record_timing(time.time() - start_time)
                trace.finalize_hop()
                final_answer = "Stopped early: Model attempted to repeat a semantically similar search query."
                break

            trace.record_decision("SEARCH", next_query=next_query)
            trace.record_timing(time.time() - start_time)
            trace.finalize_hop()
            current_query = next_query
        else:
            final_answer = str(decision)
            trace.record_decision("STOP", stop_reason="fallback")
            trace.record_timing(time.time() - start_time)
            trace.finalize_hop()
            break

    if trace.hops and not trace.hops[-1].get("decision", {}).get("stop_reason"):
        trace.hops[-1]["decision"]["stop_reason"] = "max_hops_reached"

    end_time = time.time()
    latency = end_time - start_time

    estimated_tokens = total_input_tokens + total_output_tokens
    if estimated_tokens == 0:
        total_memory_chars = sum([len(item.text) for item in memory.items])
        estimated_tokens = total_memory_chars // 4 + len(final_answer) // 4

    if metrics_dict["input_chunks"] > 0:
        metrics_dict["filter_ratio"] = round(metrics_dict["removed_chunks"] / metrics_dict["input_chunks"], 2)
    else:
        metrics_dict["filter_ratio"] = 0.0

    metrics = {
        "retrieval_calls": retrieval_calls,
        "estimated_tokens": estimated_tokens,
        "latency_seconds": round(latency, 2),
        "hops": hop + 1,
        "memory_items": len(memory.items),
        "queries_generated": len(previous_queries),
        "answer_generated": (decision.get("action") == "ANSWER"),
        "stop_reason": trace.hops[-1]["decision"].get("stop_reason", "unknown") if trace.hops else "unknown"
    }
    metrics.update(metrics_dict)

    return {
        "answer": final_answer,
        "memory": memory.to_dict()["items"],
        "hop_trace": hop_trace,
        "reasoning_trace": trace.to_list(),
        "metrics": metrics
    }
