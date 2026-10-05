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

# Multi-Hop RAG: iterative retrieval with CONSTANT flat memory.
# All retrieved chunks are accumulated without any scoring, eviction, or
# adaptive management. The LLM reasons over the full growing context each hop.
multi_hop_prompt = PromptTemplate(
    template="""
You are a multi-hop reasoning assistant. Answer the question using ONLY the provided context chunks.
Do NOT use any outside knowledge.

Question:
{question}

Accumulated Context (all chunks retrieved so far across all hops):
{context}

Previous Search Queries:
{previous_queries}

INSTRUCTIONS:
1. Check if the context already contains enough information to fully answer the question.
2. If YES — output {{"action": "ANSWER", "answer": "<your answer>"}}
3. If NO — output {{"action": "SEARCH", "query": "<one specific sub-question for missing info>"}}
4. Never repeat a previous search query.
5. Never use knowledge outside the provided context.

Return ONLY a valid JSON object. No markdown, no explanation.
""",
    input_variables=["question", "context", "previous_queries"]
)


def run_multi_hop_rag(document_id: str, question: str, max_hops: int = 3):
    """
    Multi-Hop RAG with CONSTANT FLAT MEMORY.

    Architecture:
    - Each hop retrieves top-10 chunks via cross-document FAISS search.
    - Reranks to top-3 using CrossEncoder.
    - Appends ALL new unique chunks to a flat cumulative_context list.
    - NO scoring, NO eviction, NO adaptive management — memory only grows.
    - LLM receives the full accumulated context every hop and decides
      SEARCH (next hop) or ANSWER (terminate).
    - Duplicate query detection prevents infinite loops.
    """
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_K = 3

    start_time = time.time()

    # Constant flat memory — only appended, never pruned
    cumulative_context = []
    previous_queries = []
    retrieval_calls = 0
    current_query = question
    final_answer = "Could not formulate an answer within the maximum number of hops."
    chain = multi_hop_prompt | llm
    hop_trace = []
    total_input_tokens = 0
    total_output_tokens = 0
    decision = {"action": "STOP"}

    for hop in range(max_hops):
        # 1. RETRIEVE across all documents (cross-document multi-hop)
        retrieval_calls += 1
        try:
            raw_chunks = search_all_documents(current_query, top_k=RETRIEVAL_TOP_K)
        except Exception:
            raw_chunks = search_vector_store(document_id, current_query, top_k=RETRIEVAL_TOP_K)

        # 2. RERANK top-10 → top-3 using CrossEncoder
        reranked = rerank_chunks(current_query, raw_chunks, top_k=RERANK_TOP_K)

        if not reranked:
            hop_trace.append({
                "hop": hop + 1,
                "query": current_query,
                "chunks_retrieved": 0,
                "chunks_added": 0,
                "total_context_size": len(cumulative_context),
                "action": "STOP",
                "next_query": None,
                "stop_reason": "no_useful_chunks"
            })
            final_answer = "Stopped: No relevant chunks found."
            break

        # 3. APPEND new unique chunks to flat constant memory (no eviction)
        added = 0
        for c in reranked:
            if not any(existing["chunk_id"] == c["chunk_id"] for existing in cumulative_context):
                c["hop_added"] = hop + 1
                cumulative_context.append(c)
                added += 1

        if current_query not in previous_queries:
            previous_queries.append(current_query)

        # 4. BUILD full context string from ALL accumulated chunks
        context_text = "\n\n".join(
            [f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in cumulative_context]
        )
        queries_text = "\n".join(previous_queries)

        # 5. LLM REASONING over full flat context
        response = chain.invoke({
            "question": question,
            "context": context_text,
            "previous_queries": queries_text
        })

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_input_tokens += response.usage_metadata.get("input_tokens", 0)
            total_output_tokens += response.usage_metadata.get("output_tokens", 0)

        content_obj = response.content
        if isinstance(content_obj, list):
            output_text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content_obj])
        else:
            output_text = str(content_obj)

        text = output_text.strip().replace("```json", "").replace("```", "").strip()
        try:
            decision = json.loads(text)
        except Exception:
            decision = {"action": "ANSWER", "answer": text}

        action = decision.get("action", "ANSWER")
        next_query = decision.get("query", "").strip() if action == "SEARCH" else None

        mem_chars = sum(len(c.get("text", "")) for c in cumulative_context)
        mem_tokens = mem_chars // 4

        hop_trace.append({
            "hop": hop + 1,
            "query": current_query,
            "chunks_retrieved": len(reranked),
            "chunks_added": added,
            "total_context_size": len(cumulative_context),
            "memory_items_count": len(cumulative_context),
            "memory_chars": mem_chars,
            "memory_tokens_est": mem_tokens,
            "action": action,
            "next_query": next_query,
            "stop_reason": None
        })

        # 6. TERMINATION
        if action == "ANSWER":
            final_answer = decision.get("answer", "")
            hop_trace[-1]["stop_reason"] = "sufficient_evidence"
            break
        elif action == "SEARCH" and next_query:
            is_duplicate = is_semantically_similar_query(next_query, previous_queries, threshold=0.90)
            if next_query in previous_queries or is_duplicate:
                hop_trace[-1]["stop_reason"] = "repeated_query"
                final_answer = "Stopped: Repeated search query detected."
                break
            current_query = next_query
        else:
            hop_trace[-1]["stop_reason"] = "fallback"
            final_answer = str(decision)
            break

    if hop_trace and not hop_trace[-1].get("stop_reason"):
        hop_trace[-1]["stop_reason"] = "max_hops_reached"

    end_time = time.time()
    latency = end_time - start_time

    estimated_tokens = total_input_tokens + total_output_tokens
    if estimated_tokens == 0:
        estimated_tokens = len(context_text) // 4 + len(final_answer) // 4 if cumulative_context else 0

    metrics = {
        "retrieval_calls": retrieval_calls,
        "estimated_tokens": estimated_tokens,
        "latency_seconds": round(latency, 2),
        "hops": len(hop_trace),
        "unique_chunks": len(cumulative_context),
        "queries_generated": len(previous_queries),
        "stop_reason": hop_trace[-1].get("stop_reason", "unknown") if hop_trace else "unknown"
    }

    # Format evidence for frontend (same shape as proposed RAG memory items)
    formatted_evidence = [
        {"id": c["chunk_id"], "text": c["text"], "score": round(c.get("score", 0.0), 4)}
        for c in cumulative_context
    ]

    # Flat working memory — same chunks, tagged with hop_added, no scores/eviction
    flat_memory = [
        {
            "id": c["chunk_id"],
            "text": c["text"],
            "score": round(c.get("score", 0.0), 4),
            "hop_added": c.get("hop_added", 1),
            "status": "ACTIVE"
        }
        for c in cumulative_context
    ]

    return {
        "answer": final_answer,
        "evidence": formatted_evidence,
        "memory": flat_memory,
        "hop_trace": hop_trace,
        "metrics": metrics,
        "evaluation": {"f1": "--", "coverage": "--", "unsupported": "--"}
    }