from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.retrieval.vector_store import search_vector_store
from app.retrieval.reranker import rerank_chunks
from app.utils.embeddings import is_semantically_similar_query
import time
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

multi_hop_prompt = PromptTemplate(
    template="""
You are an advanced reasoning assistant. Your goal is to answer the question using ONLY the provided context.

Question:
{question}

Context:
{context}

Previous Search Queries:
{previous_queries}

INSTRUCTIONS:
1. Identify which parts of the question are already answered by the Context.
2. Identify any missing information required to fully answer the question.
3. If you have sufficient evidence to answer the question, or if you cannot find it after multiple searches, output a JSON object with action "ANSWER" and provide the final answer.
4. If you need more information, generate exactly ONE specific query for the missing information. Output a JSON object with action "SEARCH" and the new query.
5. Do not search for information already present. Do not repeat previous searches. Do not use outside knowledge.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching one of these two structures, with no markdown formatting:
{{"action": "SEARCH", "query": "..."}}
OR
{{"action": "ANSWER", "answer": "..."}}
""",
    input_variables=["question", "context", "previous_queries"]
)

def run_multi_hop_rag(document_id: str, question: str, max_hops: int = 3):
    # Phase 2 configurations
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_K = 3
    MAX_EVIDENCE_CHUNKS = 12
    
    start_time = time.time()
    
    cumulative_context = []
    previous_queries = []
    retrieval_calls = 0
    current_query = question
    final_answer = "Could not formulate an answer within the maximum number of hops."
    chain = multi_hop_prompt | llm
    
    reasoning_trace = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for hop in range(max_hops):
        hop_record = {
            "hop": hop + 1,
            "query": current_query,
            "retrieved_chunks": [],
            "decision": None,
            "next_query": None,
            "stop_reason": None
        }
        
        # 1. RETRIEVE & RERANK
        retrieval_calls += 1
        raw_chunks = search_vector_store(document_id, current_query, top_k=RETRIEVAL_TOP_K)
        
        # Rerank Top 10 -> Top 3
        new_chunks = rerank_chunks(current_query, raw_chunks, top_k=RERANK_TOP_K)
        
        # 2. HANDLE EMPTY RETRIEVAL
        if not new_chunks:
            hop_record["stop_reason"] = "no_useful_chunks"
            reasoning_trace.append(hop_record)
            final_answer = "Stopped early: No relevant chunks found for query."
            break
            
        for c in new_chunks:
            hop_record["retrieved_chunks"].append(c["chunk_id"])
            if not any(existing['chunk_id'] == c['chunk_id'] for existing in cumulative_context):
                cumulative_context.append(c)
                
        # MAX CONTEXT LIMIT: If over 12 chunks, rerank all and keep top 8
        if len(cumulative_context) > MAX_EVIDENCE_CHUNKS:
            cumulative_context = rerank_chunks(question, cumulative_context, top_k=8)
            
        # 3. UPDATE PREVIOUS QUERIES
        if current_query not in previous_queries:
            previous_queries.append(current_query)
            
        context_text = "\\n\\n".join([f"Chunk {c['chunk_id']}:\\n{c['text']}" for c in cumulative_context])
        queries_text = "\\n".join(previous_queries) if previous_queries else "None"
        
        # 4. CALL LLM (Structured Output)
        response = chain.invoke({
            "question": question, 
            "context": context_text,
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
        except:
            decision = {"action": "ANSWER", "answer": text}
            
        hop_record["decision"] = decision.get("action", "ANSWER")
        
        # 5. TERMINATION CONDITIONS
        if hop_record["decision"] == "ANSWER":
            final_answer = decision.get("answer", "")
            hop_record["stop_reason"] = "sufficient_evidence"
            reasoning_trace.append(hop_record)
            break
        elif hop_record["decision"] == "SEARCH":
            next_query = decision.get("query", "").strip()
            hop_record["next_query"] = next_query
            
            # Semantic Duplicate Query Detection
            is_semantic_duplicate = is_semantically_similar_query(next_query, previous_queries, threshold=0.90)
            
            if next_query in previous_queries or is_semantic_duplicate:
                hop_record["stop_reason"] = "repeated_query"
                reasoning_trace.append(hop_record)
                final_answer = "Stopped early: Model attempted to repeat a semantically similar search query."
                break
                
            current_query = next_query
            reasoning_trace.append(hop_record)
        else:
            final_answer = str(decision)
            hop_record["stop_reason"] = "fallback"
            reasoning_trace.append(hop_record)
            break
            
    if hop == max_hops - 1 and reasoning_trace and not reasoning_trace[-1].get("stop_reason"):
         reasoning_trace[-1]["stop_reason"] = "max_hops_reached"
         
    end_time = time.time()
    latency = end_time - start_time
    
    estimated_tokens = total_input_tokens + total_output_tokens
    if estimated_tokens == 0:
        estimated_tokens = len(context_text) // 4 + len(final_answer) // 4 if 'context_text' in locals() else 0
        
    metrics = {
        "retrieval_calls": retrieval_calls,
        "estimated_tokens": estimated_tokens,
        "latency_seconds": round(latency, 2),
        "hops": hop + 1,
        "unique_chunks": len(cumulative_context),
        "queries_generated": len(previous_queries),
        "answer_generated": (decision.get("action") == "ANSWER" if 'decision' in locals() else False),
        "stop_reason": reasoning_trace[-1].get("stop_reason") if reasoning_trace else "unknown"
    }
    
    return {
        "answer": final_answer,
        "evidence": cumulative_context,
        "reasoning_trace": reasoning_trace,
        "metrics": metrics,
        "evaluation": {
             "f1": "--",
             "coverage": "--",
             "unsupported": "--"
        }
    }