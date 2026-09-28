from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.retrieval.vector_store import search_vector_store
import time

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
    max_tokens=1024,
)

prompt_template = """
You are a helpful and precise assistant. Use the following context to answer the user's question.
If the answer cannot be found in the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def run_standard_rag(document_id: str, question: str, top_k: int = 5):
    start_time = time.time()
    
    retrieval_calls = 1
    evidence_chunks = search_vector_store(document_id, question, top_k=top_k)
    
    context_text = "\n\n".join([f"Chunk {c['chunk_id']}:\n{c['text']}" for c in evidence_chunks])
    
    chain = PROMPT | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    content_obj = response.content
    if isinstance(content_obj, list):
        ans_text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content_obj])
    else:
        ans_text = str(content_obj)
        
    end_time = time.time()
    latency = end_time - start_time
    
    formatted_evidence = [{"id": chunk["chunk_id"], "text": chunk["text"]} for chunk in evidence_chunks]
        
    metrics = {
        "retrieval_calls": retrieval_calls,
        "tokens": len(context_text) // 4 + len(ans_text) // 4,
        "latency": f"{latency:.2f}s"
    }
    
    return {
        "answer": ans_text,
        "evidence": formatted_evidence,
        "metrics": metrics,
        "evaluation": {
             "f1": "--",
             "coverage": "--",
             "unsupported": "--"
        }
    }
