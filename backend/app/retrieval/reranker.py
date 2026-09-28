from typing import List, Dict

try:
    from sentence_transformers import CrossEncoder
    # Load a small fast cross-encoder model
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    print(f"Warning: Could not load CrossEncoder model. Error: {e}")
    reranker_model = None

def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Reranks retrieved chunks using a cross-encoder model.
    """
    if not chunks:
        return []
        
    if reranker_model is None:
        # Fallback to vector search order if reranker is not available
        return chunks[:top_k]
        
    pairs = [[query, chunk["text"]] for chunk in chunks]
    try:
        scores = reranker_model.predict(pairs)
        
        # Attach rerank score and sort
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
            
        chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return chunks[:top_k]
    except Exception as e:
        print(f"Error during reranking: {e}")
        return chunks[:top_k]
