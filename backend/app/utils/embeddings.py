from sentence_transformers import SentenceTransformer
import os

model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# We load the model once when this module is imported to save time later
try:
    embedding_model = SentenceTransformer(model_name)
except Exception as e:
    print(f"Warning: Could not load embedding model {model_name}. Error: {e}")
    embedding_model = None

def get_embedding(text: str) -> list:
    if not embedding_model:
        raise RuntimeError("Embedding model not loaded properly.")
    return embedding_model.encode(text).tolist()

query_cache = {}
embedding_cache = {}

def get_cached_embedding(text: str) -> list:
    if text not in embedding_cache:
        embedding_cache[text] = get_embedding(text)
    return embedding_cache[text]

def compute_cosine_similarity(vec1: list, vec2: list) -> float:
    import numpy as np
    v1 = np.array(vec1).astype('float32')
    v2 = np.array(vec2).astype('float32')
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def is_semantically_similar_query(new_query: str, previous_queries: list, threshold: float = 0.90) -> bool:
    if not previous_queries:
        return False
        
    if new_query not in query_cache:
        query_cache[new_query] = get_embedding(new_query)
    new_vec = query_cache[new_query]
    
    for q in previous_queries:
        if q not in query_cache:
            query_cache[q] = get_embedding(q)
        q_vec = query_cache[q]
        
        sim = compute_cosine_similarity(new_vec, q_vec)
        if sim >= threshold:
            return True
    return False
