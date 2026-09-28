from app.utils.embeddings import get_cached_embedding, compute_cosine_similarity
from typing import List, Dict
import os

EVIDENCE_SIMILARITY_THRESHOLD = float(
    os.getenv("EVIDENCE_SIMILARITY_THRESHOLD", "0.0")
)

def filter_chunks(
    query: str,
    chunks: List[Dict],
    threshold: float = EVIDENCE_SIMILARITY_THRESHOLD
) -> List[Dict]:
    if not chunks:
        return []

    query_emb = get_cached_embedding(query)

    filtered_chunks = []
    for chunk in chunks:
        chunk["embedding"] = get_cached_embedding(
            chunk["text"]
        )

        sim = compute_cosine_similarity(
            query_emb,
            chunk["embedding"]
        )
        chunk["semantic_score"] = sim
        
        print(f"Chunk={chunk.get('chunk_id', 'unknown')} Similarity={sim:.4f} Threshold={threshold:.2f}")

        if sim >= threshold:
            filtered_chunks.append(chunk)

    return filtered_chunks
