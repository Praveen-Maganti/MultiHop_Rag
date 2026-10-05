import faiss
import numpy as np
from typing import List, Dict
from app.utils.embeddings import get_embedding, get_embeddings_batch

# In-memory store: {document_id: {"index": faiss_index, "chunks": [chunk_dicts]}}
vector_db = {}

def initialize_vector_store(document_id: str, chunks: List[Dict]):
    """
    Initializes a FAISS index for a specific document and adds chunks.
    """
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings_batch(texts)

    embedding_matrix = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embedding_matrix)
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    vector_db[document_id] = {
        "index": index,
        "chunks": chunks
    }

def search_vector_store(document_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    Searches the FAISS index for the given document and query.
    """
    if document_id not in vector_db:
        raise ValueError(f"Document {document_id} not found in vector store.")

    store = vector_db[document_id]
    index = store["index"]
    chunks = store["chunks"]

    query_embedding = np.array([get_embedding(query)]).astype('float32')
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(I[0]):
        if idx != -1 and idx < len(chunks):
            chunk_copy = chunks[idx].copy()
            chunk_copy["score"] = float(D[0][i])
            chunk_copy["distance"] = 1.0 - float(D[0][i])
            results.append(chunk_copy)

    return results

def search_all_documents(query: str, top_k: int = 5) -> List[Dict]:
    """
    Searches ALL indexed documents and returns the top_k results globally.
    Required for multi-hop RAG that spans multiple uploaded files.
    """
    if not vector_db:
        return []

    query_embedding = np.array([get_embedding(query)]).astype('float32')
    faiss.normalize_L2(query_embedding)

    all_results = []
    for doc_id, store in vector_db.items():
        index = store["index"]
        chunks = store["chunks"]
        k = min(top_k, len(chunks))
        D, I = index.search(query_embedding, k)
        for i, idx in enumerate(I[0]):
            if idx != -1 and idx < len(chunks):
                chunk_copy = chunks[idx].copy()
                chunk_copy["score"] = float(D[0][i])
                chunk_copy["distance"] = 1.0 - float(D[0][i])
                chunk_copy["source_document_id"] = doc_id
                all_results.append(chunk_copy)

    # Sort globally by score, return top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]

def get_indexed_document_ids() -> List[str]:
    """Returns all currently indexed document IDs."""
    return list(vector_db.keys())
