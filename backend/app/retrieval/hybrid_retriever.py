"""
Hybrid Retriever Engine
Supports Dense Vector Retrieval (FAISS), Lexical Retrieval (BM25 token-rank),
Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking, and Memory-Aware Querying.
"""

import math
import re
from typing import List, Dict, Any, Optional
from app.core.config import get_config
from app.retrieval.vector_store import search_all_documents, search_vector_store, vector_db
from app.retrieval.reranker import rerank_chunks
from app.core.state import Evidence, ReasoningState


def tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def compute_bm25_score(query_tokens: List[str], doc_tokens: List[str], avg_doc_len: float, total_docs: int, df: Dict[str, int]) -> float:
    k1 = 1.5
    b = 0.75
    doc_len = len(doc_tokens)
    score = 0.0
    doc_freq_counts = {}
    for t in doc_tokens:
        doc_freq_counts[t] = doc_freq_counts.get(t, 0) + 1

    for q in query_tokens:
        if q not in doc_freq_counts:
            continue
        tf = doc_freq_counts[q]
        n_q = df.get(q, 1)
        idf = math.log((total_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / (avg_doc_len + 1e-6)))
        score += idf * (numerator / denominator)
    return score


class HybridRetriever:
    """
    Unified, memory-aware hybrid retrieval interface.
    Combines dense FAISS search, lexical token BM25 ranking, and Cross-Encoder reranking.
    """
    def __init__(self):
        self.config = get_config().retrieval

    def lexical_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Performs lexical BM25 ranking over all indexed chunks in vector_db."""
        if not vector_db:
            return []

        all_chunks = []
        for doc_id, store in vector_db.items():
            for c in store.get("chunks", []):
                copy_c = dict(c)
                copy_c["source_document_id"] = doc_id
                all_chunks.append(copy_c)

        if not all_chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return all_chunks[:top_k]

        # Calculate document frequency
        df: Dict[str, int] = {}
        tokenized_docs = []
        for c in all_chunks:
            tokens = tokenize(c.get("text", ""))
            tokenized_docs.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        avg_len = sum(len(t) for t in tokenized_docs) / len(tokenized_docs)
        total_docs = len(all_chunks)

        scored = []
        for idx, c in enumerate(all_chunks):
            s = compute_bm25_score(query_tokens, tokenized_docs[idx], avg_len, total_docs, df)
            c["lexical_score"] = float(s)
            scored.append(c)

        scored.sort(key=lambda x: x.get("lexical_score", 0), reverse=True)
        return scored[:top_k]

    def dense_search(self, query: str, document_id: Optional[str] = None, top_k: int = 8) -> List[Dict[str, Any]]:
        """Dense FAISS retrieval."""
        try:
            if document_id and document_id in vector_db:
                results = search_vector_store(document_id, query, top_k=top_k)
            else:
                results = search_all_documents(query, top_k=top_k)
            for r in results:
                r["dense_score"] = float(r.get("score", 0.0))
            return results
        except Exception:
            return []

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]],
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """Merges dense and lexical rankings via RRF."""
        rrf_constant = self.config.rrf_k
        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item.get("chunk_id", str(item.get("text", "")[:30]))
            chunk_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_constant + rank + 1))

        for rank, item in enumerate(lexical_results):
            cid = item.get("chunk_id", str(item.get("text", "")[:30]))
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_constant + rank + 1))

        sorted_cids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        fused_chunks = []
        for cid in sorted_cids[:top_k]:
            c = chunk_lookup[cid]
            c["rrf_score"] = round(fused_scores[cid], 5)
            fused_chunks.append(c)
        return fused_chunks

    def retrieve(
        self,
        query: str,
        document_id: Optional[str] = None,
        state: Optional[ReasoningState] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Main Retrieval Pipeline:
        1. Query Expansion/Contextualization using state (if present)
        2. Hybrid Dense + Lexical retrieval
        3. RRF Fusion
        4. Cross-Encoder Reranking
        """
        # 1. Fetch dense and lexical candidates
        dense_cand = self.dense_search(query, document_id, top_k=self.config.top_k_dense)
        lexical_cand = self.lexical_search(query, top_k=self.config.top_k_lexical)

        # 2. Fuse candidates
        fused = self.reciprocal_rank_fusion(dense_cand, lexical_cand, top_k=self.config.top_k_dense + self.config.top_k_lexical)

        # 3. Cross-Encoder Rerank
        reranked = rerank_chunks(query, fused, top_k=top_k)
        return reranked
