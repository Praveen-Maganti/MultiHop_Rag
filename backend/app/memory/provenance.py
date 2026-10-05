"""
Provenance Tracking Engine
Provides end-to-end evidence lineage tracking and citation generation.
"""

from typing import List, Dict, Any, Optional
from app.core.state import Provenance, MemoryItem, Evidence


class ProvenanceTracker:
    """
    Tracks and links memory claims back to source documents, chunks,
    retrieval ranks, queries, and verification records.
    """
    def __init__(self):
        self.provenance_graph: Dict[str, List[Provenance]] = {}

    def record_provenance(
        self,
        item_id: str,
        source_doc: str,
        chunk_id: str,
        query: str,
        rank: int,
        hop: int,
        retriever_score: float = 0.0,
        reranker_score: float = 0.0,
        url: Optional[str] = None
    ) -> Provenance:
        prov = Provenance(
            source_document=source_doc,
            source_url=url,
            retrieval_query=query,
            retrieval_rank=rank,
            hop_number=hop,
            chunk_id=chunk_id,
            retriever_score=retriever_score,
            reranker_score=reranker_score,
            verification_status="UNVERIFIED"
        )
        if item_id not in self.provenance_graph:
            self.provenance_graph[item_id] = []
        self.provenance_graph[item_id].append(prov)
        return prov

    def format_citations_for_answer(self, active_items: List[MemoryItem]) -> List[Dict[str, Any]]:
        citations = []
        for idx, item in enumerate(active_items):
            for p in item.provenance:
                citations.append({
                    "citation_id": f"[{idx+1}]",
                    "claim": item.content,
                    "document": p.source_document,
                    "chunk_id": p.chunk_id,
                    "hop": p.hop_number,
                    "query": p.retrieval_query
                })
        return citations
