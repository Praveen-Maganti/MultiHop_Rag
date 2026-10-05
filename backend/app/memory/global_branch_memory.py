"""
Branch and Global Memory Hierarchy for MCTS
Implements Phase 12 & 13: Isolated Branch Memory with Verified Global Promotion.
"""

from typing import List, Dict, Any, Optional
from app.core.state import MemoryItem, MemoryStatus
from app.memory.adaptive_manager import AdaptiveMemoryManager
from app.utils.embeddings import compute_cosine_similarity


class GlobalBranchMemorySystem:
    """
    Manages Global Shared Memory alongside isolated Branch-Local Memories for MCTS.
    Prevents cross-branch contamination by verifying facts before global promotion.
    """
    def __init__(self, global_max_items: int = 10, branch_max_items: int = 5):
        self.global_memory = AdaptiveMemoryManager(max_items=global_max_items)
        self.branch_memories: Dict[str, AdaptiveMemoryManager] = {}
        self.branch_max_items = branch_max_items

    def create_branch_memory(self, branch_id: str) -> AdaptiveMemoryManager:
        branch_mgr = AdaptiveMemoryManager(max_items=self.branch_max_items)
        # Seed branch with current verified global items (read-only snapshot)
        for g_item in self.global_memory.get_active_items():
            if g_item.verification_status == "VERIFIED" or g_item.confidence >= 0.85:
                # Add copy to branch
                branch_mgr.add_item(
                    content=g_item.content,
                    source_chunk_ids=list(g_item.source_chunk_ids),
                    source_text=g_item.source_text,
                    query="Seeded from Global Memory",
                    hop=g_item.hop,
                    relevance=g_item.relevance_score,
                    evidence_strength=g_item.evidence_strength,
                    provenance_list=list(g_item.provenance)
                )
        self.branch_memories[branch_id] = branch_mgr
        return branch_mgr

    def get_branch_memory(self, branch_id: str) -> AdaptiveMemoryManager:
        if branch_id not in self.branch_memories:
            return self.create_branch_memory(branch_id)
        return self.branch_memories[branch_id]

    def promote_to_global(
        self,
        branch_id: str,
        item_id: str,
        confidence_threshold: float = 0.80,
        current_hop: int = 1
    ) -> Optional[MemoryItem]:
        """
        Promotes a high-confidence, non-contradicted branch item to the shared global memory.
        """
        branch_mgr = self.get_branch_memory(branch_id)
        candidate = next((i for i in branch_mgr.items if i.id == item_id), None)
        if not candidate:
            return None

        # Check eligibility
        if candidate.confidence < confidence_threshold or candidate.contradiction_status == "DETECTED":
            return None

        # Add to global memory and consolidate
        promoted = self.global_memory.add_item(
            content=candidate.content,
            source_chunk_ids=list(candidate.source_chunk_ids),
            source_text=candidate.source_text,
            query=f"Promoted from Branch {branch_id}",
            hop=current_hop,
            relevance=candidate.relevance_score,
            evidence_strength=candidate.evidence_strength,
            provenance_list=list(candidate.provenance)
        )
        promoted.status = MemoryStatus.VERIFIED
        promoted.verification_status = "VERIFIED"
        self.global_memory.consolidate_memory(current_hop=current_hop, main_question="")
        return promoted
