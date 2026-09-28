from typing import List, Dict
import uuid

class MemoryItem:
    def __init__(self, source_chunk_ids: list, claim: str, text: str, source_query: str = "", hop_added: int = 1, embedding: list = None):
        self.id = f"mem_{uuid.uuid4().hex[:8]}"
        self.claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        self.source_chunk_ids = source_chunk_ids
        self.claim = claim
        self.text = text
        self.source_query = source_query
        self.hop_added = hop_added
        self.last_used_hop = hop_added
        self.embedding = embedding
        
        self.relevance_score = 0.0
        self.evidence_score = 0.0
        self.reliability = 0.0
        self.redundancy_score = 0.0
        self.novelty_score = 0.0
        self.conflict_score = 0.0
        self.conflict_importance = 0.0
        self.utility_score = 0.0
        self.retention_score = 0.0
        
        self.in_conflict_with = []
        self.status = "ACTIVE"

    def calculate_retention_score(self, current_hop: int, weights=None):
        if weights is None:
            weights = {"w1": 1.0, "w2": 1.0, "w3": 1.0, "w4": 1.0, "w5": 1.0, "w6": 0.1}
            
        self.novelty_score = max(0.0, 1.0 - self.redundancy_score)
        age_penalty = (current_hop - self.last_used_hop) * weights["w6"]
        
        score = (
            weights["w1"] * self.relevance_score +
            weights["w2"] * self.evidence_score +
            weights["w3"] * self.novelty_score +
            weights["w4"] * self.conflict_importance -
            weights["w5"] * self.redundancy_score -
            age_penalty
        )
        self.utility_score = score
        self.retention_score = score
        return self.retention_score

class WorkingMemory:
    def __init__(self):
        self.memory_id = f"wm_{uuid.uuid4().hex[:8]}"
        self.items: List[MemoryItem] = []
        
    def add_item(self, item: MemoryItem):
        self.items.append(item)
        
    def get_all_text(self) -> str:
        active_items = [i for i in self.items if i.status in ["ACTIVE", "VERIFIED"]]
        return "\\n\\n".join([f"Evidence {item.id} (Claim: {item.claim}): {item.text}" for item in active_items])
        
    def to_dict(self):
        return {
            "memory_id": self.memory_id,
            "items": [
                {
                    "id": i.id,
                    "claim_id": i.claim_id,
                    "claim": i.claim,
                    "text": i.text,
                    "source_chunk_ids": i.source_chunk_ids,
                    "source_query": i.source_query,
                    "hop_added": i.hop_added,
                    "relevance_score": i.relevance_score,
                    "evidence_score": i.evidence_score,
                    "reliability": i.reliability,
                    "redundancy_score": i.redundancy_score,
                    "novelty_score": i.novelty_score,
                    "conflict_score": i.conflict_score,
                    "conflict_importance": i.conflict_importance,
                    "utility_score": i.utility_score,
                    "retention_score": i.retention_score,
                    "status": i.status
                }
                for i in self.items
            ]
        }
