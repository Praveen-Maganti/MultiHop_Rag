"""
Formal Common State Model for Adaptive State-Aware Multi-Hop RAG
Contains fully typed, serializable models for reasoning states, memory items,
provenance records, planning decisions, and verification results.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import uuid


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"
    UNCERTAIN = "UNCERTAIN"
    CONTRADICTED = "CONTRADICTED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    DISCARDED = "DISCARDED"


class MemoryOperation(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    DELETE = "DELETE"
    RETAIN = "RETAIN"
    DOWNWEIGHT = "DOWNWEIGHT"
    VERIFY = "VERIFY"
    MARK_CONTRADICTION = "MARK_CONTRADICTION"
    RESOLVE_CONTRADICTION = "RESOLVE_CONTRADICTION"
    SUMMARIZE = "SUMMARIZE"
    CONSOLIDATE = "CONSOLIDATE"


class ActionType(str, Enum):
    A1_DECOMPOSE_AND_ANSWER = "A1: Decompose & Answer"
    A2_CONSOLIDATE = "A2: Consolidate"
    A3_REFINE = "A3: Refine"
    A4_REDIRECT = "A4: Redirect"
    A5_CONCLUDE = "A5: Conclude"
    RETRIEVE = "RETRIEVE"
    VERIFY = "VERIFY"
    MERGE_MEMORY = "MERGE_MEMORY"
    RESOLVE_CONFLICT = "RESOLVE_CONFLICT"
    REFORMULATE_QUERY = "REFORMULATE_QUERY"


class Provenance(BaseModel):
    source_document: str = Field(default="", description="ID or filename of document")
    source_url: Optional[str] = Field(default=None)
    retrieval_query: str = Field(default="")
    retrieval_timestamp: float = Field(default_factory=time.time)
    retrieval_rank: int = Field(default=0)
    hop_number: int = Field(default=1)
    chunk_id: str = Field(default="")
    retriever_score: float = Field(default=0.0)
    reranker_score: float = Field(default=0.0)
    verification_status: str = Field(default="UNVERIFIED")

    def to_citation(self) -> str:
        return f"[Doc: {self.source_document} | Chunk: {self.chunk_id} | Hop {self.hop_number}]"


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    content: str = Field(description="The factual claim or distilled information")
    entity: Optional[str] = Field(default=None, description="Primary subject entity")
    relation: Optional[str] = Field(default=None, description="Relationship or predicate")
    value: Optional[str] = Field(default=None, description="Object or attribute value")
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_text: str = Field(default="", description="Raw context snippet")
    source_type: str = Field(default="retrieval", description="retrieval, reflection, or user")
    provenance: List[Provenance] = Field(default_factory=list)
    hop: int = Field(default=1)
    
    # Quantitative Scores (0.0 to 1.0)
    confidence: float = Field(default=0.85)
    reliability: float = Field(default=0.80)
    relevance_score: float = Field(default=0.80)
    importance_score: float = Field(default=0.80)
    novelty_score: float = Field(default=1.0)
    future_utility_score: float = Field(default=0.70)
    evidence_strength: float = Field(default=0.80)
    redundancy_score: float = Field(default=0.0)
    retention_score: float = Field(default=0.80)
    
    # Status and Audit Attributes
    contradiction_status: str = Field(default="NONE", description="NONE, DETECTED, RESOLVED")
    conflicting_item_ids: List[str] = Field(default_factory=list)
    verification_status: str = Field(default="UNVERIFIED", description="UNVERIFIED, VERIFIED, REFUTED")
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    
    # Access and Lifecycle tracking
    last_accessed_hop: int = Field(default=1)
    access_count: int = Field(default=1)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    embedding: Optional[List[float]] = Field(default=None, exclude=True)
    decision_reason: str = Field(default="Added during evidence intake")


class Evidence(BaseModel):
    chunk_id: str
    text: str
    source_document: str = ""
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    hop_retrieved: int = 1
    query_used: str = ""


class SubQuestion(BaseModel):
    id: str = Field(default_factory=lambda: f"sq_{uuid.uuid4().hex[:6]}")
    question: str
    dependencies: List[str] = Field(default_factory=list, description="IDs of preceding subquestions needed")
    expected_output_type: str = Field(default="entity", description="entity, comparison, fact, summary")
    status: str = Field(default="PENDING", description="PENDING, IN_PROGRESS, RESOLVED")
    resolved_answer: Optional[str] = None
    hop_assigned: int = 1


class MemoryUpdate(BaseModel):
    operation: MemoryOperation
    target_item_id: str
    reason: str
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)


class RetrievalResult(BaseModel):
    query: str
    hop: int
    raw_retrieved_count: int
    reranked_count: int
    selected_evidence: List[Evidence] = Field(default_factory=list)
    latency_seconds: float = 0.0


class VerificationResult(BaseModel):
    is_supported: bool = True
    factual_consistency_score: float = 1.0
    evidence_sufficiency_score: float = 1.0
    logical_validity_score: float = 1.0
    critique: str = ""
    unsupported_claims: List[str] = Field(default_factory=list)


class PlanningDecision(BaseModel):
    action: ActionType
    subquestion: Optional[str] = None
    rationale: str
    confidence: float = 1.0
    next_queries: List[str] = Field(default_factory=list)
    requires_retrieval: bool = True
    stop_reason: Optional[str] = None


class ReasoningStep(BaseModel):
    step_number: int
    action: ActionType
    subquestion: Optional[str] = None
    subanswer: Optional[str] = None
    retrieval_query: Optional[str] = None
    retrieved_documents_count: int = 0
    memory_operations: List[MemoryUpdate] = Field(default_factory=list)
    path_reward: float = 0.8
    verification: Optional[VerificationResult] = None
    active_memory_count: int = 0
    duration_seconds: float = 0.0


class ReasoningState(BaseModel):
    """
    Formal state model Si for Adaptive State-Aware RAG.
    Completely serializable across hops, branches, and API endpoints.
    """
    state_id: str = Field(default_factory=lambda: f"state_{uuid.uuid4().hex[:8]}")
    query: str
    current_subquestion: Optional[str] = None
    intermediate_answer: Optional[str] = None
    working_memory: List[MemoryItem] = Field(default_factory=list)
    episodic_history: List[Dict[str, str]] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    unresolved_questions: List[SubQuestion] = Field(default_factory=list)
    confidence: float = 0.85
    reasoning_step: int = 0
    max_steps: int = 4
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list)
    selected_documents: List[Dict[str, Any]] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    step_history: List[ReasoningStep] = Field(default_factory=list)
    total_retrieval_count: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    status: str = Field(default="IN_PROGRESS", description="IN_PROGRESS, CONCLUDED, FAILED")
    final_answer: Optional[str] = None

    def get_active_memory(self) -> List[MemoryItem]:
        return [item for item in self.working_memory if item.status in [MemoryStatus.ACTIVE, MemoryStatus.VERIFIED]]

    def get_memory_context_text(self) -> str:
        active = self.get_active_memory()
        if not active:
            return "Working memory is currently empty."
        lines = []
        for idx, item in enumerate(active):
            cit = item.provenance[0].to_citation() if item.provenance else f"[Item {item.id}]"
            lines.append(f"[{idx+1}] {item.content} (Confidence: {item.confidence:.2f}, Provenance: {cit})")
        return "\n".join(lines)
