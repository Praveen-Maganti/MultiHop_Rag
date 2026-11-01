"""
Adaptive State-Aware RAG Configuration
Central typed configuration with configurable thresholds and weights for all sub-systems.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os

class MemoryWeightsConfig(BaseModel):
    alpha_relevance: float = Field(default=2.0, description="Weight for question relevance")
    beta_reliability: float = Field(default=1.5, description="Weight for source reliability")
    gamma_novelty: float = Field(default=1.0, description="Weight for novelty / non-redundancy")
    delta_future_utility: float = Field(default=1.2, description="Weight for future hop utility")
    epsilon_evidence_strength: float = Field(default=1.8, description="Weight for factual direct evidence")
    zeta_verification: float = Field(default=1.5, description="Weight for verification status")
    eta_redundancy_penalty: float = Field(default=1.5, description="Penalty for semantic redundancy")
    theta_conflict_penalty: float = Field(default=2.0, description="Penalty for unverified contradiction")
    lambda_age_cost: float = Field(default=0.1, description="Decay per elapsed hop")

class MemoryConfig(BaseModel):
    max_items: int = Field(default=6, description="Max active items in working memory")
    max_tokens: int = Field(default=2048, description="Max tokens for working memory context")
    redundancy_threshold: float = Field(default=0.88, description="Cosine similarity for deduplication")
    conflict_similarity_threshold: float = Field(default=0.72, description="Cosine similarity threshold to trigger conflict check")
    retention_threshold: float = Field(default=0.50, description="Minimum retention score to keep item in memory")
    weights: MemoryWeightsConfig = Field(default_factory=MemoryWeightsConfig)
    dynamic_weights: bool = Field(default=True, description="Enable State-Conditioned Dynamic Weight Scheduling (Method 1)")
    auto_consolidate: bool = Field(default=True, description="Run consolidation after every reasoning step")

class RetrievalConfig(BaseModel):
    mode: str = Field(default="hybrid", description="dense, lexical, or hybrid")
    top_k_dense: int = Field(default=8, description="Dense retrieval top k")
    top_k_lexical: int = Field(default=8, description="Lexical BM25 retrieval top k")
    rerank_top_k: int = Field(default=4, description="Top k after CrossEncoder reranking")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant")
    dense_weight: float = Field(default=0.6, description="Weight for dense search in score fusion")
    lexical_weight: float = Field(default=0.4, description="Weight for lexical search in score fusion")
    similarity_threshold: float = Field(default=0.35, description="Minimum similarity threshold")

class PlanningConfig(BaseModel):
    mode: str = Field(default="adaptive", description="socratic, mcts, or adaptive")
    max_hops: int = Field(default=4, description="Max reasoning hops in Socratic mode")
    mcts_max_depth: int = Field(default=4, description="Max tree depth in MCTS mode")
    mcts_num_rollouts: int = Field(default=3, description="Number of MCTS rollouts")
    mcts_uct_c: float = Field(default=1.0, description="UCT exploration constant")
    complexity_threshold: int = Field(default=2, description="Expected hops to trigger MCTS over Socratic")
    enable_verification: bool = Field(default=True, description="Enable self-reflection and verification step")

class RewardWeightsConfig(BaseModel):
    w1_answer_correctness: float = Field(default=2.0)
    w2_evidence_quality: float = Field(default=1.5)
    w3_memory_quality: float = Field(default=1.2)
    w4_retrieval_efficiency: float = Field(default=0.8)
    w5_reasoning_efficiency: float = Field(default=0.8)
    w6_unnecessary_retrieval_penalty: float = Field(default=1.0)
    w7_memory_redundancy_penalty: float = Field(default=1.0)

class AdaptiveRAGConfig(BaseModel):
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    reward_weights: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    reranker_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    generator_model_name: str = Field(default="gemini-3.5-flash-lite")

# Global singleton configuration
system_config = AdaptiveRAGConfig()

def get_config() -> AdaptiveRAGConfig:
    return system_config

def update_config(new_settings: Dict[str, Any]) -> AdaptiveRAGConfig:
    global system_config
    system_config = AdaptiveRAGConfig.parse_obj({**system_config.dict(), **new_settings})
    return system_config
