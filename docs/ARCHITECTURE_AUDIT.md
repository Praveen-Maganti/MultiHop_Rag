# Architecture Audit & System Inspection Report
**Project:** MultiHop_Rag → Adaptive State-Aware Multi-Hop RAG  
**Date:** October 2026  
**Auditor:** Senior AI/ML Research Engineer & Software Architect  

---

## 1. Executive Summary & Audit Overview

This audit evaluates the current state of the `MultiHop_Rag` codebase against the research-grade target of **Adaptive State-Aware Multi-Hop RAG** centered around **Adaptive Memory Management**.

The current repository contains an initial Capstone prototype and recent State-Aware RAG modules. It possesses basic FAISS indexing, cross-encoder reranking, prompt-based memory scoring, Socratic/MCTS execution scripts, and a React frontend. However, it lacks a unified serializable state machine, structured provenance schemas, hybrid retrieval, dynamic action orchestration, branch-isolated memory promotion, automated ablation frameworks, and configurable multi-objective memory scoring.

---

## 2. Current Architecture & Component Inventory

```
CURRENT DATA FLOW:
User Query / Document 
   │
   ├──► FastAPI Endpoint (`/api/chat`, `/api/documents`)
   │      │
   │      ├──► Vector Store (`vector_store.py`: FAISS IndexFlatIP + all-MiniLM-L6-v2)
   │      ├──► CrossEncoder (`reranker.py`: ms-marco-MiniLM-L-6-v2)
   │      │
   │      ├──► Multi-Hop Baseline (`multi_hop_rag.py`: Flat append context)
   │      ├──► Proposed Memory RAG (`proposed_rag.py`: Regex claim splitting + prompt scoring)
   │      └──► State-Aware RAG (`state_aware_rag.py`: Socratic loop + MCTS prototype)
   │
   └──► React Vite Frontend (`App.jsx`, `ChatInterface.jsx`, `ComparisonView.jsx`)
```

### Existing Components:
1. **Backend API (`backend/app/api/`)**:
   - `upload.py`: Handles `.docx`, `.pdf`, `.txt` file uploads, chunking, and FAISS indexing.
   - `chat.py`: Dispatches chat requests to standard multihop or state-aware pipelines.
   - `experiments.py`: Runs batch evaluation benchmarks and writes to SQLite database `experiments.db`.
   - `metrics.py`: Queries aggregate statistics from SQLite.
2. **Retrieval Layer (`backend/app/retrieval/`)**:
   - `vector_store.py`: In-memory dictionary storing FAISS `IndexFlatIP` per document and global multi-document search.
   - `reranker.py`: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) reranking top-10 to top-3.
   - `semantic_filter.py`: Cosine threshold filtering against query embedding.
3. **Memory Modules (`backend/app/memory/`)**:
   - `working_memory.py`: Basic `MemoryItem` and `WorkingMemory` container.
   - `claim_extractor.py`: Regex sentence splitting with KV fallback.
   - `evidence_scorer.py`: Prompt-based relevance, support, and reliability scoring.
   - `conflict_detector.py`: LLM pairwise contradiction detector gated by cosine similarity.
   - `memory_compressor.py`: Deduplication and utility-based item eviction.
   - `utility_estimator.py`: Prompt-based utility prediction.
4. **State-Aware RAG Pipelines (`backend/app/rag/`)**:
   - `state_aware_prompts.py`: Formal prompt templates from ACL 2026 paper.
   - `state_aware_rag.py`: Socratic mode and MCTS exploration with shared working memory.
   - `standard_rag.py` & `multi_hop_rag.py`: Baseline comparative systems.
5. **Evaluation & Tracing (`backend/app/evaluation/` & `tracing/`)**:
   - `evaluator.py`: Token-level F1 and evidence coverage metrics.
   - `dual_reward.py`: Step-level Path Reward ($R_p$) and Outcome Reward ($R_o$).
   - `reasoning_trace.py`: Hop-by-hop event logging.
6. **Frontend UI (`frontend/src/`)**:
   - React + Vite dashboard displaying side-by-side comparison, reasoning hops, active memory, and metric tables.

---

## 3. What Can Be Reused vs. Modified vs. Missing

| Layer / Capability | Status | Strategy |
| :--- | :--- | :--- |
| **Vector Indexing & FAISS** | Functional | **Reuse & Extend**: Keep FAISS backend; add hybrid retrieval (BM25/Lexical + Dense RRF fusion). |
| **Cross-Encoder Reranking** | Functional | **Reuse**: Retain `ms-marco-MiniLM-L-6-v2` with batching and error fallback. |
| **Document Parsers** | Functional | **Reuse**: Retain PyMuPDF (`fitz`), python-docx, and txt parsing. |
| **State Representation** | Incomplete | **Create New**: Build formal Pydantic/dataclass `ReasoningState`, `MemoryItem`, `Evidence`, `Provenance`, `PlanningDecision`. |
| **Adaptive Memory Controller** | Heuristic / Fragmented | **Refactor & Upgrade**: Build central `AdaptiveMemoryManager` with 11 distinct operations (ADD, UPDATE, MERGE, DELETE, RETAIN, DOWNWEIGHT, VERIFY, CONFLICT, RESOLVE, SUMMARIZE, CONSOLIDATE). |
| **Configurable Scoring** | Hardcoded Weights | **Create New**: Parameterized formula with configurable $\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta, \lambda$ weights. |
| **Provenance Tracking** | Basic Chunk IDs | **Create New**: Full end-to-end evidence citation, retrieval rank, query origin, and verification status. |
| **Query Decomposer** | Prompt-only | **Upgrade**: Add dependency graph structure, adaptive single vs multi-hop complexity classifier. |
| **Uncertainty & Adaptive Retrieval** | None | **Create New**: Memory sufficiency evaluation before triggering search calls. |
| **Verification Engine** | Fragmented | **Create New**: Dedicated `Verifier` for claim support, sufficiency, and logical validity. |
| **Branch vs Global Memory (MCTS)** | Shared only | **Upgrade**: Implement branch-local memory isolation and verified promotion to global memory. |
| **Ablation Framework** | None | **Create New**: Systematic 10-variant comparative ablation runner (`A` through `J`). |
| **RL-Ready Architecture** | Partial (Prompts only) | **Create New**: Gym-style step environment $(S_i, a_i, r_i, S_{i+1})$ with dual + efficiency rewards. |

---

## 4. Proposed Target Architecture

```
                                  USER QUERY x
                                       │
                                       ▼
                             ┌──────────────────┐
                             │  Query Analyzer  │
                             └─────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
            [Single-Hop Path]                     [Multi-Hop Path]
                    │                                     │
                    │                             Query Decomposer
                    │                           (DAG Dependency Graph)
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                             ┌──────────────────┐
                             │ Adaptive Planner │◄────────────────┐
                             └─────────┬────────┘                 │
                                       │                          │
        ┌───────────────┬──────────────┼───────────────┬──────────┘
        ▼               ▼              ▼               ▼
   [RETRIEVE]       [REFINE]      [REDIRECT]      [CONSOLIDATE]
        │
        ▼
   Hybrid Retriever (Dense FAISS + BM25 Lexical + RRF Fusion)
        │
        ▼
   Cross-Encoder Reranker
        │
        ▼
   Semantic Evidence Filter
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE MEMORY MANAGER                         │
│  Operations: ADD | UPDATE | MERGE | DELETE | RETAIN | DOWNWEIGHT |     │
│              VERIFY | MARK_CONFLICT | RESOLVE_CONFLICT | CONSOLIDATE   │
│                                                                        │
│  Multi-Objective Score:                                                │
│    S = α·Rel + β·Relb + γ·Nov + δ·FutUtil + ε·EvStr + ζ·Verif          │
│        - η·Red - θ·ConfRisk - λ·Cost                                   │
│                                                                        │
│  Memory Hierarchy:                                                     │
│    ├── Working Memory (Active Hops & Distilled Facts)                  │
│    ├── Branch Memory (MCTS Isolated Explorations)                      │
│    ├── Global Memory (Verified Cross-Branch Knowledge)                 │
│    └── Provenance Graph (Full Audit Trace to Source Chunks)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                             Reasoning Engine
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Socratic Engine                 Adaptive MCTS Engine
            (Linear 1-Chain)             (5-Action Tree with UCT)
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         Verification & Synthesis
                                    │
                           ┌────────┴────────┐
                           ▼                 ▼
                      [CONTINUE]        [CONCLUDE]
                           │                 │
                      (Next Hop)             ▼
                                       Final Answer
                                  + Evidence Provenance
```

---

## 5. File Modification and Creation Plan

### New Files to Create:
1. `docs/ARCHITECTURE_AUDIT.md` (This document)
2. `docs/ADAPTIVE_STATE_RAG_ARCHITECTURE.md`
3. `docs/MEMORY_MANAGEMENT.md`
4. `docs/RETRIEVAL_AND_PLANNING.md`
5. `docs/EVALUATION.md`
6. `docs/ABLATION_PLAN.md`
7. `backend/app/core/config.py` (Typed system configuration)
8. `backend/app/core/state.py` (Common State & Memory data models)
9. `backend/app/core/logging.py` (Observability & structured reasoning trace recorder)
10. `backend/app/memory/adaptive_manager.py` (Central Adaptive Memory Controller)
11. `backend/app/memory/scoring.py` (Multi-objective configurable scoring)
12. `backend/app/memory/provenance.py` (Provenance graph & evidence citations)
13. `backend/app/memory/global_branch_memory.py` (MCTS Branch vs Global Memory hierarchy)
14. `backend/app/retrieval/hybrid_retriever.py` (Hybrid Dense + Lexical + RRF retriever)
15. `backend/app/retrieval/uncertainty.py` (Memory sufficiency & uncertainty evaluator)
16. `backend/app/planning/decomposer.py` (Adaptive query decomposition & dependency DAG)
17. `backend/app/planning/verifier.py` (Self-reflection, contradiction & factual verifier)
18. `backend/app/planning/adaptive_planner.py` (Unified action selector & dynamic controller)
19. `backend/app/evaluation/ablation.py` (10-variant systematic ablation runner)
20. `backend/app/evaluation/rl_interface.py` (Gym-like RL transition interface & reward function)
21. `backend/tests/test_adaptive_memory.py` (Comprehensive unit test suite)
22. `backend/tests/test_end_to_end_multihop.py` (End-to-end integration test suite)

### Existing Files to Update:
1. `backend/app/api/chat.py` (Support unified adaptive state and modes)
2. `backend/app/rag/state_aware_rag.py` (Integrate with new core state and adaptive manager)
3. `backend/app/memory/working_memory.py` (Bridge with new `MemoryItem` schema)
4. `frontend/src/ComparisonView.jsx` (Render structured provenance, confidence, and memory ops)
5. `README.md` (Updated system documentation and research architecture)

---

## 6. Risk Assessment & Compatibility Strategy

1. **Gemini API Rate Limiting (15 RPM Free Tier)**:
   - *Risk*: High number of sub-LLM calls per multi-hop query can trigger HTTP 429.
   - *Mitigation*: Built-in caching, uncertainty-based retrieval gating (skip retrieval when memory is sufficient), and exponential backoff retry with automatic fallback.
2. **Backward Compatibility**:
   - *Risk*: Modifying memory structures could break existing endpoints (`/api/documents`, `/api/experiments`).
   - *Mitigation*: Adapter layers and backward-compatible serialization so legacy API shapes (`memory`, `hop_trace`, `metrics`) remain fully functional.
3. **Execution Latency in MCTS**:
   - *Risk*: Large tree rollouts can take considerable time.
   - *Mitigation*: Adaptive complexity estimator routes simpler questions to Socratic mode and limits MCTS rollouts to complex questions with small bounded depth.
