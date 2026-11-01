# Adaptive State-Aware Multi-Hop RAG

A research-grade **Adaptive State-Aware Multi-Hop Retrieval-Augmented Generation (RAG)** platform featuring **Adaptive Dynamic Memory Management** as its core cognitive workspace.

---

## 🌟 Key Research Contributions & Features

1. **Adaptive Working Memory ($M_i$)**: Memory is not treated as a static list of strings. It is actively managed via **11 dynamic operations** (`ADD`, `UPDATE`, `MERGE`, `DELETE`, `RETAIN`, `DOWNWEIGHT`, `VERIFY`, `MARK_CONTRADICTION`, `RESOLVE_CONTRADICTION`, `SUMMARIZE`, `CONSOLIDATE`).
2. **Multi-Objective Retention Scoring**: Quantifies memory utility dynamically across relevance, novelty, evidence strength, reliability, verification status, and age decay.
3. **Hybrid Retrieval**: Combines dense semantic vector retrieval (FAISS), lexical token search (BM25 ranking), Reciprocal Rank Fusion (RRF), and Cross-Encoder neural reranking (`ms-marco-MiniLM-L-6-v2`).
4. **Adaptive Planning & Multi-Hop Traversal**:
   - **Socratic Planning Mode**: Linear iterative decomposition with uncertainty-gated retrieval.
   - **MCTS Complex Reasoning Mode**: Monte Carlo Tree Search exploring 5 distinct actions ($A_1$ Decompose, $A_2$ Consolidate, $A_3$ Refine, $A_4$ Redirect, $A_5$ Conclude) with branch-local isolation and verified global promotion.
5. **Path-Outcome Dual Reward System**: Evaluates local intermediate reasoning quality ($R_p$) and terminal goal satisfaction ($R_o$) using the ACL 2026 Dual Reward formulation.
6. **End-to-End Provenance & Citations**: Every fact tracks its chunk ID, retrieval query, rank, timestamp, and verification status.
7. **10-Variant Systematic Ablation Suite**: Isolates the performance contribution of every component from standard RAG to the full adaptive architecture.

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
Make sure you have Python 3.10+ and Node.js installed.
Set your Google Gemini API key in `backend/.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Start the Backend API (FastAPI)
```powershell
cd backend
.\.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```
Or directly using the helper script:
```powershell
cd backend
.\run.ps1
```

### 3. Start the Frontend (React + Vite)
```powershell
cd frontend
npm run dev
```

Visit `http://localhost:5173` in your browser.

---

## 🧪 Running Tests

### Unit Tests (Adaptive Memory & Scoring)
```powershell
cd backend
python tests/test_adaptive_memory.py
```

### End-to-End Integration & Multi-Hop Tests
```powershell
cd backend
python tests/test_end_to_end_multihop.py
```

---

## 📚 Technical Documentation

- [`docs/ARCHITECTURE_AUDIT.md`](docs/ARCHITECTURE_AUDIT.md): Comprehensive baseline audit and inspection report.
- [`docs/ADAPTIVE_STATE_RAG_ARCHITECTURE.md`](docs/ADAPTIVE_STATE_RAG_ARCHITECTURE.md): Complete system architecture and state model specifications.
- [`docs/MEMORY_MANAGEMENT.md`](docs/MEMORY_MANAGEMENT.md): Detailed algorithm specifications for the 11 dynamic memory operations.
- [`docs/RETRIEVAL_AND_PLANNING.md`](docs/RETRIEVAL_AND_PLANNING.md): Hybrid retrieval, RRF, and planning engines.
- [`docs/EVALUATION.md`](docs/EVALUATION.md): Dual reward metrics and benchmarking guide.
- [`docs/ABLATION_PLAN.md`](docs/ABLATION_PLAN.md): 10-variant experimental ablation matrix.
