import sqlite3
import os

DB_PATH = "experiments.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT,
            document_id TEXT,
            question TEXT,
            ground_truth TEXT,
            system_type TEXT,
            answer TEXT,
            retrieved_chunks INTEGER,
            memory_items INTEGER,
            hops INTEGER,
            retrieval_calls INTEGER,
            tokens INTEGER,
            latency REAL,
            evidence_coverage REAL,
            unsupported_claim_rate REAL,
            f1 REAL,
            precision REAL,
            recall REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ablation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ablation_id    TEXT,
            document_id    TEXT,
            question       TEXT,
            ground_truth   TEXT,
            variant        TEXT,
            answer         TEXT,
            f1             REAL,
            precision      REAL,
            recall         REAL,
            evidence_coverage     REAL,
            unsupported_claim_rate REAL,
            hops           INTEGER,
            retrieval_calls INTEGER,
            memory_items   INTEGER,
            tokens         INTEGER,
            latency        REAL,
            conflict_llm_calls    INTEGER,
            conflict_calls_saved  INTEGER,
            duplicates_removed    INTEGER,
            low_utility_removed   INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_ablation_run(run_data: dict) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ablation_runs (
            ablation_id, document_id, question, ground_truth, variant, answer,
            f1, precision, recall, evidence_coverage, unsupported_claim_rate,
            hops, retrieval_calls, memory_items, tokens, latency,
            conflict_llm_calls, conflict_calls_saved, duplicates_removed, low_utility_removed
        ) VALUES (
            :ablation_id, :document_id, :question, :ground_truth, :variant, :answer,
            :f1, :precision, :recall, :evidence_coverage, :unsupported_claim_rate,
            :hops, :retrieval_calls, :memory_items, :tokens, :latency,
            :conflict_llm_calls, :conflict_calls_saved, :duplicates_removed, :low_utility_removed
        )
    """, run_data)
    conn.commit()
    conn.close()
    return "success"

def save_experiment_run(run_data: dict) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO experiments (
            experiment_id, document_id, question, ground_truth, system_type,
            answer, retrieved_chunks, memory_items, hops, retrieval_calls,
            tokens, latency, evidence_coverage, unsupported_claim_rate,
            f1, precision, recall
        ) VALUES (
            :experiment_id, :document_id, :question, :ground_truth, :system_type,
            :answer, :retrieved_chunks, :memory_items, :hops, :retrieval_calls,
            :tokens, :latency, :evidence_coverage, :unsupported_claim_rate,
            :f1, :precision, :recall
        )
    """, run_data)
    conn.commit()
    conn.close()
    return "success"
