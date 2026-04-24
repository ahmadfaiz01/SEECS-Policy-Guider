"""
benchmark.py
This script runs the actual experiments required for the report.
It tests our custom LSH and SimHash against the standard TF-IDF,
and saves the results as CSVs so we can graph them later.

To run:
    python experiments/benchmark.py
"""
import json
import sys
import time
import tracemalloc
import copy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexing.minhash_lsh import MinHashLSH
from indexing.simhash import SimHashIndex
from indexing.tfidf_retriever import TFIDFRetriever

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CHUNKS_FILE = Path("data/processed/chunks.jsonl")

TEST_QUERIES = [
    "What is the minimum CGPA requirement for graduation?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What is the grading scale at SEECS?",
    "What are the requirements for the Dean's Honor List?",
    "What is the maximum course load per semester?",
    "What is the policy on academic probation?",
    "How are final grades calculated?",
    "What is the procedure for dropping a course?",
    "What is the internship requirement for graduation?",
    "What are the thesis requirements for MS students?",
    "What is the policy on plagiarism?",
    "How can a student appeal a grade?",
    "What are the requirements for changing a major?",
]

def load_chunks() -> list:
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def timed_query(index, query: str, top_k: int = 5):
    tracemalloc.start()
    t0 = time.perf_counter()
    results = index.query(query, top_k)
    latency_ms = (time.perf_counter() - t0) * 1000
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return results, latency_ms, mem_peak / 1024

def timed_fn(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    results = fn()
    latency_ms = (time.perf_counter() - t0) * 1000
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return results, latency_ms, mem_peak / 1024

def hybrid_lsh_query(minhash, simhash, tfidf, query: str, top_k: int = 5):
    candidate_limit = max(50, top_k * 30)
    candidate_ids = []
    candidate_ids.extend(cid for cid, _ in minhash.query(query, top_k=candidate_limit))
    candidate_ids.extend(cid for cid, _ in simhash.query(query, top_k=candidate_limit))
    return tfidf.score_candidates(query, candidate_ids, top_k=top_k)

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k

# ── Experiment 1: Exact vs Approximate ───────────────────────────────────────

def exp1_comparison(chunks: list) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("EXP 1: Exact vs Approximate Retrieval")
    print("=" * 55)

    tfidf = TFIDFRetriever()
    tfidf.fit([c["chunk_id"] for c in chunks], [c["text"] for c in chunks])

    minhash = MinHashLSH(num_perm=256, num_bands=128, shingle_k=1)
    for c in chunks: minhash.add(c["chunk_id"], c["text"])

    simhash = SimHashIndex(hash_bits=64, hamming_threshold=None)
    for c in chunks: simhash.add(c["chunk_id"], c["text"])

    records = []
    for query in TEST_QUERIES:
        tfidf_res, t_tfidf, m_tfidf = timed_query(tfidf, query)
        hy_res,     t_hy,    m_hy    = timed_fn(lambda q=query: hybrid_lsh_query(minhash, simhash, tfidf, q))
        mh_res,    t_mh,    m_mh    = timed_query(minhash, query)
        sh_res,    t_sh,    m_sh    = timed_query(simhash, query)

        gt = {cid for cid, _ in tfidf_res}

        records.append({
            "query": query[:50],
            "tfidf_latency_ms":   round(t_tfidf, 3),
            "hybrid_lsh_latency_ms": round(t_hy, 3),
            "minhash_latency_ms": round(t_mh, 3),
            "simhash_latency_ms": round(t_sh, 3),
            "tfidf_mem_kb":       round(m_tfidf, 2),
            "hybrid_lsh_mem_kb":  round(m_hy, 2),
            "minhash_mem_kb":     round(m_mh, 2),
            "simhash_mem_kb":     round(m_sh, 2),
            "hybrid_lsh_precision5": precision_at_k([r[0] for r in hy_res], gt, 5),
            "minhash_precision5": precision_at_k([r[0] for r in mh_res], gt, 5),
            "simhash_precision5": precision_at_k([r[0] for r in sh_res], gt, 5),
        })
        print(f"  Query: {query[:45]}...")
        print(f"    TF-IDF: {t_tfidf:.2f}ms | Hybrid: {t_hy:.2f}ms | MinHash: {t_mh:.2f}ms | SimHash: {t_sh:.2f}ms")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "exp1_comparison.csv", index=False)
    print(f"\n[EXP1] Saved -> {RESULTS_DIR / 'exp1_comparison.csv'}")

    print(df[["tfidf_latency_ms", "hybrid_lsh_latency_ms", "minhash_latency_ms", "simhash_latency_ms",
              "hybrid_lsh_precision5", "minhash_precision5", "simhash_precision5"]].mean().round(3).to_string())
    return df

# ── Experiment 2: Parameter Sensitivity ──────────────────────────────────────

def exp2_minhash_sensitivity(chunks: list) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("EXP 2a: MinHash Parameter Sensitivity")
    print("=" * 55)

    tfidf = TFIDFRetriever()
    tfidf.fit([c["chunk_id"] for c in chunks], [c["text"] for c in chunks])

    configs = [
        (64, 32), (64, 64), (128, 64), (128, 128), (256, 128), (256, 256),
    ]
    records = []
    
    for num_perm, num_bands in configs:
        idx = MinHashLSH(num_perm=num_perm, num_bands=num_bands, shingle_k=1)
        for c in chunks: idx.add(c["chunk_id"], c["text"])

        latencies, precisions = [], []
        for query in TEST_QUERIES[:8]:
            gt_res = tfidf.query(query, top_k=5)
            gt_set = {r[0] for r in gt_res}
            
            res, lat, _ = timed_query(idx, query, top_k=5)
            latencies.append(lat)
            precisions.append(precision_at_k([r[0] for r in res], gt_set, 5))

        rec = {
            "num_perm": num_perm,
            "num_bands": num_bands,
            "rows_per_band": num_perm // num_bands,
            "avg_latency_ms": round(np.mean(latencies), 3),
            "avg_precision5": round(np.mean(precisions), 4),
        }
        records.append(rec)
        print(f"  perm={num_perm}, bands={num_bands}: "
              f"lat={rec['avg_latency_ms']}ms, P@5={rec['avg_precision5']}")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "exp2a_minhash_sensitivity.csv", index=False)
    print(f"\n[EXP2a] Saved -> {RESULTS_DIR / 'exp2a_minhash_sensitivity.csv'}")
    return df


def exp2_simhash_sensitivity(chunks: list) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("EXP 2b: SimHash Hamming Threshold Sensitivity")
    print("=" * 55)

    tfidf = TFIDFRetriever()
    tfidf.fit([c["chunk_id"] for c in chunks], [c["text"] for c in chunks])

    thresholds = [3, 5, 8, 10, 12, 15, 20, None]
    records = []

    for threshold in thresholds:
        idx = SimHashIndex(hash_bits=64, hamming_threshold=threshold)
        for c in chunks: idx.add(c["chunk_id"], c["text"])

        precisions, recalls = [], []
        latencies = []
        for query in TEST_QUERIES[:8]:
            gt_res = tfidf.query(query, top_k=5)
            gt_set = {r[0] for r in gt_res}
            
            res, lat, _ = timed_query(idx, query, top_k=5)
            retrieved = [r[0] for r in res]
            
            latencies.append(lat)
            precisions.append(precision_at_k(retrieved, gt_set, 5))
            hit_count = sum(1 for r in retrieved if r in gt_set)
            recalls.append(hit_count / max(len(gt_set), 1))

        rec = {
            "hamming_threshold": threshold if threshold is not None else "None",
            "avg_latency_ms": round(np.mean(latencies), 3),
            "avg_precision5": round(np.mean(precisions), 4),
            "avg_recall5":    round(np.mean(recalls), 4),
        }
        records.append(rec)
        print(f"  threshold={threshold}: P@5={rec['avg_precision5']}, R@5={rec['avg_recall5']}")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "exp2b_simhash_sensitivity.csv", index=False)
    print(f"\n[EXP2b] Saved -> {RESULTS_DIR / 'exp2b_simhash_sensitivity.csv'}")
    return df

# ── Experiment 3: Scalability ─────────────────────────────────────────────────

def exp3_scalability(chunks: list) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("EXP 3: Scalability Test (Corpus Duplication)")
    print("=" * 55)

    multipliers = [1, 2, 5, 10, 20]
    query = TEST_QUERIES[0]
    records = []

    for mult in multipliers:
        scaled = []
        for i in range(mult):
            for c in chunks:
                dup = dict(c)
                dup["chunk_id"] = f"{c['chunk_id']}__dup{i}"
                scaled.append(dup)

        n = len(scaled)
        print(f"\n  Corpus x{mult} -> {n} chunks")

        tf = TFIDFRetriever()
        tf.fit([c["chunk_id"] for c in scaled], [c["text"] for c in scaled])
        _, t_tf, _ = timed_query(tf, query)

        mh = MinHashLSH(num_perm=256, num_bands=128, shingle_k=1)
        for c in scaled: mh.add(c["chunk_id"], c["text"])
        _, t_mh, _ = timed_query(mh, query)

        sh = SimHashIndex(hash_bits=64, hamming_threshold=None)
        for c in scaled: sh.add(c["chunk_id"], c["text"])
        _, t_hy, _ = timed_fn(lambda: hybrid_lsh_query(mh, sh, tf, query))
        _, t_sh, _ = timed_query(sh, query)

        records.append({
            "multiplier": mult,
            "num_chunks": n,
            "tfidf_latency_ms":   round(t_tf, 3),
            "hybrid_lsh_latency_ms": round(t_hy, 3),
            "minhash_latency_ms": round(t_mh, 3),
            "simhash_latency_ms": round(t_sh, 3),
        })
        print(f"    TF-IDF={t_tf:.2f}ms | Hybrid={t_hy:.2f}ms | MinHash={t_mh:.2f}ms | SimHash={t_sh:.2f}ms")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "exp3_scalability.csv", index=False)
    print(f"\n[EXP3] Saved -> {RESULTS_DIR / 'exp3_scalability.csv'}")
    return df


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")

    exp1_comparison(chunks)
    exp2_minhash_sensitivity(chunks)
    exp2_simhash_sensitivity(chunks)
    exp3_scalability(chunks)

    print("\n  All experiments complete. Results in experiments/results/")
