"""
LLM-assisted answer accuracy benchmark.

This is an evaluation tool, not the production QA path. It builds a reference
answer from a stronger evidence pool, generates answers from each retrieval
strategy, then asks an LLM judge to score strategy answers against the reference.

Run:
    python experiments/llm_accuracy_benchmark.py --limit 8
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from benchmark import TEST_QUERIES
from retrieval.pipeline import RetrievalPipeline

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INDEX_DIR = ROOT / "data" / "indexes"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

METHODS = ["hybrid_lsh", "minhash", "simhash", "tfidf"]
METHOD_LABELS = {
    "hybrid_lsh": "Hybrid LSH",
    "minhash": "MinHash LSH",
    "simhash": "SimHash",
    "tfidf": "TF-IDF Baseline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark answer accuracy with an LLM judge.")
    parser.add_argument("--limit", type=int, default=8, help="Number of test queries to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks used by each strategy answer.")
    parser.add_argument("--oracle-k", type=int, default=12, help="TF-IDF chunks used for reference answer.")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--sleep", type=float, default=3.0, help="Delay between LLM calls.")
    parser.add_argument("--methods", default="hybrid_lsh,minhash,simhash,tfidf")
    return parser.parse_args()


def call_groq(messages: List[Dict], model: str, max_tokens: int = 700, retries: int = 5) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing. Add it to .env first.")

    for attempt in range(retries):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        if response.status_code == 429 and attempt < retries - 1:
            wait = float(response.headers.get("retry-after", 12 + attempt * 8))
            print(f"  Rate limited. Waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    raise RuntimeError("Groq request failed after retries.")


def chunk_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        start_page = chunk.get("start_page", "?")
        end_page = chunk.get("end_page", start_page)
        pages = f"p.{start_page}" if start_page == end_page else f"pp.{start_page}-{end_page}"
        section = chunk.get("section", "General")
        parts.append(f"[{i}] {source} | {section} | {pages}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def reference_answer(question: str, chunks: List[Dict], model: str) -> str:
    context = chunk_context(chunks)
    return call_groq(
        [
            {
                "role": "system",
                "content": (
                    "You create reference academic policy answers from handbook excerpts. "
                    "Use only supplied excerpts. If answer is absent, say absent."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nReference evidence:\n{context}\n\nReference answer:",
            },
        ],
        model=model,
    )


def strategy_answer(question: str, chunks: List[Dict], model: str) -> str:
    context = chunk_context(chunks)
    return call_groq(
        [
            {
                "role": "system",
                "content": (
                    "Answer academic policy questions using only supplied retrieved excerpts. "
                    "If the excerpts do not contain the answer, say the excerpts do not contain it."
                ),
            },
            {
                "role": "user",
                "content": f"Retrieved excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
        model=model,
        max_tokens=500,
    )


def judge_answer(question: str, reference: str, candidate: str, model: str) -> Dict:
    raw = call_groq(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. Compare candidate answer to reference answer. "
                    "Return only valid JSON with keys: correctness, completeness, hallucination_risk, verdict, notes. "
                    "Scores are integers 0-5. correctness 5 means same policy meaning. "
                    "hallucination_risk 5 means high unsupported content, 0 means none."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Reference answer:\n{reference}\n\n"
                    f"Candidate answer:\n{candidate}\n\n"
                    "JSON:"
                ),
            },
        ],
        model=model,
        max_tokens=350,
    )

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    json_text = match.group(0) if match else raw
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {
            "correctness": None,
            "completeness": None,
            "hallucination_risk": None,
            "verdict": "parse_error",
            "notes": raw,
        }


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    pipeline = RetrievalPipeline(str(INDEX_DIR), str(CHUNKS_FILE))
    queries = TEST_QUERIES[: args.limit]

    rows = []
    details_path = RESULTS_DIR / "llm_accuracy_benchmark.jsonl"
    csv_path = RESULTS_DIR / "llm_accuracy_benchmark.csv"

    with open(details_path, "w", encoding="utf-8") as detail_file:
        for q_idx, question in enumerate(queries, 1):
            print(f"[{q_idx}/{len(queries)}] {question}")

            oracle = pipeline.query(question, method="tfidf", top_k=args.oracle_k)
            ref = reference_answer(question, oracle.chunks, args.model)
            time.sleep(args.sleep)

            selected_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
            for method in selected_methods:
                if method not in pipeline.available_methods:
                    continue

                result = pipeline.query(question, method=method, top_k=args.top_k)
                answer = strategy_answer(question, result.chunks, args.model)
                time.sleep(args.sleep)
                scores = judge_answer(question, ref, answer, args.model)

                row = {
                    "query": question,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "latency_ms": round(result.latency_ms, 3),
                    "correctness": scores.get("correctness"),
                    "completeness": scores.get("completeness"),
                    "hallucination_risk": scores.get("hallucination_risk"),
                    "verdict": scores.get("verdict"),
                    "notes": scores.get("notes"),
                    "reference_answer": ref,
                    "candidate_answer": answer,
                }
                rows.append(row)
                detail_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                detail_file.flush()
                time.sleep(args.sleep)

    fieldnames = [
        "query",
        "method",
        "method_label",
        "latency_ms",
        "correctness",
        "completeness",
        "hallucination_risk",
        "verdict",
        "notes",
        "reference_answer",
        "candidate_answer",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved details: {details_path}")


if __name__ == "__main__":
    main()
