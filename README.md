# SEECS Policy QA System

Scalable academic policy question-answering over SEECS UG and PG handbooks. The project is retrieval-first: MinHash LSH, SimHash, and TF-IDF retrieve evidence, then the optional LLM only writes an answer from retrieved chunks.

## Requirement Coverage

| Requirement | Status | Where |
| --- | --- | --- |
| PDF ingestion and cleaning | Implemented | `src/ingestion/`, `build_index.py` |
| 200-500 word chunking target | Implemented with tunable chunking | `chunk_pages(..., chunk_size=150, overlap=50)` in `build_index.py` |
| MinHash + LSH | Implemented from scratch | `src/indexing/minhash_lsh.py` |
| SimHash + Hamming distance | Implemented from scratch | `src/indexing/simhash.py` |
| TF-IDF cosine baseline | Implemented | `src/indexing/tfidf_retriever.py` |
| Query-time comparison | Implemented | `src/retrieval/pipeline.py`, `app.py` |
| Grounded answer generation | Implemented with optional Groq API | `src/generation/answer_gen.py` |
| Evidence and page/source display | Implemented | `app.py` |
| Competitive extension | PageRank section authority boost | `src/extensions/pagerank.py` |
| Required experiments | Implemented | `experiments/benchmark.py` |

## Project Structure

```text
src/
  ingestion/       PDF parsing, cleaning, chunking
  indexing/        MinHash LSH, SimHash, TF-IDF baseline
  retrieval/       Query pipeline with latency and memory tracking
  generation/      Optional grounded LLM answer synthesis
  extensions/      PageRank section ranking
experiments/
  benchmark.py     Exact vs approximate, sensitivity, scalability tests
  plots.py         Figure generation from CSV results
data/
  raw/             Handbook PDFs
  processed/       Generated chunks
  indexes/         Generated indexes
app.py             Streamlit demo UI
build_index.py     Rebuilds chunks and indexes
```

## Quick Start

```powershell
pip install -r requirements.txt
copy .env.example .env
python build_index.py
streamlit run app.py
```

Add `GROQ_API_KEY` to `.env` only if you want synthesized answers. Retrieval, comparison, evidence display, and analytics work without the LLM key.

## Experiments

```powershell
python experiments/benchmark.py
python experiments/plots.py
python experiments/llm_accuracy_benchmark.py --limit 8 --sleep 8
```

Generated CSVs:

| File | Purpose |
| --- | --- |
| `exp1_comparison.csv` | TF-IDF vs Hybrid LSH vs MinHash vs SimHash latency, memory, Precision@5 |
| `exp2a_minhash_sensitivity.csv` | Effect of hash functions and bands |
| `exp2b_simhash_sensitivity.csv` | Effect of Hamming threshold |
| `exp3_scalability.csv` | Corpus duplication scalability test |
| `llm_accuracy_benchmark.csv` | LLM-judged answer correctness/completeness/risk by strategy |
| `manual_evaluation_template.csv` | Manual 10-15 query correctness checklist for report |

## Demo Script

1. Run `streamlit run app.py`.
2. Ask a sample query such as `What is the attendance policy?`.
3. Show primary Hybrid LSH results, standalone MinHash, standalone SimHash, and TF-IDF baseline side by side.
4. Point out latency, memory, retrieved chunks, page references, and optional PageRank boost.
5. Open Performance Analytics to show exact vs approximate tradeoffs, parameter sensitivity, and scalability.

## Important Constraint

Do not upload the PDF directly to a chatbot. The LLM receives only top retrieved chunks from the retrieval pipeline, and the UI displays those chunks as supporting evidence.
