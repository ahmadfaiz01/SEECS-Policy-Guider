# Project Evaluation Against Requirements

## Short Verdict

Project is already on the right track and is clearly a retrieval system, not merely a chatbot. It has real MinHash LSH, SimHash, TF-IDF baseline, Streamlit evidence display, PageRank extension, and experiment scripts.

Main risk before these updates: experimental story was weak because MinHash default used very broad one-row LSH bands, SimHash showed near-zero overlap with TF-IDF, and the answer path preferred TF-IDF instead of the LSH-based retrieval path. The current code now makes Hybrid LSH the primary answer path while keeping TF-IDF as the non-approximate baseline.

## Requirement Match

| Requirement | Evaluation |
| --- | --- |
| Data ingestion | Meets. PDFs are parsed with `pdfplumber`, cleaned, chunked, and saved as JSONL. |
| Meaningful chunks | Mostly meets. Current build uses 150-word chunks, below requested 200-500 example range. This is defensible for short policy queries, but mention the tradeoff in report. |
| MinHash + LSH | Meets. Implemented from scratch with shingles, signatures, bands, buckets, and candidate retrieval. |
| SimHash | Meets. Implemented from scratch with weighted bit voting and Hamming similarity. |
| Baseline | Meets. TF-IDF + cosine similarity implemented as full-corpus non-approximate baseline. |
| Query processing | Meets. App compares Hybrid LSH, MinHash, SimHash, and TF-IDF. |
| Answer generation | Meets if LLM key configured. Prompt restricts answer to retrieved excerpts and UI shows sources. |
| Output interface | Meets. Streamlit UI shows answer, chunks, scores, pages, and source references. |
| Extension | Meets. PageRank ranks handbook sections and can boost retrieved chunks. |
| Exact vs approximate experiments | Meets in code. Re-run benchmarks after updates so CSVs include Hybrid LSH. |
| Parameter sensitivity | Meets in code. MinHash permutations/bands and SimHash threshold are tested. |
| Scalability test | Meets in code. Corpus duplication test exists. |

## What To Change Before Submission

1. Rebuild indexes after the MinHash default change:

```powershell
python build_index.py
```

2. Re-run experiments so report numbers match current code:

```powershell
python experiments/benchmark.py
python experiments/plots.py
```

3. Fill `experiments/manual_evaluation_template.csv` during demo/testing. This covers the required qualitative correctness evaluation.

4. In the report, do not call TF-IDF "human ground truth." Call it the exact/non-approximate lexical baseline. For qualitative correctness, manually judge 10-15 queries.

5. Mention chunk size tradeoff. Requirement suggests 200-500 words, but 150-word chunks improve short-query matching and citation focus. If instructor is strict, change `chunk_size=150` to `chunk_size=250` in `build_index.py` and rebuild.

6. Keep chatbot framing minimal. Say: "LLM is only a synthesis layer over retrieved evidence."

## Recommended Report Claims

Use these claims only after re-running experiments:

| Claim | Evidence |
| --- | --- |
| Approximate retrieval trades accuracy for candidate reduction | `exp1_comparison.csv`, `exp2a_minhash_sensitivity.csv` |
| More bands usually increase recall/candidates and may increase latency | `exp2a_minhash_sensitivity.csv` |
| Smaller SimHash Hamming threshold is stricter and may reduce recall | `exp2b_simhash_sensitivity.csv` |
| Corpus duplication shows scalability trend | `exp3_scalability.csv` |
| Hybrid LSH improves answer quality by reranking approximate candidates | `exp1_comparison.csv` and Streamlit demo |

## Demo Talking Points

1. "The system first parses handbooks into cited chunks."
2. "MinHash LSH and SimHash produce approximate candidate sets."
3. "TF-IDF is kept as exact baseline for comparison, not as a bypass."
4. "Hybrid LSH is primary: approximate candidate generation first, reranking second."
5. "Answer synthesis receives only retrieved chunks, and evidence is always displayed."
