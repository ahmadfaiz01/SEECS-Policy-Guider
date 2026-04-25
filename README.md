# SEECS Policy QA System

A scalable, academic policy question-answering system over the NUST SEECS Undergraduate and Postgraduate handbooks. 

This project is built as a **strict Retrieval-Augmented Generation (RAG) pipeline**. Instead of relying on an LLM's internal memory (which risks hallucinations), the system uses **MinHash LSH, SimHash, and TF-IDF** to quickly retrieve relevant exact handbook excerpts as evidence. Only these verified excerpts are sent to the LLM to synthesize an answer.

---

## 🏗️ System Architecture

```mermaid
graph TD
    %% Ingestion
    subgraph Data Ingestion
        A[Handbook PDFs] --> B[pdfplumber & regex cleaning]
        B --> C[Chunking 150-word passages]
        C --> D[(chunks.jsonl)]
    end

    %% Indexing
    subgraph Indexing Algorithms
        D --> E[TF-IDF Matrix]
        D --> F[MinHash LSH Signatures]
        D --> G[SimHash Fingerprints]
        D --> H[PageRank Network Graph]
    end

    %% Query Pipeline
    subgraph Query Execution
        I[User Question] --> J{Retrieval Strategy}
        J -->|Exact Baseline| K[TF-IDF Cosine Similarity]
        J -->|Approximate| L[MinHash LSH / SimHash]
        
        K --> M[Candidate Excerpts]
        L --> M
        
        H -.->|Authority Boost| M
    end

    %% Generation & UI
    subgraph Output
        M --> N[Groq API / Llama-3]
        N --> O[Streamlit UI]
        M -.->|Displays Evidence| O
    end
```

---

## ✅ Requirement Coverage

| Requirement | Status | Where |
| --- | --- | --- |
| PDF ingestion and cleaning | Implemented | `src/ingestion/`, `build_index.py` |
| 200-500 word chunking target | Implemented with tunable chunking | `chunk_pages(..., chunk_size=150, overlap=50)` |
| MinHash + LSH | Implemented from scratch | `src/indexing/minhash_lsh.py` |
| SimHash + Hamming distance | Implemented from scratch | `src/indexing/simhash.py` |
| TF-IDF cosine baseline | Implemented | `src/indexing/tfidf_retriever.py` |
| Query-time comparison | Implemented | `src/retrieval/pipeline.py`, `app.py` |
| Grounded answer generation | Implemented with optional Groq API | `src/generation/answer_gen.py` |
| Evidence and page/source display | Implemented | `app.py` |
| Competitive extension | PageRank section authority boost | `src/extensions/pagerank.py` |
| Required experiments | Implemented | `experiments/benchmark.py` |

## 📂 Project Structure

```text
src/
  ingestion/       PDF parsing, cleaning, and context-aware chunking
  indexing/        MinHash LSH, SimHash, and exact TF-IDF baseline
  retrieval/       Query pipeline with strict latency and memory tracking
  generation/      Grounded LLM answer synthesis (Groq API)
  extensions/      PageRank computation for policy cross-references
experiments/
  benchmark.py     Exact vs approximate, sensitivity, and scalability testing
  plots.py         Figure generation from benchmark results
data/
  processed/       Generated jsonl text chunks
  indexes/         Generated index binaries (not tracked in Git)
app.py             Streamlit dashboard
build_index.py     Generates data/ and indexes from raw PDFs
```

## 🚀 Quick Start (Local & Cloud)

**Local Deployment:**
```powershell
pip install -r requirements.txt
copy .env.example .env
python build_index.py
streamlit run app.py
```
*Note: Add your `GROQ_API_KEY` to `.env` if you want synthesized answers. Retrieval and analytics work purely locally without the LLM key.*

**Streamlit Community Cloud Deployment:**
This repository is pre-configured for automatic deployment on Streamlit Cloud. 
1. Go to share.streamlit.io and connect your GitHub.
2. Select your repository and set `app.py` as the main file.
3. In the Advanced Settings **Secrets** block, add `GROQ_API_KEY="your_key"`.
4. Deploy! The app will automatically run `build_index.py` on startup to generate the indexes on the cloud server.

## 🧪 Experiments

```powershell
python experiments/benchmark.py
python experiments/plots.py
```

Generated CSVs:

| File | Purpose |
| --- | --- |
| `exp1_comparison.csv` | TF-IDF vs Hybrid LSH vs MinHash vs SimHash latency, memory, Precision@5 |
| `exp2a_minhash_sensitivity.csv` | Effect of hash functions and bands |
| `exp2b_simhash_sensitivity.csv` | Effect of Hamming threshold |
| `exp3_scalability.csv` | Corpus duplication scalability test |
| `manual_evaluation_template.csv` | Blank template for manual human evaluation in final report |
