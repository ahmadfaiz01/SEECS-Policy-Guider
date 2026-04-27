"""
build_index.py
Run once before starting the app.

It parses handbook PDFs, cleans text, chunks content, and builds:
- MinHash LSH
- SimHash
- TF-IDF baseline
- PageRank section graph
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from extensions.pagerank import HandbookPageRank
from indexing.minhash_lsh import MinHashLSH
from indexing.simhash import SimHashIndex
from indexing.tfidf_retriever import TFIDFRetriever
from ingestion.chunker import chunk_pages
from ingestion.pdf_parser import parse_pdf
from ingestion.preprocessor import clean_pages

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/indexes")
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def ingest_pdfs() -> list:
    """Read raw PDFs and turn them into searchable chunks."""
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDFs found in {RAW_DIR}. Add the handbook PDFs first.")
        sys.exit(1)

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"\n[Ingestion] Processing {pdf_path.name}...")
        pages = parse_pdf(str(pdf_path))
        pages = clean_pages(pages)
        chunks = chunk_pages(pages, chunk_size=150, overlap=50)
        print(f"  -> Extracted {len(pages)} pages into {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n[Ingestion] Saved {len(all_chunks)} chunks to {CHUNKS_FILE}")
    return all_chunks


def build_minhash(chunks: list) -> None:
    print("\n[MinHash+LSH] Generating signatures and banding...")
    t0 = time.perf_counter()
    idx = MinHashLSH(num_perm=256, num_bands=128, shingle_k=1)
    for chunk in chunks:
        idx.add(chunk["chunk_id"], chunk["text"])
    idx.save(str(INDEX_DIR / "minhash_lsh.pkl"))
    print(f"  -> Done in {time.perf_counter() - t0:.2f}s | {idx.stats()}")


def build_simhash(chunks: list) -> None:
    print("\n[SimHash] Computing 64-bit fingerprints...")
    t0 = time.perf_counter()
    idx = SimHashIndex(hash_bits=64, hamming_threshold=None)
    for chunk in chunks:
        idx.add(chunk["chunk_id"], chunk["text"])
    idx.save(str(INDEX_DIR / "simhash.pkl"))
    print(f"  -> Done in {time.perf_counter() - t0:.2f}s | {idx.stats()}")


def build_tfidf(chunks: list) -> None:
    print("\n[TF-IDF] Building exact baseline matrix...")
    t0 = time.perf_counter()
    idx = TFIDFRetriever(max_features=50_000)
    idx.fit([c["chunk_id"] for c in chunks], [c["text"] for c in chunks])
    idx.save(str(INDEX_DIR / "tfidf.pkl"))
    print(f"  -> Done in {time.perf_counter() - t0:.2f}s | {idx.stats()}")


def build_pagerank(chunks: list) -> None:
    print("\n[PageRank] Finding cross-references and running PageRank...")
    t0 = time.perf_counter()
    pr = HandbookPageRank(alpha=0.85)
    pr.build(chunks)
    pr.save(str(INDEX_DIR / "pagerank.pkl"))
    print(f"  -> Done in {time.perf_counter() - t0:.2f}s | Top 3: {pr.top_sections(3)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SEECS Policy QA - Initializing System")
    print("=" * 60)

    chunks = ingest_pdfs()
    build_minhash(chunks)
    build_simhash(chunks)
    build_tfidf(chunks)
    build_pagerank(chunks)

    print("\nAll systems go. Indexes are built and saved.")
    print(f"Indexes saved to: {INDEX_DIR.resolve()}")
    print("\nRun the app with:")
    print("streamlit run app.py")
