"""
pipeline.py
The traffic controller. This ties together MinHash, SimHash, and TF-IDF.
It loads the data from disk, routes your query to the right index, 
and times exactly how long it takes and how much memory it uses.
"""
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from indexing.minhash_lsh import MinHashLSH
from indexing.simhash import SimHashIndex
from indexing.tfidf_retriever import TFIDFRetriever

Method = Literal["hybrid_lsh", "minhash", "simhash", "tfidf"]


class QueryResult:
    """A neat little package containing the results, timing, and memory info."""
    def __init__(
        self,
        method: str,
        chunks: List[Dict],
        scores: List[float],
        latency_ms: float,
        memory_kb: float,
    ):
        self.method = method
        self.chunks = chunks
        self.scores = scores
        self.latency_ms = latency_ms
        self.memory_kb = memory_kb

    def to_dict(self) -> Dict:
        """For easier debugging or saving to JSON."""
        return {
            "method": self.method,
            "latency_ms": round(self.latency_ms, 3),
            "memory_kb": round(self.memory_kb, 2),
            "hits": [
                {"score": round(s, 4), **{k: v for k, v in c.items() if k != "text"}}
                for c, s in zip(self.chunks, self.scores)
            ],
        }


class RetrievalPipeline:
    def __init__(self, index_dir: str, chunks_path: str):
        self._chunks: Dict[str, Dict] = {}
        self._load_chunks(chunks_path)

        self.minhash: Optional[MinHashLSH] = None
        self.simhash: Optional[SimHashIndex] = None
        self.tfidf: Optional[TFIDFRetriever] = None
        
        # Pull in whatever indexes we successfully built earlier
        self._load_indexes(Path(index_dir))

    def _load_chunks(self, path: str) -> None:
        """Load the raw text chunks so we can actually show them to the user."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self._chunks[chunk["chunk_id"]] = chunk
        print(f"[Pipeline] Sweet, loaded {len(self._chunks)} chunks.")

    def _load_indexes(self, index_dir: Path) -> None:
        """Try to load each index. Don't crash if one is missing, just warn."""
        for name, cls, attr in [
            ("minhash_lsh.pkl", MinHashLSH, "minhash"),
            ("simhash.pkl", SimHashIndex, "simhash"),
            ("tfidf.pkl", TFIDFRetriever, "tfidf"),
        ]:
            path = index_dir / name
            if path.exists():
                index = cls.load(str(path))
                setattr(self, attr, index)
                print(f"[Pipeline] Found {attr}: {index.stats()}")
            else:
                print(f"[Pipeline] Just a heads up: {path} is missing. {attr} won't work.")

    def query(self, question: str, method: Method = "minhash", top_k: int = 5) -> QueryResult:
        """
        Runs a query and aggressively tracks time and memory. 
        This is super useful for our experiments.
        """
        # Start the stopwatch and memory tracker
        tracemalloc.start()
        t0 = time.perf_counter()

        # Let the specific method do the heavy lifting
        raw = self._dispatch(question, method, top_k)

        # Stop tracking
        latency_ms = (time.perf_counter() - t0) * 1000
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Attach the actual text to the IDs we got back
        chunks = [self._chunks.get(cid, {"chunk_id": cid, "text": "[missing]"}) for cid, _ in raw]
        scores = [s for _, s in raw]

        return QueryResult(method, chunks, scores, latency_ms, mem_peak / 1024)

    def query_all(self, question: str, top_k: int = 5) -> Dict[str, QueryResult]:
        """Fire off the query to every available method at once."""
        available: List[Method] = []
        if self.minhash and self.simhash: available.append("hybrid_lsh")
        if self.minhash: available.append("minhash")
        if self.simhash: available.append("simhash")
        if self.tfidf:   available.append("tfidf")
        return {m: self.query(question, m, top_k) for m in available}

    def _dispatch(self, question: str, method: Method, top_k: int) -> List[Tuple[str, float]]:
        """Helper to call the correct index based on the requested method."""
        if method == "hybrid_lsh":
            return self._query_hybrid_lsh(question, top_k)
        if method == "minhash":
            if not self.minhash: raise RuntimeError("MinHash index isn't loaded!")
            return self.minhash.query(question, top_k)
        if method == "simhash":
            if not self.simhash: raise RuntimeError("SimHash index isn't loaded!")
            return self.simhash.query(question, top_k)
        if method == "tfidf":
            if not self.tfidf: raise RuntimeError("TF-IDF index isn't loaded!")
            return self.tfidf.query(question, top_k)
        raise ValueError(f"Wait, what method is {method}?")

    def _query_hybrid_lsh(self, question: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Approximate-first retrieval:
        1. MinHash LSH proposes candidates from shared buckets.
        2. SimHash proposes near fingerprints by Hamming similarity.
        3. Candidate-only TF-IDF reranks the small pool for answer quality.

        This keeps the required LSH/SimHash retrieval in the critical path while
        avoiding a chatbot-style direct answer over the whole PDF.
        """
        if not self.minhash and not self.simhash:
            raise RuntimeError("No approximate indexes are loaded!")

        candidate_limit = max(50, top_k * 30)
        scores: Dict[str, float] = {}

        if self.minhash:
            for cid, score in self.minhash.query(question, top_k=candidate_limit):
                scores[cid] = max(scores.get(cid, 0.0), 0.60 * score)

        if self.simhash:
            for cid, score in self.simhash.query(question, top_k=candidate_limit):
                scores[cid] = max(scores.get(cid, 0.0), 0.40 * score)

        candidate_ids = list(scores.keys())
        if self.tfidf and candidate_ids:
            return self.tfidf.score_candidates(question, candidate_ids, top_k)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> List[Dict]:
        return list(self._chunks.values())

    @property
    def available_methods(self) -> List[Method]:
        methods: List[Method] = []
        if self.minhash and self.simhash: methods.append("hybrid_lsh")
        if self.minhash: methods.append("minhash")
        if self.simhash: methods.append("simhash")
        if self.tfidf:   methods.append("tfidf")
        return methods
