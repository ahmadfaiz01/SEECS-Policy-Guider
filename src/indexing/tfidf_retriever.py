"""
tfidf_retriever.py
This is the "old reliable" non-approximate baseline. It compares the query against
every chunk vector with cosine similarity, so it is exact with respect to the
TF-IDF representation. It is not human relevance ground truth, but it gives a
stable baseline for MinHash/SimHash tradeoff experiments.
"""
import pickle
import numpy as np
import sys

_blocked_pyarrow = False
try:
    import pyarrow  # noqa: F401
except Exception:
    # sklearn treats missing pyarrow gracefully, but this Windows policy error
    # happens during sklearn import unless pyarrow is temporarily hidden.
    sys.modules["pyarrow"] = None
    _blocked_pyarrow = True

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Sequence, Tuple

if _blocked_pyarrow and sys.modules.get("pyarrow") is None:
    del sys.modules["pyarrow"]


class TFIDFRetriever:
    def __init__(self, max_features: int = 50_000):
        # We cap the vocabulary so memory doesn't explode
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),   # Grab single words and pairs (bigrams)
            sublinear_tf=True,    # Dampens the effect of words appearing way too many times
        )
        self.chunk_ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._matrix = None
        self._fitted = False

    def fit(self, chunk_ids: List[str], texts: List[str]) -> None:
        """
        Builds the massive TF-IDF matrix for all our chunks.
        This takes a moment...
        """
        self.chunk_ids = list(chunk_ids)
        self._id_to_index = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self._matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        The downside: it has to compare the query vector against every single chunk vector.
        This is what we show in our scalability experiments as the "bad" curve.
        """
        if not self._fitted:
            raise RuntimeError("Whoops, run fit() before trying to query.")

        qvec = self.vectorizer.transform([text])
        # Calculate cosine similarity against everything
        sims = cosine_similarity(qvec, self._matrix).flatten()
        
        # Grab the highest scores
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunk_ids[i], float(sims[i])) for i in top_idx]

    def score_candidates(
        self,
        text: str,
        candidate_ids: Sequence[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Rerank only a candidate subset.
        Hybrid approximate retrieval uses LSH/SimHash to avoid a full scan, then uses
        the same cosine scoring as the baseline inside that much smaller candidate set.
        """
        if not self._fitted:
            raise RuntimeError("Whoops, run fit() before trying to query.")
        if not candidate_ids:
            return []

        self._ensure_id_to_index()
        pairs = [
            (cid, self._id_to_index[cid])
            for cid in dict.fromkeys(candidate_ids)
            if cid in self._id_to_index
        ]
        if not pairs:
            return []

        qvec = self.vectorizer.transform([text])
        row_idx = [idx for _, idx in pairs]
        sims = cosine_similarity(qvec, self._matrix[row_idx]).flatten()
        order = np.argsort(sims)[::-1][:top_k]
        return [(pairs[i][0], float(sims[i])) for i in order]

    def _ensure_id_to_index(self) -> None:
        # Older pickle files may not contain this helper map.
        if not getattr(self, "_id_to_index", None):
            self._id_to_index = {cid: i for i, cid in enumerate(self.chunk_ids)}

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TFIDFRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)

    def stats(self) -> Dict:
        return {
            "num_chunks": len(self.chunk_ids),
            "vocab_size": len(self.vectorizer.vocabulary_) if self._fitted else 0,
            "matrix_shape": list(self._matrix.shape) if self._matrix is not None else None,
        }
