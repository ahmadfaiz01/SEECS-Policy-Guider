"""
minhash_lsh.py
This is the core of the project.
MinHash estimates how similar two texts are without having to compare every single word.
LSH (Locality Sensitive Hashing) groups similar texts into "buckets" so we don't have to 
search the entire dataset when someone asks a question.

It's all about trading a tiny bit of accuracy for a massive speedup!
"""
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# We need a big prime number for our hash functions to avoid collisions
_PRIME: int = (1 << 61) - 1
_MOD: int = (1 << 32)  # keeps our shingle hashes manageable (32-bit)


class MinHashLSH:
    def __init__(self, num_perm: int = 128, num_bands: int = 16, shingle_k: int = 3):
        # num_perm has to divide evenly by num_bands, otherwise the math gets messy
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.shingle_k = shingle_k

        # Generate random coefficients for our hash functions: h(x) = (a*x + b) % P
        # Using a fixed seed so results are reproducible
        rng = np.random.RandomState(42)
        self.hash_a = rng.randint(1, _PRIME, size=num_perm, dtype=np.int64)
        self.hash_b = rng.randint(0, _PRIME, size=num_perm, dtype=np.int64)

        # Create our LSH buckets (one hash table per band)
        self.band_tables: List[Dict[tuple, List[str]]] = [
            defaultdict(list) for _ in range(num_bands)
        ]
        
        # Save the full signatures so we can rank the candidates later
        self.signatures: Dict[str, np.ndarray] = {}

    def _shingle(self, text: str) -> List[int]:
        """
        Breaks text into overlapping groups of words (shingles) and hashes them.
        For example, "the quick brown fox" -> "the quick brown", "quick brown fox".
        """
        import zlib
        words = text.lower().split()
        if not words:
            return []
        return [
            zlib.crc32(" ".join(words[i: i + self.shingle_k]).encode("utf-8")) % _MOD
            for i in range(max(1, len(words) - self.shingle_k + 1))
        ]

    def _minhash(self, shingles: List[int]) -> np.ndarray:
        """
        Creates the MinHash signature.
        We run all the shingles through our hash functions and keep the minimum value.
        This gives us a fixed-size signature that approximates the Jaccard similarity.
        """
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.int64)

        shingle_arr = np.array(shingles, dtype=np.int64)
        sig = np.full(self.num_perm, _PRIME, dtype=np.int64)

        # Numpy magic to process all shingles at once (way faster than python loops)
        for x in shingle_arr:
            vals = (self.hash_a * int(x) + self.hash_b) % _PRIME
            np.minimum(sig, vals, out=sig)

        return sig

    def _band_key(self, sig: np.ndarray, band: int) -> tuple:
        """Slices the signature for a specific band to use as a dictionary key."""
        s = band * self.rows_per_band
        return tuple(sig[s: s + self.rows_per_band].tolist())

    def add(self, chunk_id: str, text: str) -> None:
        """Indexes a chunk of text."""
        sig = self._minhash(self._shingle(text))
        self.signatures[chunk_id] = sig
        
        # Throw it into the right bucket for each band
        for b in range(self.num_bands):
            self.band_tables[b][self._band_key(sig, b)].append(chunk_id)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Finds the most similar chunks to the query.
        Instead of comparing the query against everything, we only check chunks 
        that fell into the same buckets!
        """
        sig = self._minhash(self._shingle(text))

        # Step 1: Find candidate chunks (anyone in the same bucket)
        candidates: set = set()
        for b in range(self.num_bands):
            key = self._band_key(sig, b)
            candidates.update(self.band_tables[b].get(key, []))

        if not candidates:
            return []

        # Step 2: Actually calculate the estimated similarity for these candidates
        scored = [
            (cid, float(np.mean(sig == self.signatures[cid])))
            for cid in candidates
        ]
        
        # Sort by best score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def save(self, path: str) -> None:
        """Saves the index to disk so we don't have to rebuild it every time."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "MinHashLSH":
        with open(path, "rb") as f:
            return pickle.load(f)

    def stats(self) -> Dict:
        return {
            "num_chunks": len(self.signatures),
            "num_perm": self.num_perm,
            "num_bands": self.num_bands,
            "rows_per_band": self.rows_per_band,
        }
