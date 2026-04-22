"""
simhash.py
SimHash generates a fixed-size fingerprint (like a 64-bit ID) for each chunk.
If two chunks are similar, their fingerprints will be similar too.

How it works:
  1. Look at each word and give it a weight (TF-IDF style).
  2. Hash the word.
  3. The word "votes" on the bits of the fingerprint based on its weight.
  4. Squish all the votes down to a single 64-bit number.
"""
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional


class SimHashIndex:
    def __init__(self, hash_bits: int = 64, hamming_threshold: Optional[int] = 10):
        self.hash_bits = hash_bits
        # How many bits can be different before we say "nah, not similar enough"
        self.hamming_threshold = hamming_threshold
        self.fingerprints: Dict[str, int] = {}
        self._chunk_ids: List[str] = []

    def _token_hash(self, token: str) -> int:
        """Just a quick MD5 hash to turn a word into a number."""
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) & ((1 << self.hash_bits) - 1)

    def _tf_weights(self, text: str) -> Dict[str, float]:
        """
        Counts how many times words appear. We ignore tiny words (like "a" or "is")
        and numbers to keep the focus on the actual content.
        """
        weights: Dict[str, float] = {}
        for token in text.lower().split():
            if len(token) > 2 and not token.isdigit():
                weights[token] = weights.get(token, 0.0) + 1.0
        return weights

    def _simhash(self, text: str) -> int:
        """
        The magic formula. We tally up the votes for each bit position 
        based on the words in the text, and output the final fingerprint.
        """
        votes = np.zeros(self.hash_bits, dtype=np.float64)

        for token, weight in self._tf_weights(text).items():
            h = self._token_hash(token)
            for bit in range(self.hash_bits):
                # If the bit is 1, add weight. If 0, subtract weight.
                votes[bit] += weight if (h >> bit) & 1 else -weight

        fp = 0
        for bit in range(self.hash_bits):
            # If the final vote is positive, the bit is 1. Otherwise 0.
            if votes[bit] > 0:
                fp |= 1 << bit
        return fp

    def hamming(self, fp1: int, fp2: int) -> int:
        """Counts how many bits differ between two fingerprints."""
        return bin(fp1 ^ fp2).count("1")

    def similarity(self, fp1: int, fp2: int) -> float:
        """Turns the Hamming distance into a neat percentage score."""
        return 1.0 - self.hamming(fp1, fp2) / self.hash_bits

    def add(self, chunk_id: str, text: str) -> None:
        """Generates and saves the fingerprint for a chunk."""
        fp = self._simhash(text)
        self.fingerprints[chunk_id] = fp
        self._chunk_ids.append(chunk_id)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        To find matches, we unfortunately have to scan everything (O(n)).
        But comparing fingerprints is literally just an XOR operation, 
        so it's incredibly fast even on thousands of chunks.
        """
        qfp = self._simhash(text)

        scored = [
            (cid, self.similarity(qfp, fp))
            for cid, fp in self.fingerprints.items()
        ]

        # Drop anything that's too different based on our threshold
        if self.hamming_threshold is not None:
            max_sim = 1.0 - self.hamming_threshold / self.hash_bits
            scored = [(cid, s) for cid, s in scored if s >= max_sim]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SimHashIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    def stats(self) -> Dict:
        return {
            "num_chunks": len(self._chunk_ids),
            "hash_bits": self.hash_bits,
            "hamming_threshold": self.hamming_threshold,
        }
