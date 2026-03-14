"""
vector_store.py
===============
FAISS-backed vector store for fast approximate nearest-neighbour retrieval.

Supports two index types:
  - "flat"  : Exact search (IndexFlatIP) — perfect recall, best for < 500k vecs
  - "ivf"   : Approximate search (IndexIVFFlat) — faster for large corpora
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import faiss

from src.chunker import Chunk


class VectorStore:
    """
    Build, search, and optionally persist a FAISS index over text chunks.

    Parameters
    ----------
    dim        : int   – embedding dimensionality
    index_type : str   – "flat" | "ivf"
    nlist      : int   – number of IVF clusters (only used for index_type="ivf")
    nprobe     : int   – number of IVF clusters to visit at query time
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        nlist: int = 200,
        nprobe: int = 20,
    ):
        self.dim = dim
        self.index_type = index_type
        self._chunks: List[Chunk] = []

        if index_type == "flat":
            # Inner-product index (works with L2-normalised vectors = cosine sim)
            self._index = faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            quantiser = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(
                quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._nlist = nlist
            self._nprobe = nprobe
        else:
            raise ValueError("index_type must be 'flat' or 'ivf'")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunk embeddings to the index.

        Parameters
        ----------
        chunks     : List[Chunk] in the same order as embeddings
        embeddings : float32 ndarray, shape (N, dim)
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings must have equal length."
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self.index_type == "ivf" and not self._index.is_trained:
            print(f"Training IVF index with {len(embeddings):,} vectors…")
            self._index.train(embeddings)
            self._index.nprobe = self._nprobe

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        print(f"  → Index now holds {self._index.ntotal:,} vectors.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks by cosine similarity.

        Parameters
        ----------
        query_embedding : float32 ndarray, shape (1, dim) or (dim,)
        top_k           : number of results

        Returns
        -------
        List of (Chunk, score) tuples, ranked by descending score.
        """
        qe = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )
        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(qe, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist both the FAISS index and chunk metadata to disk."""
        import pickle, os
        faiss.write_index(self._index, path + ".faiss")
        with open(path + ".chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        print(f"VectorStore saved to {path}.*")

    @classmethod
    def load(cls, path: str, dim: int, index_type: str = "flat") -> "VectorStore":
        """Restore a previously saved VectorStore."""
        import pickle
        store = cls(dim=dim, index_type=index_type)
        store._index = faiss.read_index(path + ".faiss")
        with open(path + ".chunks.pkl", "rb") as f:
            store._chunks = pickle.load(f)
        print(f"VectorStore loaded: {store._index.ntotal:,} vectors.")
        return store

    @property
    def size(self) -> int:
        return self._index.ntotal
