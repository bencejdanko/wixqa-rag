"""
retriever.py
============
High-level retrieval interface wrapping VectorStore + Embedder.

Also contains:
  - BM25Retriever  : sparse keyword-based retrieval baseline
  - HybridRetriever: reciprocal-rank fusion (RRF) of dense + sparse results
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from src.chunker import Chunk
from src.embedder import Embedder
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Encode a query and search the FAISS index for the nearest chunks.

    Parameters
    ----------
    embedder     : Embedder      – for encoding queries
    vector_store : VectorStore   – pre-built FAISS index
    top_k        : int           – default number of results
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> List[Tuple[Chunk, float]]:
        k = top_k or self.top_k
        qe = self.embedder.encode_query(query)
        return self.vector_store.search(qe, top_k=k)


# ---------------------------------------------------------------------------
# BM25 sparse retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Sparse TF-IDF / BM25-style retrieval using sklearn's TfidfVectorizer.
    Serves as a keyword baseline and one half of the hybrid retriever.
    """

    def __init__(self, chunks: List[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse as sp

        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.9,
            min_df=2,
            ngram_range=(1, 2),
        )
        corpus = [c.text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(corpus)  # (N, vocab)

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        import scipy.sparse as sp
        qv = self._vectorizer.transform([query])
        scores = (self._matrix @ qv.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Hybrid retriever (RRF fusion)
# ---------------------------------------------------------------------------

def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion weight."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Combine dense + sparse retrievers with Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    dense  : DenseRetriever
    sparse : BM25Retriever
    alpha  : float – weight towards dense (1.0 = pure dense, 0.0 = pure sparse)
    top_k  : int   – final number of results after fusion
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        alpha: float = 0.7,
        top_k: int = 5,
    ):
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha
        self.top_k = top_k

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> List[Tuple[Chunk, float]]:
        k = top_k or self.top_k
        fetch_k = max(k * 4, 20)

        dense_results = self.dense.retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse.retrieve(query, top_k=fetch_k)

        scores: dict[str, float] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            scores[chunk.chunk_id] = (
                scores.get(chunk.chunk_id, 0.0)
                + self.alpha * _rrf_score(rank)
            )

        for rank, (chunk, _) in enumerate(sparse_results):
            scores[chunk.chunk_id] = (
                scores.get(chunk.chunk_id, 0.0)
                + (1 - self.alpha) * _rrf_score(rank)
            )

        # Build lookup for chunks seen in either list
        chunk_lookup: dict[str, Chunk] = {
            c.chunk_id: c for c, _ in dense_results + sparse_results
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(chunk_lookup[cid], score) for cid, score in ranked]
