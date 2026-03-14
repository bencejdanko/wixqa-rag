"""
reranker.py
===========
Cross-encoder reranking of retrieved chunks.

After initial bi-encoder retrieval returns a candidate set, the cross-encoder
scores every (query, passage) pair jointly for higher-precision ranking.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, strong)
"""

from __future__ import annotations
from typing import List, Tuple

from src.chunker import Chunk


class CrossEncoderReranker:
    """
    Re-rank retrieved (chunk, score) pairs using a cross-encoder model.

    Parameters
    ----------
    model_name : str   – HuggingFace cross-encoder model
    top_n      : int   – number of passages to keep after reranking
    device     : str   – "cuda" | "cpu" | None (auto)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        device: str | None = None,
    ):
        from sentence_transformers.cross_encoder import CrossEncoder
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading cross-encoder '{model_name}' on {device}…")
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_n: int | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Re-score and re-rank a list of (Chunk, score) pairs.

        Parameters
        ----------
        query      : str                       – original query
        candidates : List[(Chunk, float)]      – initial retrieval results
        top_n      : int | None                – override; default is self.top_n

        Returns
        -------
        List[(Chunk, float)] sorted by descending cross-encoder score.
        """
        n = top_n or self.top_n
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        ce_scores = self.model.predict(pairs)

        scored = sorted(
            zip([c for c, _ in candidates], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in scored[:n]]
