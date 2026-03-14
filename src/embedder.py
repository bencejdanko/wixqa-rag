"""
embedder.py
===========
Wrapper around sentence-transformers for batch embedding of chunks and queries.
Supports GPU acceleration when available (Colab T4/A100).
"""

from __future__ import annotations
from typing import List, Union
import numpy as np
from tqdm.auto import tqdm

# Default model: strong multilingual-capable model available on HuggingFace
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """
    Encode texts to dense vectors using a sentence-transformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path.
    batch_size : int
        Number of texts to encode per forward pass.
    device : str | None
        "cuda", "cpu", or None (auto-detect).
    normalize : bool
        L2-normalise embeddings (enables cosine similarity via dot-product).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 256,
        device: str | None = None,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{model_name}' on {device}…")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  → embedding dim = {self.dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings → float32 numpy array shaped (N, dim).

        Returns normalised vectors when self.normalize=True.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → float32 array shaped (1, dim)."""
        return self.encode([query])
