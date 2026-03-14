"""
chunker.py
==========
Chunking strategies for WixQA knowledge-base articles.

Supported strategies
--------------------
- "fixed"     : Fixed-size character / token windows (with overlap)
- "recursive" : LangChain RecursiveCharacterTextSplitter  (sentence-aware)
- "sentence"  : NLTK sentence-boundary splitting
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple
import nltk


# Download NLTK sentence tokenizer once (silent on repeat calls)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class Chunk:
    """A single text chunk derived from a KB article."""

    __slots__ = ("chunk_id", "article_id", "text", "metadata")

    def __init__(
        self,
        chunk_id: str,
        article_id: str,
        text: str,
        metadata: Dict | None = None,
    ):
        self.chunk_id = chunk_id
        self.article_id = article_id
        self.text = text.strip()
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, "
            f"article={self.article_id!r}, "
            f"len={len(self.text)})"
        )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _fixed_chunks(
    text: str, chunk_size: int = 512, overlap: int = 64
) -> List[str]:
    """Split text by fixed character count with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _recursive_chunks(
    text: str, chunk_size: int = 512, overlap: int = 64
) -> List[str]:
    """LangChain-style recursive splitting on paragraph / sentence / word boundaries."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def _sentence_chunks(
    text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1
) -> List[str]:
    """Split by NLTK sentences; group N sentences per chunk with sentence-level overlap."""
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [text]
    chunks = []
    step = max(1, sentences_per_chunk - overlap_sentences)
    for i in range(0, len(sentences), step):
        window = sentences[i : i + sentences_per_chunk]
        chunks.append(" ".join(window).strip())
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

STRATEGIES = ("fixed", "recursive", "sentence")


def chunk_articles(
    articles: List[Dict],
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 64,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> List[Chunk]:
    """
    Chunk a list of KB articles using the specified strategy.

    Parameters
    ----------
    articles          : list of article dicts (must have "id" and "contents")
    strategy          : "fixed" | "recursive" | "sentence"
    chunk_size        : characters per chunk (fixed / recursive)
    overlap           : character overlap   (fixed / recursive)
    sentences_per_chunk : sentences per chunk (sentence strategy)
    overlap_sentences   : sentence overlap   (sentence strategy)

    Returns
    -------
    List[Chunk]
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    all_chunks: List[Chunk] = []
    for article in articles:
        art_id = article["id"]
        text   = article.get("contents", "") or ""
        if not text.strip():
            continue

        if strategy == "fixed":
            raw = _fixed_chunks(text, chunk_size, overlap)
        elif strategy == "recursive":
            raw = _recursive_chunks(text, chunk_size, overlap)
        else:  # sentence
            raw = _sentence_chunks(text, sentences_per_chunk, overlap_sentences)

        for idx, chunk_text in enumerate(raw):
            chunk = Chunk(
                chunk_id=f"{art_id}__chunk_{idx}",
                article_id=art_id,
                text=chunk_text,
                metadata={
                    "article_type": article.get("article_type", ""),
                    "url": article.get("url", ""),
                    "chunk_index": idx,
                    "total_chunks": len(raw),
                },
            )
            all_chunks.append(chunk)

    return all_chunks


def describe_chunks(chunks: List[Chunk]) -> Dict:
    """Return descriptive statistics about a chunk list."""
    if not chunks:
        return {}
    lengths = [len(c.text) for c in chunks]
    import statistics
    return {
        "total_chunks": len(chunks),
        "unique_articles": len({c.article_id for c in chunks}),
        "avg_chars": round(statistics.mean(lengths), 1),
        "median_chars": statistics.median(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
