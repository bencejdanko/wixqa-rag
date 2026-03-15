"""
Fixed-size tokenization, pretty straightforward, we 
also allow overlap

For semantic chunking,

There are other techniques, like recursive character
splitting. Sometimes heuristics on known document
structures can perform better

https://docs.langchain.com/oss/python/integrations/splitters
langchain-text-splitters - some heuristic algorithms that 
may also work
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter


nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)


class Chunk:
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

# we return fixed size chunks that have optional 
# overlap with eachother
def _fixed_chunks(
    text: str, chunk_size: int = 512, overlap: int = 64
) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

# helper for gathering sentences for semantic chunkin 
def get_sentences(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# Split text recursively by looking for a list of 
# separators in order (paragraphs \n\n, newlines \n, 
# sentence endings . , words  ).
#
# attempts to keep sentences whole
# without requiring a tokenizers
def _recursive_chunks(
    text: str, chunk_size: int = 512, overlap: int = 64
) -> List[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]
