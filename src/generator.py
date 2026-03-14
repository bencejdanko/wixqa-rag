"""
generator.py
============
Answer generation using an OpenAI-compatible chat LLM (OpenRouter).

Prompting strategies
--------------------
- "basic"        : Simple retrieval → answer prompt.
- "cot"          : Chain-of-thought — ask the model to reason step-by-step.
- "cite"         : Instructed to cite the source article URLs inline.
- "compression"  : Compress / summarise retrieved context before generating.
"""

from __future__ import annotations
from typing import List, Tuple, Dict

from src.chunker import Chunk


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are a helpful Wix support assistant. "
    "Answer questions using ONLY the information from the provided context. "
    "If the context does not contain enough information to answer fully, "
    "say so clearly. Do NOT make up information."
)

_BASIC_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""

_COT_TEMPLATE = """\
Context:
{context}

Question: {question}

Think step by step before writing your final answer.

Step-by-step reasoning:
1."""

_CITE_TEMPLATE = """\
Context (each passage is preceded by its source URL):
{context_with_urls}

Question: {question}

Provide a clear, complete answer. Cite the source URL(s) inline where relevant using [URL] notation.

Answer:"""

_COMPRESSION_SYSTEM = (
    "You are a document compressor. "
    "Extract only the sentences from the context that are directly relevant "
    "to answering the question. Output only the compressed context, nothing else."
)

_COMPRESSION_COMPRESS_TEMPLATE = """\
Context:
{context}

Question: {question}

Relevant sentences:"""

_COMPRESSION_ANSWER_TEMPLATE = """\
Relevant Context:
{compressed}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """
    Generate answers from retrieved chunks using a chat LLM.

    Parameters
    ----------
    client           : openai.OpenAI-compatible client
    model            : str   – LLM model identifier
    strategy         : str   – "basic" | "cot" | "cite" | "compression"
    max_context_chars: int   – hard cap on context length
    max_tokens       : int   – max tokens for the generated answer
    temperature      : float – sampling temperature
    """

    STRATEGIES = ("basic", "cot", "cite", "compression")

    def __init__(
        self,
        client,
        model: str = "google/gemini-2.0-flash-001",
        strategy: str = "basic",
        max_context_chars: int = 6000,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.client = client
        self.model = model
        self.strategy = strategy
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_context(self, chunks: List[Tuple[Chunk, float]]) -> str:
        """Concatenate chunk texts, truncating at max_context_chars."""
        parts = []
        total = 0
        for chunk, _ in chunks:
            if total + len(chunk.text) > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining > 100:
                    parts.append(chunk.text[:remaining])
                break
            parts.append(chunk.text)
            total += len(chunk.text)
        return "\n\n---\n\n".join(parts)

    def _context_with_urls(self, chunks: List[Tuple[Chunk, float]]) -> str:
        parts = []
        total = 0
        for chunk, _ in chunks:
            header = f"[Source: {chunk.metadata.get('url', 'N/A')}]\n"
            entry = header + chunk.text
            if total + len(entry) > self.max_context_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n---\n\n".join(parts)

    def _chat(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
    ) -> Dict:
        """
        Generate an answer given a query and retrieved chunks.

        Returns
        -------
        Dict with keys:
            "answer"   : str  – generated answer
            "strategy" : str  – prompting strategy used
            "model"    : str  – model identifier
            "n_chunks" : int  – number of chunks used
        """
        if self.strategy == "basic":
            context = self._truncate_context(retrieved)
            prompt = _BASIC_TEMPLATE.format(context=context, question=query)
            answer = self._chat(_SYSTEM_BASE, prompt)

        elif self.strategy == "cot":
            context = self._truncate_context(retrieved)
            prompt = _COT_TEMPLATE.format(context=context, question=query)
            answer = self._chat(_SYSTEM_BASE, prompt)

        elif self.strategy == "cite":
            ctx_urls = self._context_with_urls(retrieved)
            prompt = _CITE_TEMPLATE.format(
                context_with_urls=ctx_urls, question=query
            )
            answer = self._chat(_SYSTEM_BASE, prompt)

        elif self.strategy == "compression":
            context = self._truncate_context(retrieved)
            # Step 1: compress context
            compress_prompt = _COMPRESSION_COMPRESS_TEMPLATE.format(
                context=context, question=query
            )
            compressed = self._chat(_COMPRESSION_SYSTEM, compress_prompt)
            # Step 2: generate answer from compressed context
            answer_prompt = _COMPRESSION_ANSWER_TEMPLATE.format(
                compressed=compressed, question=query
            )
            answer = self._chat(_SYSTEM_BASE, answer_prompt)

        else:
            answer = ""

        return {
            "answer": answer,
            "strategy": self.strategy,
            "model": self.model,
            "n_chunks": len(retrieved),
        }
