"""
query_rewriter.py
=================
Query rewriting strategies to improve retrieval recall.

Strategies
----------
- "none"         : Pass-through — no rewriting.
- "hyd"          : HyDE (Hypothetical Document Embeddings) — generate a short
                   hypothetical answer and embed that instead of the question.
- "step_back"    : Step-back prompting — abstract the question to a broader concept.
- "multi_query"  : Generate N paraphrases; retrieve for each; de-duplicate results.
"""

from __future__ import annotations
from typing import List, Tuple

from src.chunker import Chunk


# ---------------------------------------------------------------------------
# Shared LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, client, model: str, max_tokens: int = 256) -> str:
    """Utility: call OpenRouter-compatible chat completion and return text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Query rewriting implementations
# ---------------------------------------------------------------------------

class QueryRewriter:
    """
    Wraps different query-rewriting strategies behind a common interface.

    Parameters
    ----------
    strategy : str      – "none" | "hyd" | "step_back" | "multi_query"
    client   : openai.OpenAI-compatible client (e.g., OpenRouter)
    model    : str      – LLM model name used for rewriting
    n_queries: int      – number of paraphrases for multi_query strategy
    """

    STRATEGIES = ("none", "hyd", "step_back", "multi_query")

    def __init__(
        self,
        strategy: str = "none",
        client=None,
        model: str = "google/gemini-2.0-flash-001",
        n_queries: int = 3,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        if strategy != "none" and client is None:
            raise ValueError("A client must be provided for non-trivial rewriting strategies.")
        self.strategy = strategy
        self.client = client
        self.model = model
        self.n_queries = n_queries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, query: str) -> List[str]:
        """
        Rewrite the query and return a list of strings to use for retrieval.

        For "multi_query" this can return multiple queries.
        All other strategies return a single-element list.
        """
        if self.strategy == "none":
            return [query]
        elif self.strategy == "hyd":
            return [self._hyd(query)]
        elif self.strategy == "step_back":
            return [self._step_back(query)]
        elif self.strategy == "multi_query":
            return self._multi_query(query)
        return [query]

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _hyd(self, query: str) -> str:
        prompt = (
            "You are a Wix product expert. "
            "Write a short, factual answer (2-4 sentences) to the following question "
            "as if it appeared in the Wix Help Center. "
            "Do NOT add greetings or meta-commentary.\n\n"
            f"Question: {query}\n\nHypothetical Answer:"
        )
        return _call_llm(prompt, self.client, self.model, max_tokens=128)

    def _step_back(self, query: str) -> str:
        prompt = (
            "Rephrase the following specific question into a more general question "
            "that covers the broader concept. Output only the rephrased question.\n\n"
            f"Original question: {query}\n\nGeneralized question:"
        )
        return _call_llm(prompt, self.client, self.model, max_tokens=64)

    def _multi_query(self, query: str) -> List[str]:
        prompt = (
            f"Generate {self.n_queries} diverse paraphrases of the following question. "
            "Output one paraphrase per line, numbered 1., 2., 3., … "
            "Do not include any other text.\n\n"
            f"Question: {query}"
        )
        raw = _call_llm(prompt, self.client, self.model, max_tokens=256)
        lines = [
            line.lstrip("0123456789. ").strip()
            for line in raw.splitlines()
            if line.strip()
        ]
        queries = [query] + lines[:self.n_queries]
        return queries
