"""
pipeline.py
===========
Top-level RAG pipeline orchestrator.

Ties together: chunker → embedder → vector store → retriever
               → (optional reranker) → (optional query rewriter) → generator.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional

from src.chunker import Chunk, chunk_articles, describe_chunks
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from src.reranker import CrossEncoderReranker
from src.query_rewriter import QueryRewriter
from src.generator import Generator


class RAGPipeline:
    """
    Production-style RAG pipeline with configurable components.

    Parameters
    ----------
    articles          : List[Dict] – KB corpus articles
    client            : OpenAI-compatible API client
    # --- Chunking ---
    chunk_strategy    : str   – "recursive" | "fixed" | "sentence"
    chunk_size        : int   – chars per chunk
    chunk_overlap     : int   – overlap chars
    # --- Embedding ---
    embedding_model   : str   – HuggingFace sentence-transformer name
    # --- Retrieval ---
    retriever_type    : str   – "dense" | "bm25" | "hybrid"
    top_k_retrieval   : int   – first-stage retrieval size
    # --- Reranking ---
    use_reranker      : bool  – cross-encoder reranking toggle
    reranker_model    : str   – HuggingFace cross-encoder name
    top_k_rerank      : int   – passages kept after reranking
    # --- Query rewriting ---
    query_strategy    : str   – "none" | "hyd" | "step_back" | "multi_query"
    # --- Generation ---
    generator_model   : str   – LLM model identifier (OpenRouter)
    gen_strategy      : str   – "basic" | "cot" | "cite" | "compression"
    max_context_chars : int   – max chars of context fed to the LLM
    max_tokens        : int   – max tokens for generated answer
    """

    def __init__(
        self,
        articles: List[Dict],
        client,
        # Chunking
        chunk_strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        # Embedding
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        # Retrieval
        retriever_type: str = "dense",
        top_k_retrieval: int = 10,
        # Reranking
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_rerank: int = 5,
        # Query rewriting
        query_strategy: str = "none",
        # Generation
        generator_model: str = "google/gemini-2.0-flash-001",
        gen_strategy: str = "basic",
        max_context_chars: int = 6000,
        max_tokens: int = 512,
    ):
        self.config = dict(
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            retriever_type=retriever_type,
            top_k_retrieval=top_k_retrieval,
            use_reranker=use_reranker,
            reranker_model=reranker_model if use_reranker else "—",
            top_k_rerank=top_k_rerank,
            query_strategy=query_strategy,
            generator_model=generator_model,
            gen_strategy=gen_strategy,
            max_context_chars=max_context_chars,
        )

        # -------- 1. Chunking --------
        print(f"\n[1/5] Chunking articles (strategy={chunk_strategy}, "
              f"size={chunk_size}, overlap={chunk_overlap})…")
        self.chunks = chunk_articles(
            articles,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        stats = describe_chunks(self.chunks)
        print(f"  → {stats['total_chunks']:,} chunks | "
              f"avg {stats['avg_chars']} chars/chunk")

        # -------- 2. Embedding --------
        print(f"\n[2/5] Embedding chunks with '{embedding_model}'…")
        self.embedder = Embedder(model_name=embedding_model)
        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.encode(texts)

        # -------- 3. Vector store --------
        print(f"\n[3/5] Building FAISS index…")
        self.vector_store = VectorStore(dim=self.embedder.dim, index_type="flat")
        self.vector_store.add(self.chunks, embeddings)

        # -------- 4. Retriever --------
        print(f"\n[4/5] Initialising retriever (type={retriever_type})…")
        dense_ret = DenseRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=top_k_retrieval,
        )
        if retriever_type == "dense":
            self.retriever = dense_ret
        elif retriever_type == "bm25":
            self.retriever = BM25Retriever(self.chunks)
        elif retriever_type == "hybrid":
            sparse_ret = BM25Retriever(self.chunks)
            self.retriever = HybridRetriever(
                dense=dense_ret,
                sparse=sparse_ret,
                top_k=top_k_retrieval,
            )
        else:
            raise ValueError(f"Unknown retriever_type: {retriever_type}")

        # -------- 4b. Optional reranker --------
        self.reranker: Optional[CrossEncoderReranker] = None
        self.top_k_rerank = top_k_rerank
        if use_reranker:
            print(f"\n     Loading cross-encoder reranker '{reranker_model}'…")
            self.reranker = CrossEncoderReranker(
                model_name=reranker_model, top_n=top_k_rerank
            )

        # -------- 4c. Query rewriter --------
        self.query_rewriter = QueryRewriter(
            strategy=query_strategy,
            client=client,
            model=generator_model,
        )

        # -------- 5. Generator --------
        print(f"\n[5/5] Generator ready (model={generator_model}, strategy={gen_strategy})")
        self.generator = Generator(
            client=client,
            model=generator_model,
            strategy=gen_strategy,
            max_context_chars=max_context_chars,
            max_tokens=max_tokens,
        )

        print("\n✅ RAG pipeline initialised.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        End-to-end inference for a single query.

        Returns dict with:
            "answer"    : str
            "retrieved" : List[Tuple[Chunk, float]]
            "queries_used": List[str]
        """
        k = top_k or self.config["top_k_retrieval"]

        # Query rewriting (may produce multiple queries)
        queries = self.query_rewriter.rewrite(query)

        # Retrieve for each rewritten query and merge by chunk_id
        seen: dict[str, Tuple[Chunk, float]] = {}
        for q in queries:
            for chunk, score in self.retriever.retrieve(q, top_k=k):
                if chunk.chunk_id not in seen or score > seen[chunk.chunk_id][1]:
                    seen[chunk.chunk_id] = (chunk, score)

        retrieved = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:k]

        # Optional reranking
        if self.reranker:
            retrieved = self.reranker.rerank(query, retrieved)

        # Generation
        gen_result = self.generator.generate(query, retrieved)

        return {
            "answer": gen_result["answer"],
            "retrieved": retrieved,
            "queries_used": queries,
        }

    # ------------------------------------------------------------------
    # Config summary
    # ------------------------------------------------------------------

    def print_config(self) -> None:
        """Print the baseline configuration table."""
        rows = [
            ("Chunk size / overlap",  f"{self.config['chunk_size']} / {self.config['chunk_overlap']} chars"),
            ("Chunk strategy",        self.config["chunk_strategy"]),
            ("Embedding model",       self.config["embedding_model"]),
            ("Vector database",       "FAISS (IndexFlatIP)"),
            ("Retriever type",        self.config["retriever_type"]),
            ("Reranker",              self.config["reranker_model"]),
            ("Query rewriting",       self.config["query_strategy"]),
            ("Generator model",       self.config["generator_model"]),
            ("Prompting strategy",    self.config["gen_strategy"]),
            ("Max context chars",     str(self.config["max_context_chars"])),
            ("Total chunks indexed",  f"{self.vector_store.size:,}"),
        ]
        col1 = max(len(r[0]) for r in rows) + 2
        print("\n" + "=" * (col1 + 32))
        print(f"{'Component':<{col1}}{'Configuration Used'}")
        print("=" * (col1 + 32))
        for label, value in rows:
            print(f"{label:<{col1}}{value}")
        print("=" * (col1 + 32))
