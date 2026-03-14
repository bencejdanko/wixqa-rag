"""
evaluator.py
============
Evaluation metrics for the WixQA RAG pipeline.

Metrics
-------
Retrieval:
  - Recall@k      : fraction of gold article IDs retrieved in top-k chunks
  - Precision@k   : fraction of retrieved articles that are relevant
  - MRR           : Mean Reciprocal Rank of the first relevant chunk

Generation:
  - ROUGE-1/2/L   : n-gram overlap between generated and reference answer
  - Exact Match   : binary string match (normalised)
  - LLM-as-Judge  : GPT/Gemini rates factuality and completeness (0–5 scale)
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import re
import numpy as np


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_article_ids: List[str],
    gold_article_ids: List[str],
    k: int,
) -> float:
    """Fraction of gold articles found in the top-k retrieved chunk article IDs."""
    if not gold_article_ids:
        return 0.0
    retrieved_k = set(retrieved_article_ids[:k])
    gold = set(gold_article_ids)
    return len(retrieved_k & gold) / len(gold)


def precision_at_k(
    retrieved_article_ids: List[str],
    gold_article_ids: List[str],
    k: int,
) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved_article_ids[:k]
    gold = set(gold_article_ids)
    hits = sum(1 for a in retrieved_k if a in gold)
    return hits / k


def mrr(
    retrieved_article_ids: List[str],
    gold_article_ids: List[str],
) -> float:
    """Mean Reciprocal Rank (single query, multiple gold articles)."""
    gold = set(gold_article_ids)
    for rank, aid in enumerate(retrieved_article_ids, start=1):
        if aid in gold:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retrieved_chunks,          # List[Tuple[Chunk, float]]
    gold_article_ids: List[str],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute recall@k, precision@k, and MRR for one query."""
    retrieved_aids = [chunk.article_id for chunk, _ in retrieved_chunks]
    results = {}
    for k in k_values:
        results[f"recall@{k}"]    = recall_at_k(retrieved_aids, gold_article_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_aids, gold_article_ids, k)
    results["mrr"] = mrr(retrieved_aids, gold_article_ids)
    return results


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(_normalize(prediction) == _normalize(reference))


def rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert evaluator for question-answering systems.
Score the provided answer on two dimensions, each from 0 to 5:
  Factuality   (0 = fully wrong or hallucinated, 5 = completely accurate)
  Completeness (0 = misses everything, 5 = addresses all aspects of the question)
Return ONLY a JSON object with keys "factuality" and "completeness".
Example: {"factuality": 4, "completeness": 3}
"""

_JUDGE_PROMPT = """\
Question: {question}

Reference Answer: {reference}

Generated Answer: {prediction}

Scores (JSON only):"""


def llm_judge(
    question: str,
    prediction: str,
    reference: str,
    client,
    model: str = "google/gemini-2.0-flash-001",
) -> Dict[str, float]:
    """Call an LLM to judge factuality and completeness (0–5 each)."""
    import json
    prompt = _JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON even if model wraps it in ```json ... ```
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            return {
                "llm_factuality": float(scores.get("factuality", 0)),
                "llm_completeness": float(scores.get("completeness", 0)),
            }
    except Exception as e:
        print(f"LLM judge error: {e}")
    return {"llm_factuality": np.nan, "llm_completeness": np.nan}


# ---------------------------------------------------------------------------
# Aggregate evaluation over a dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(
    qa_rows: List[Dict],
    retriever,
    generator,
    client=None,
    model: str = "google/gemini-2.0-flash-001",
    use_llm_judge: bool = False,
    top_k: int = 5,
    k_values: List[int] = [1, 3, 5, 10],
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Run end-to-end evaluation on a list of QA rows.

    Parameters
    ----------
    qa_rows       : list of dicts with "question", "answer", "article_ids"
    retriever     : DenseRetriever | HybridRetriever  (has .retrieve(query, top_k))
    generator     : Generator
    client        : LLM client (needed for LLM judge)
    model         : str – model for LLM judge
    use_llm_judge : bool – whether to call LLM judge (slow; rate-limited)
    top_k         : int – number of chunks to retrieve
    k_values      : list of k for recall/precision
    max_samples   : int | None – truncate for quick debugging
    verbose       : bool – print progress

    Returns
    -------
    (per_sample_results, aggregate_metrics)
    """
    from tqdm.auto import tqdm

    rows = qa_rows[:max_samples] if max_samples else qa_rows
    per_sample = []

    for row in tqdm(rows, desc="Evaluating", disable=not verbose):
        question     = row["question"]
        reference    = row["answer"]
        gold_art_ids = row.get("article_ids", [])

        # Retrieve
        retrieved = retriever.retrieve(question, top_k=top_k)

        # Retrieval metrics
        ret_metrics = evaluate_retrieval(retrieved, gold_art_ids, k_values)

        # Generate
        gen_result = generator.generate(question, retrieved)
        prediction = gen_result["answer"]

        # Generation metrics
        gen_metrics = rouge_scores(prediction, reference)
        gen_metrics["exact_match"] = exact_match(prediction, reference)

        # Optional LLM judge
        if use_llm_judge and client:
            judge = llm_judge(question, prediction, reference, client, model)
            gen_metrics.update(judge)

        sample_result = {
            "question": question,
            "reference": reference,
            "prediction": prediction,
            **ret_metrics,
            **gen_metrics,
        }
        per_sample.append(sample_result)

    # Aggregate
    agg: Dict[str, float] = {}
    numeric_keys = [k for k in per_sample[0] if isinstance(per_sample[0][k], (int, float))]
    for key in numeric_keys:
        vals = [r[key] for r in per_sample if not np.isnan(r[key])]
        agg[key] = float(np.mean(vals)) if vals else np.nan

    return per_sample, agg
