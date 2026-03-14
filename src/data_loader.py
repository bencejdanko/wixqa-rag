"""
data_loader.py
==============
Loads WixQA datasets and the Wix KB corpus directly from HuggingFace.
No local files required — designed for Google Colab.
"""

from __future__ import annotations
from datasets import load_dataset
from typing import Dict, List, Tuple
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset split names (HuggingFace config names)
# ---------------------------------------------------------------------------
_DATASET_REPO = "Wix/WixQA"
_KB_CONFIG = "wix_kb_corpus"
_EXPERT_CONFIG = "wixqa_expertwritten"
_SIMULATED_CONFIG = "wixqa_simulated"
_SYNTHETIC_CONFIG = "wixqa_synthetic"


def load_kb_corpus() -> List[Dict]:
    """
    Load the full Wix knowledge-base corpus (6,221 articles).

    Returns
    -------
    List[Dict] with keys:
        id          (str)  – article identifier
        url         (str)  – public URL of the article
        contents    (str)  – full article text
        article_type(str)  – "article" | "feature_request" | "known_issue"
    """
    print("Loading Wix KB corpus from HuggingFace…")
    ds = load_dataset(_DATASET_REPO, _KB_CONFIG, split="train")
    articles = list(ds)
    print(f"  → {len(articles):,} articles loaded.")
    return articles


def load_qa_split(split: str) -> List[Dict]:
    """
    Load a WixQA QA split.

    Parameters
    ----------
    split : str
        One of "expert", "simulated", "synthetic".

    Returns
    -------
    List[Dict] with keys:
        question    (str)
        answer      (str)
        article_ids (List[str])
    """
    config_map = {
        "expert": _EXPERT_CONFIG,
        "simulated": _SIMULATED_CONFIG,
        "synthetic": _SYNTHETIC_CONFIG,
    }
    if split not in config_map:
        raise ValueError(f"split must be one of {list(config_map.keys())}")
    config = config_map[split]
    print(f"Loading WixQA-{split} from HuggingFace…")
    ds = load_dataset(_DATASET_REPO, config, split="train")
    rows = list(ds)
    print(f"  → {len(rows):,} QA pairs loaded.")
    return rows


def build_kb_lookup(articles: List[Dict]) -> Dict[str, Dict]:
    """Return a dict mapping article id → article dict for O(1) lookup."""
    return {a["id"]: a for a in articles}


def qa_to_dataframe(qa_rows: List[Dict]) -> pd.DataFrame:
    """Convert QA split to a pandas DataFrame for easy inspection."""
    return pd.DataFrame(qa_rows)
