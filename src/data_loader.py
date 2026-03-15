from __future__ import annotations
from datasets import load_dataset
from typing import Dict, List, Tuple
import pandas as pd

_DATASET_REPO = "Wix/WixQA"
_KB_CONFIG = "wix_kb_corpus"
_EXPERT_CONFIG = "wixqa_expertwritten"
_SIMULATED_CONFIG = "wixqa_simulated"
_SYNTHETIC_CONFIG = "wixqa_synthetic"

def load_kb_corpus() -> List[Dict]:
    """
    id (str) - article identifier
    url (str) - public URL of the article
    contents (str) - full article text
    article_type(str) - "article" | "feature_request" | "known_issue"
    """
    ds = load_dataset(_DATASET_REPO, _KB_CONFIG, split="train")
    articles = list(ds)
    print(f"{len(articles)} articles loaded.")
    return articles


def load_qa_split(split: str) -> List[Dict]:
    """
    Load "expert", "simulated", "synthetic" 
    splits from huggingface

    https://huggingface.co/datasets/Wix/WixQA
    """

    config_map = {
        "expert": _EXPERT_CONFIG,
        "simulated": _SIMULATED_CONFIG,
        "synthetic": _SYNTHETIC_CONFIG,
    }
    
    config = config_map[split]

    print(f"Loading WixQA-{split} from HuggingFace…")

    ds = load_dataset(_DATASET_REPO, config, split="train")
    rows = list(ds)
    print(f"{len(rows)} QA pairs loaded.")
    return rows


# lookup table {a["id"]: a for a in articles}
# convert qa rows to df pd.DataFrame(qa_rows)
