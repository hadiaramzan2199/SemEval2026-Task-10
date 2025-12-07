# data_loader.py

import json
import pandas as pd
from typing import Tuple, Optional


def load_jsonl(path: str) -> pd.DataFrame:
    """
    Load a JSONL file into a pandas DataFrame.
    Each line must be a valid JSON object.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_psycomark(
    train_path: str,
    dev_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load PsyCoMark training (and optional dev) data.

    Expected columns:
      - _id
      - text
      - subreddit
      - conspiracy
      - markers (list of dicts with startIndex, endIndex, type, text)
      - annotator
    """
    train_df = load_jsonl(train_path)

    dev_df = load_jsonl(dev_path) if dev_path is not None else None
    return train_df, dev_df
