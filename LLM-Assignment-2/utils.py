# utils.py

import os
import json
from typing import Dict
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_f1_per_label(f1_dict: Dict[str, float], out_path: str, title: str = "F1 per label"):
    """
    Plot a bar chart of F1 scores per label.
    """
    ensure_dir(os.path.dirname(out_path))

    labels = list(f1_dict.keys())
    scores = [f1_dict[l] for l in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1 score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
