#utils.py

import os
import json
import matplotlib.pyplot as plt

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def plot_f1_per_label(f1_dict, out_pdf, title="F1 per label"):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    labels = list(f1_dict.keys())
    scores = [f1_dict[k] for k in labels]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("F1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)   # save as PDF
    plt.close()
