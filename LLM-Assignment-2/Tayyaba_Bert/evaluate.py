import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from preprocess import get_spans

def overlap_f1(pred_spans, true_spans):
    TP = FP = FN = 0
    for p, t in zip(pred_spans, true_spans):
        matched = set()
        for ps in p:
            found = False
            for i, ts in enumerate(t):
                if ps[2] == ts[2]:  # label
                    if not (ps[1] <= ts[0] or ps[0] >= ts[1]):  # overlap
                        TP += 1
                        matched.add(i)
                        found = True
                        break
            if not found:
                FP += 1
        FN += len(t) - len(matched)
    
    prec = TP / (TP + FP + 1e-8)
    rec = TP / (TP + FN + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

def overlap_f1_per_class(pred_spans, true_spans):
    marker_types = ['Actor', 'Action', 'Effect', 'Victim', 'Evidence']
    results = {}

    for m_type in marker_types:
        TP = FP = FN = 0
        for p, t in zip(pred_spans, true_spans):
            matched = set()
            for ps in p:
                if ps[2] == m_type:
                    found = False
                    for i, ts in enumerate(t):
                        if ts[2] == m_type:
                            if not (ps[1] <= ts[0] or ps[0] >= ts[1]):
                                TP += 1
                                matched.add(i)
                                found = True
                                break
                    if not found:
                        FP += 1
        for i, ts in enumerate(t):
            if ts[2] == m_type and i not in matched:
                FN += 1
        
        prec = TP / (TP + FP + 1e-8)
        rec = TP / (TP + FN + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0
        results[m_type] = {'precision': prec, 'recall': rec, 'f1': f1}
    return results

def plot_training_loss(loss_history, save_path="Plots/training_loss.png"):
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    print(f"Training loss plot saved to {save_path}")

def plot_per_class_f1(results, save_path="Plots/per_class_f1.png"):
    marker_types = list(results.keys())
    f1_scores = [results[m]['f1'] for m in marker_types]

    plt.figure(figsize=(7,5))
    plt.bar(marker_types, f1_scores)
    plt.title("Per-Class Overlap F1 Scores")
    plt.xlabel("Marker Type")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    print(f"Per-class F1 plot saved to {save_path}")
