import os
import sys
import torch
import json
import matplotlib.pyplot as plt

# Import from parent directory
sys.path.append('..')
from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split

from model_roberta_ner import RoBERTaSequenceTagger

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def plot_f1_per_label(f1_dict, out_path, title="F1 per label"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    labels = list(f1_dict.keys())
    scores = [f1_dict[l] for l in labels]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1 score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../data/train_rehydrated.jsonl')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--plots_dir', type=str, default='plots')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print("=" * 60)
    print("ROBERTA BASELINE - Sequence Labeling (Subtask 1)")
    print("=" * 60)
    
    print("\n[1/5] Loading dataset...")
    train_df, _ = load_psycomark(args.train_path, dev_path=None)
    
    print("\n[2/5] Building BIO corpus...")
    corpus = build_bio_corpus(train_df)
    print(f"Total sentences: {len(corpus)}")
    
    print("\n[3/5] Splitting data...")
    train_corpus, val_corpus = train_val_split(corpus, test_size=0.2, random_state=42)
    print(f"Train: {len(train_corpus)} sentences")
    print(f"Validation: {len(val_corpus)} sentences")
    
    print("\n[4/5] Initializing RoBERTa model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RoBERTaSequenceTagger(
        model_name="xlm-roberta-base",
        device=device
    )
    
    model.num_epochs = args.epochs
    model.batch_size = args.batch_size
    
    print(f"\nLabel mapping ({len(model.label_list)} labels):")
    for label, idx in model.label2id.items():
        print(f"  {label}: {idx}")
    
    print("\n[5/5] Training model...")
    model.train(train_corpus, val_corpus)
    
    print("\nEvaluating on validation set...")
    metrics = model.evaluate(val_corpus)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    print("\nPer-label F1 scores:")
    for label, f1 in metrics['per_label_f1'].items():
        print(f"  {label}: {f1:.4f}")
    
    model_path = os.path.join(args.results_dir, "roberta_ner_model.pth")
    model.save(model_path)
    
    metrics_path = os.path.join(args.results_dir, "roberta_ner_metrics.json")
    save_json({
        "f1_weighted": float(metrics["f1_weighted"]),
        "f1_macro": float(metrics["f1_macro"]),
        "per_label_f1": {k: float(v) for k, v in metrics["per_label_f1"].items()}
    }, metrics_path)
    
    plot_path = os.path.join(args.plots_dir, "roberta_ner_f1_per_label.png")
    plot_f1_per_label(
        metrics["per_label_f1"],
        plot_path,
        title="RoBERTa NER Baseline — F1 per label"
    )
    
    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plot saved to: {plot_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
