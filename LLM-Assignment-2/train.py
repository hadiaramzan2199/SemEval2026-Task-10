# train.py

import os
import argparse

from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
from utils import save_json, plot_f1_per_label
from models.model_crf import CRFBaseline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train_rehydrated.jsonl",
        help="Path to rehydrated PsyCoMark training file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save metrics and models",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )
    # later: parser.add_argument("--model", choices=["crf","bilstm_crf","bert","roberta"], default="crf")
    return parser.parse_args()


def main():
    args = get_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print("Loading dataset...")
    train_df, _ = load_psycomark(args.train_path, dev_path=None)

    print("Building BIO-tagged corpus...")
    corpus = build_bio_corpus(train_df)

    print("Splitting into train/validation...")
    train_corpus, val_corpus = train_val_split(corpus, test_size=0.2, random_state=42)
    print(f"Train sentences: {len(train_corpus)}, Val sentences: {len(val_corpus)}")

    print("Initializing CRF baseline model...")
    model = CRFBaseline()

    print("Training CRF...")
    model.fit(train_corpus)

    print("Evaluating on validation set...")
    metrics = model.evaluate(val_corpus)

    print("\n===== Evaluation Report (CRF Baseline) =====")
    print("Weighted F1:", metrics["f1_weighted"])
    print("Macro F1:", metrics["f1_macro"])
    print("\nFull classification report:\n")
    print(metrics["report_str"])

    # Save model
    model_path = os.path.join(args.results_dir, "baseline_crf.pkl")
    model.save(model_path)
    print(f"\nSaved CRF model to: {model_path}")

    # Save metrics JSON
    metrics_json_path = os.path.join(args.results_dir, "baseline_crf_metrics.json")
    save_json(
        {
            "f1_weighted": metrics["f1_weighted"],
            "f1_macro": metrics["f1_macro"],
            "per_label_f1": metrics["per_label_f1"],
        },
        metrics_json_path,
    )
    print(f"Saved metrics to: {metrics_json_path}")

    # Save F1-per-label bar plot
    plot_path = os.path.join(args.plots_dir, "baseline_crf_f1_per_label.png")
    plot_f1_per_label(
        metrics["per_label_f1"],
        plot_path,
        title="CRF Baseline — F1 per label"
    )
    print(f"Saved F1 per label plot to: {plot_path}")


if __name__ == "__main__":
    main()
