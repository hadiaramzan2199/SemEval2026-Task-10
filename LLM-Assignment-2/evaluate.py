# evaluate.py

import os
import argparse

from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
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
        "--model_path",
        type=str,
        default=os.path.join("results", "baseline_crf.pkl"),
        help="Path to saved CRF model",
    )
    return parser.parse_args()


def main():
    args = get_args()

    print("Loading saved CRF model from:", args.model_path)
    model = CRFBaseline.load(args.model_path)

    print("Reloading dataset and building BIO corpus...")
    train_df, _ = load_psycomark(args.train_path, dev_path=None)
    corpus = build_bio_corpus(train_df)

    # Use same split as in train.py (same random_state)
    _, val_corpus = train_val_split(corpus, test_size=0.2, random_state=42)

    print("Evaluating on validation set...")
    metrics = model.evaluate(val_corpus)

    print("\n===== Evaluation Report (CRF Baseline) =====")
    print("Weighted F1:", metrics["f1_weighted"])
    print("Macro F1:", metrics["f1_macro"])
    print("\nFull classification report:\n")
    print(metrics["report_str"])


if __name__ == "__main__":
    main()
