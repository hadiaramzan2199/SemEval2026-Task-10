import os
import argparse
import torch
from sklearn.metrics import classification_report

from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
from model_roberta_crf import RoBERTaCRFTagger


LABEL_LIST = [
    "O",
    "B-ACTION", "I-ACTION",
    "B-ACTOR", "I-ACTOR",
    "B-EFFECT", "I-EFFECT",
    "B-EVIDENCE", "I-EVIDENCE",
    "B-VICTIM", "I-VICTIM"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="results/roberta_crf_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, max_length = RoBERTaCRFTagger.load(args.model_path, device=device)
    model.to(device)
    print("Loaded model:", args.model_path, "| max_length:", max_length)

    train_df, _ = load_psycomark(args.train_path, dev_path=None)
    corpus = build_bio_corpus(train_df)
    _, val_corpus = train_val_split(corpus, test_size=0.2, random_state=42)

    preds = model.predict_labels(val_corpus, device=device, max_length=max_length)

    true_flat, pred_flat = [], []
    for sample, p in zip(val_corpus, preds):
        t = sample["labels"]
        m = min(len(t), len(p))
        true_flat.extend(t[:m])
        pred_flat.extend(p[:m])

    print("\n===== RoBERTa + CRF Evaluation =====")
    print(classification_report(true_flat, pred_flat, zero_division=0))


if __name__ == "__main__":
    main()
