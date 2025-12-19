#evaluate_roberta_crf

import os
import argparse
import torch
from sklearn.metrics import classification_report

from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
from model_roberta_crf import RoBERTaCRF


LABELS = [
    "O",
    "B-ACTOR", "I-ACTOR",
    "B-ACTION", "I-ACTION",
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

    model, max_len = RoBERTaCRF.load(args.model_path, device=device)
    model.to(device)
    model.eval()
    tokenizer = model.tokenizer
    print("Loaded:", args.model_path, "| max_length:", max_len)

    df, _ = load_psycomark(args.train_path)
    corpus = build_bio_corpus(df)
    _, val_c = train_val_split(corpus, test_size=0.2, random_state=42)

    true_flat, pred_flat = [], []

    with torch.no_grad():
        for s in val_c:
            enc = tokenizer(
                [s["tokens"]],
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            # build valid_mask + gold labels aligned to first-subword
            word_ids = enc.word_ids(batch_index=0)
            valid_mask = torch.zeros_like(attn, dtype=torch.bool)
            gold_ids = torch.full_like(input_ids, -100)

            prev = None
            for j, w in enumerate(word_ids):
                if w is None:
                    continue
                if w == prev:
                    prev = w
                    continue
                valid_mask[0, j] = True
                lab = s["labels"][w] if w < len(s["labels"]) else "O"
                gold_ids[0, j] = LABELS.index(lab) if lab in LABELS else 0
                prev = w

            preds = model(input_ids, attn, labels=None, valid_mask=valid_mask)["predictions"][0]
            pred_seq = [LABELS[k] for k in preds]

            gold_seq = []
            for j in range(valid_mask.size(1)):
                if valid_mask[0, j]:
                    gid = gold_ids[0, j].item()
                    gold_seq.append(LABELS[gid] if gid >= 0 else "O")

            m = min(len(gold_seq), len(pred_seq))
            true_flat.extend(gold_seq[:m])
            pred_flat.extend(pred_seq[:m])

    print("\n===== RoBERTa + CRF Evaluation (token-level BIO) =====")
    print(classification_report(true_flat, pred_flat, zero_division=0))


if __name__ == "__main__":
    main()
