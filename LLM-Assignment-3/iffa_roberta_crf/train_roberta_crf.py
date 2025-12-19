#train_roberta_crf.py

import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
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
label2id = {l: i for i, l in enumerate(LABELS)}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class BioDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]


def collate_fn(batch, tokenizer, max_length):
    """
    Batch: list of {"tokens":[...], "labels":[...]}
    We create:
      - input_ids, attention_mask
      - labels aligned to FIRST subword only; other subwords = -100
      - valid_mask True only for FIRST subword positions
    """
    words_list = [s["tokens"] for s in batch]
    labels_list = [s["labels"] for s in batch]

    enc = tokenizer(
        words_list,
        is_split_into_words=True,
        padding=True,              # dynamic padding per batch (saves GPU)
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    B, L = enc["input_ids"].shape
    all_labels = torch.full((B, L), -100, dtype=torch.long)
    valid_mask = torch.zeros((B, L), dtype=torch.bool)

    for i in range(B):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        for j, w in enumerate(word_ids):
            if w is None:
                continue
            if w == prev:
                # subword continuation
                prev = w
                continue
            # first subword of a word
            valid_mask[i, j] = True
            gold = labels_list[i]
            lab = gold[w] if w < len(gold) else "O"
            all_labels[i, j] = label2id.get(lab, label2id["O"])
            prev = w

    return enc["input_ids"], enc["attention_mask"], all_labels, valid_mask


def compute_metrics(true_labels_flat, pred_labels_flat):
    report = classification_report(true_labels_flat, pred_labels_flat, output_dict=True, zero_division=0)
    return {
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "report": report
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)      # keep small to avoid OOM
    parser.add_argument("--grad_accum", type=int, default=4)      # effective batch = 16
    parser.add_argument("--max_length", type=int, default=128)    # reduce from 256 to avoid OOM
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze_encoder", action="store_true")  # optional
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading:", args.train_path)
    df, _ = load_psycomark(args.train_path)

    corpus = build_bio_corpus(df)
    train_c, val_c = train_val_split(corpus, test_size=0.2, random_state=42)

    model = RoBERTaCRF(args.model_name, LABELS, freeze_encoder=args.freeze_encoder).to(device)
    tokenizer = model.tokenizer

    train_loader = DataLoader(
        BioDataset(train_c),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )
    val_loader = DataLoader(
        BioDataset(val_c),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        step = 0

        for batch in train_loader:
            input_ids, attn, labels, vmask = [x.to(device) for x in batch]

            with autocast():
                out = model(input_ids, attn, labels=labels, valid_mask=vmask)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            total_loss += float(loss.item())
            step += 1

            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print(f"\nEpoch {epoch} | train loss: {total_loss:.4f}")

        # -------- Validation --------
        model.eval()
        true_flat, pred_flat = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attn, labels, vmask = [x.to(device) for x in batch]
                preds = model(input_ids, attn, labels=None, valid_mask=vmask)["predictions"]

                # preds = list of lists (per sample, only valid positions)
                for i in range(len(preds)):
                    # collect true labels only where valid_mask True
                    vm = vmask[i].tolist()
                    gold_ids = labels[i].tolist()

                    gold_seq = []
                    for j in range(len(vm)):
                        if vm[j]:
                            gid = gold_ids[j]
                            gold_seq.append(LABELS[gid] if gid >= 0 else "O")

                    pred_seq = [LABELS[k] for k in preds[i]]

                    m = min(len(gold_seq), len(pred_seq))
                    true_flat.extend(gold_seq[:m])
                    pred_flat.extend(pred_seq[:m])

        metrics = compute_metrics(true_flat, pred_flat)
        print(f"Val weighted F1: {metrics['f1_weighted']:.4f} | macro F1: {metrics['f1_macro']:.4f}")

        # save best
        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            best_path = os.path.join(args.results_dir, "roberta_crf_best.pth")
            model.save(best_path, max_length=args.max_length)

            save_json(
                {
                    "best_val_f1_weighted": metrics["f1_weighted"],
                    "best_val_f1_macro": metrics["f1_macro"],
                    "hyperparams": vars(args),
                    "note": "BIO tagging on FIRST subword positions only"
                },
                os.path.join(args.results_dir, "roberta_crf_best_metrics.json")
            )
            print("✅ Saved best model:", best_path)

    print("\nDone. Best weighted F1:", best_f1)


if __name__ == "__main__":
    main()
