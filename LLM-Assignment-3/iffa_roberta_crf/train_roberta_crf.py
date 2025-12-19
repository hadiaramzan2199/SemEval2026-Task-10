#train_roberta_crf.py

import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
from model_roberta_crf import RoBERTaCRFTagger
from utils import save_json, plot_f1_per_label


LABELS = [
    "O",
    "B-ACTOR", "I-ACTOR",
    "B-ACTION", "I-ACTION",
    "B-EFFECT", "I-EFFECT",
    "B-EVIDENCE", "I-EVIDENCE",
    "B-VICTIM", "I-VICTIM"
]

def compute_metrics(true_flat, pred_flat):
    report_dict = classification_report(true_flat, pred_flat, output_dict=True, zero_division=0)
    report_str = classification_report(true_flat, pred_flat, zero_division=0)

    per_label_f1 = {}
    for lbl in LABELS:
        if lbl in report_dict:
            per_label_f1[lbl] = float(report_dict[lbl]["f1-score"])

    return {
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),
        "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        "per_label_f1": per_label_f1,
        "report_str": report_str,
    }

class BioDataset(Dataset):
    def __init__(self, corpus, model: RoBERTaCRFTagger):
        self.corpus = corpus
        self.model = model

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sample = self.corpus[idx]
        return self.model.encode_sentence(sample)  # input_ids, attn, labels, crf_mask, word_mask

def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    attn      = torch.stack([b[1] for b in batch])
    labels    = torch.stack([b[2] for b in batch])
    crf_mask  = torch.stack([b[3] for b in batch])
    word_mask = torch.stack([b[4] for b in batch])
    return input_ids, attn, labels, crf_mask, word_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plots_dir", type=str, default="plots")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)     # small to avoid OOM
    parser.add_argument("--grad_accum", type=int, default=8)     # effective batch = 16
    parser.add_argument("--max_length", type=int, default=192)   # reduce memory
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading:", args.train_path)
    df, _ = load_psycomark(args.train_path)
    corpus = build_bio_corpus(df)
    train_c, val_c = train_val_split(corpus, test_size=0.2, random_state=42)
    print("Train:", len(train_c), " Val:", len(val_c))

    model = RoBERTaCRFTagger(args.model_name, LABELS, max_length=args.max_length).to(device)

    train_ds = BioDataset(train_c, model)
    val_ds   = BioDataset(val_c, model)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum + 1
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_f1 = -1.0
    history = {"epoch": [], "val_f1_weighted": [], "val_f1_macro": []}

    for ep in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids, attn, labels, crf_mask, word_mask = batch
            input_ids = input_ids.to(device)
            attn      = attn.to(device)
            labels    = labels.to(device)
            crf_mask  = crf_mask.to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(input_ids, attn, labels=labels, crf_mask=crf_mask)
                    loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(input_ids, attn, labels=labels, crf_mask=crf_mask)
                loss = out["loss"] / args.grad_accum
                loss.backward()

            running_loss += float(loss.item())

            if step % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(f"Epoch {ep} | train loss: {running_loss:.4f}")

        # -------- validation ----------
        model.eval()
        true_flat, pred_flat = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attn, labels, crf_mask, word_mask = batch
                input_ids = input_ids.to(device)
                attn      = attn.to(device)
                labels    = labels.to(device)
                crf_mask  = crf_mask.to(device)
                word_mask = word_mask.to(device)

                out = model(input_ids, attn, labels=None, crf_mask=crf_mask)
                paths = out["pred_paths"]  # per sequence, includes CLS position (since mask[0]=True)

                # rebuild word-level true/pred ignoring CLS and masked-off positions
                for i in range(labels.size(0)):
                    wm = word_mask[i].tolist()
                    # positions with word_mask==1 are valid words, but wm[0] is CLS -> skip it
                    gold_seq = []
                    for pos in range(1, len(wm)):
                        if wm[pos] == 1:
                            gold_seq.append(LABELS[labels[i, pos].item()])

                    pred_ids = paths[i]
                    # pred_ids corresponds to CRF-masked positions (CLS + word-first-subwords)
                    # first element corresponds to CLS -> skip it
                    pred_seq = [LABELS[t] for t in pred_ids[1:1+len(gold_seq)]]

                    m = min(len(gold_seq), len(pred_seq))
                    true_flat.extend(gold_seq[:m])
                    pred_flat.extend(pred_seq[:m])

        metrics = compute_metrics(true_flat, pred_flat)
        print(f"Val weighted F1: {metrics['f1_weighted']:.4f} | macro F1: {metrics['f1_macro']:.4f}")

        history["epoch"].append(ep)
        history["val_f1_weighted"].append(metrics["f1_weighted"])
        history["val_f1_macro"].append(metrics["f1_macro"])

        # save best
        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]

            model_path = os.path.join(args.results_dir, "roberta_crf_best.pth")
            model.save(model_path)

            save_json(
                {
                    "best_val_f1_weighted": metrics["f1_weighted"],
                    "best_val_f1_macro": metrics["f1_macro"],
                    "per_label_f1": metrics["per_label_f1"],
                    "report": metrics["report_str"],
                    "hyperparams": vars(args),
                },
                os.path.join(args.results_dir, "roberta_crf_best_metrics.json"),
            )

            plot_f1_per_label(
                metrics["per_label_f1"],
                os.path.join(args.plots_dir, "roberta_crf_f1_per_label.pdf"),
                title="RoBERTa + CRF — F1 per label",
            )

            print("✅ Saved new best model + metrics + PDF plot.")

    save_json(history, os.path.join(args.results_dir, "roberta_crf_history.json"))
    print("Done.")

if __name__ == "__main__":
    main()
