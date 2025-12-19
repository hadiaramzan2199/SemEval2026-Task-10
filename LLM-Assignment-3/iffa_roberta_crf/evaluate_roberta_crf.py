#evaluate_roberta_crf

import os
import argparse
import torch
from sklearn.metrics import classification_report

from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
from model_roberta_crf import RoBERTaCRFTagger


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

    model = RoBERTaCRFTagger.load(args.model_path, device=device).to(device)
    print("Loaded:", args.model_path)

    df, _ = load_psycomark(args.train_path)
    corpus = build_bio_corpus(df)
    _, val_c = train_val_split(corpus, test_size=0.2, random_state=42)

    true_flat, pred_flat = [], []

    model.eval()
    with torch.no_grad():
        for sample in val_c:
            input_ids, attn, labels, crf_mask, word_mask = model.encode_sentence(sample)
            input_ids = input_ids.unsqueeze(0).to(device)
            attn      = attn.unsqueeze(0).to(device)
            labels    = labels.unsqueeze(0).to(device)
            crf_mask  = crf_mask.unsqueeze(0).to(device)
            word_mask = word_mask.unsqueeze(0).to(device)

            out = model(input_ids, attn, labels=None, crf_mask=crf_mask)
            pred_ids = out["pred_paths"][0]  # includes CLS

            wm = word_mask[0].tolist()

            gold_seq = []
            for pos in range(1, len(wm)):
                if wm[pos] == 1:
                    gold_seq.append(LABELS[labels[0, pos].item()])

            pred_seq = [LABELS[t] for t in pred_ids[1:1+len(gold_seq)]]

            m = min(len(gold_seq), len(pred_seq))
            true_flat.extend(gold_seq[:m])
            pred_flat.extend(pred_seq[:m])

    print("\n===== RoBERTa + CRF Evaluation =====")
    print(classification_report(true_flat, pred_flat, zero_division=0))

if __name__ == "__main__":
    main()
