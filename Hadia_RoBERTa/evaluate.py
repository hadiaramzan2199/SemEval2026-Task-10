import os
import torch 
import argparse
from data_loader import load_psycomark
from preprocess import build_bio_corpus, train_val_split
from model_roberta_ner import RoBERTaSequenceTagger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("results", "roberta_ner_model.pth"),
        help="Path to saved RoBERTa model",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="../data/train_rehydrated.jsonl",
        help="Path to training data",
    )
    args = parser.parse_args()

    print("Loading saved RoBERTa model from:", args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoBERTaSequenceTagger.load(args.model_path, device=device)

    print("Reloading dataset and building BIO corpus...")
    train_df, _ = load_psycomark(args.train_path, dev_path=None)
    corpus = build_bio_corpus(train_df)

    _, val_corpus = train_val_split(corpus, test_size=0.2, random_state=42)

    print("Evaluating on validation set...")
    metrics = model.evaluate(val_corpus)

    print("\n===== Evaluation Report (RoBERTa Baseline) =====")
    print("Weighted F1:", metrics["f1_weighted"])
    print("Macro F1:", metrics["f1_macro"])
    print("\nPer-label F1:")
    for label, f1 in metrics['per_label_f1'].items():
        print(f"  {label}: {f1:.4f}")

if __name__ == "__main__":
    main()
