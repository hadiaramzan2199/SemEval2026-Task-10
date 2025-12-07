"""Example Usage

From terminal:

python infer.py --model_path results/baseline_crf.pkl --text "Bill Gates controls vaccines to reduce population"
"""

import argparse
from data_loader import load_psycomark
from preprocess import tokenize_with_offsets
from models.model_crf import CRFBaseline


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="results/baseline_crf.pkl")
    p.add_argument("--text", type=str, required=True,
                   help="Input sentence to predict tags for")
    return p.parse_args()


def main():
    args = get_args()

    print("Loading CRF model:", args.model_path)
    model = CRFBaseline.load(args.model_path)

    text = args.text
    print("Input:", text)

    # tokenize
    toks_offsets = tokenize_with_offsets(text)
    tokens = [t[0] for t in toks_offsets]

    # Wrap into training-style structure
    sample = [{
        "tokens": tokens,
        "labels": None,
        "text": text
    }]

    # Predict
    preds = model.predict(sample)[0]

    print("\n===== Prediction =====")
    for tok, tag in zip(tokens, preds):
        print(f"{tok:<15} {tag}")

if __name__ == "__main__":
    main()
