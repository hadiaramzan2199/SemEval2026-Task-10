# CRF Baseline — Subtask 1

## Model

Conditional Random Field (CRF) for BIO sequence tagging.

Features:
- token shape
- prefixes/suffixes
- digit/uppercase flags
- prev/next token features

## Run

Train:
bash

python train.py --train_path ../data/train_rehydrated.jsonl


Evaluate:
bash

python evaluate.py --model_path results/baseline_crf.pkl


## Outputs

- `results/*.pkl` — model
- `plots/*.png` — F1 per label
- `results/metrics.json` — numeric results
