```markdown
# RoBERTa + CRF for Psycholinguistic Marker Extraction

**SemEval 2026 Task 10 — Subtask 1**  
**Assignment 3: Proposed Solution**

---

## Overview

This module implements a **RoBERTa + Conditional Random Field (CRF)** sequence tagging model
for **span-level psycholinguistic marker extraction** from Reddit conspiracy discourse.

The model improves over transformer-only baselines by enforcing **BIO transition consistency**
and reducing fragmented span predictions.

---

## Markers Extracted

- Actor
- Action
- Effect
- Evidence
- Victim

Using BIO tagging:
```

B-ACTOR, I-ACTOR, ...

```

---

## Architecture

```

Tokens
↓
RoBERTa Encoder
↓
Linear Emission Layer
↓
CRF (BIO constraints)
↓
Span-level Predictions

```

### Why CRF?
- Enforces valid BIO transitions
- Improves boundary coherence
- Reduces overlapping / fragmented spans

---

## File Structure

```

iffa_roberta_crf/
│
├── data/
│   └── train_rehydrated.jsonl
│
├── plots/
│   ├── roberta_crf_f1_per_label.pdf
│   └── roberta_crf_learning_curve.pdf
│
├── results/
│   ├── roberta_crf_best.pth
│   ├── roberta_crf_best_metrics.json
│   └── roberta_crf_history.json
│
├── data_loader.py
├── preprocess.py
├── model_roberta_crf.py
├── train_roberta_crf.py
├── evaluate_roberta_crf.py
├── utils.py
└── README.md

````

---

## Training

```bash
python train_roberta_crf.py \
  --train_path data/train_rehydrated.jsonl \
  --epochs 10 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_length 192 \
  --results_dir results \
  --plots_dir plots
````

✔ Mixed precision
✔ Gradient accumulation
✔ GPU-safe memory usage

---

## Evaluation

```bash
python evaluate_roberta_crf.py \
  --train_path data/train_rehydrated.jsonl \
  --model_path results/roberta_crf_best.pth
```

---

## Results (Validation)

| Metric      | Score                |
| ----------- | -------------------- |
| Weighted F1 | ~0.64                |
| Macro F1    | ~0.15                |

Per-label F1 and learning curves are saved as **PDF** for report inclusion.

---

## Assignment Contribution

**Member: Iffa Kashif**
**Responsibility:** Core Architecture & CRF Integration

* Implemented RoBERTa + CRF model
* Designed BIO-consistent decoding
* Addressed class imbalance via CRF constraints
* Generated learning curves and evaluation plots

---

## References

* Lafferty et al., *Conditional Random Fields*, ICML 2001
* Liu et al., *RoBERTa*, arXiv 2019
* SemEval 2026 Task 10
* HuggingFace Transformers
* pytorch-crf

```

---

