# RoBERTa + Domain-Adaptive Pretraining (DAPT) + CRF  
**Psycholinguistic Marker Extraction**

**SemEval 2026 Task 10 — Subtask 1**  
**Assignment 3: Proposed Solution**

---

## Overview

This module implements a **RoBERTa + CRF sequence labeling model enhanced with
Domain-Adaptive Pretraining (DAPT)** for extracting **psycholinguistic markers**
from Reddit-based conspiracy discourse.

The objective of DAPT is to adapt the language model to **domain-specific linguistic
patterns** (informal syntax, conspiratorial framing) prior to structured sequence
tagging with a CRF layer.

---

## Psycholinguistic Markers

The model extracts the following marker categories using **BIO tagging**:

- Actor  
- Action  
- Effect  
- Evidence  
- Victim  

BIO tag set:

```
B-ACTOR, I-ACTOR  
B-ACTION, I-ACTION  
B-EFFECT, I-EFFECT  
B-EVIDENCE, I-EVIDENCE  
B-VICTIM, I-VICTIM  
O
```

---

## Architecture

```
Tokens
↓
RoBERTa Encoder (Domain-Adaptively Pretrained)
↓
Linear Emission Layer
↓
CRF (BIO-constrained decoding)
↓
Span-level Marker Predictions
```

### Component Roles

**Domain-Adaptive Pretraining (DAPT)**  
- Continued masked language modeling on Reddit conspiracy text  
- Adapts contextual representations to domain-specific discourse  

**CRF Layer**  
- Enforces valid BIO transitions  
- Produces globally consistent label sequences  
- Reduces fragmented or invalid spans  

---

## File Structure

```
Hadia_Roberta_Dapt_CRF/
│
├── data/
│   ├── train_rehydrated.jsonl
│   └── dev_rehydrated.jsonl
│
├── roberta_dapt/
│   └── checkpoint-*/        (ignored in GitHub)
│
├── plots_roberta_dapt_crf/
│   ├── roberta_crf_f1_per_label.pdf
│   ├── confusion_matrix.pdf
│   └── learning_curve.pdf
│
├── results_roberta_dapt_crf/
│   ├── roberta_crf_best_metrics.json
│   └── roberta_crf_history.json
│
├── reddit_domain_corpus.txt
└── README.md
```

---

## Domain-Adaptive Pretraining (DAPT)

The RoBERTa encoder is further pretrained using **Masked Language Modeling (MLM)**
on a Reddit conspiracy corpus to capture domain-specific lexical and stylistic
patterns.

---

## Training (DAPT + CRF)

```
python train_roberta_crf.py \
  --train_path data/train_rehydrated.jsonl \
  --model_name roberta_dapt \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_length 192 \
  --results_dir results_roberta_dapt_crf \
  --plots_dir plots_roberta_dapt_crf
```

Training is configured to be memory-safe using gradient accumulation and supports
both CPU and GPU execution.

---

## Evaluation

```
python evaluate_roberta_crf.py \
  --train_path data/train_rehydrated.jsonl \
  --model_path results_roberta_dapt_crf/roberta_crf_best.pth
```

Evaluation outputs include per-label F1 scores, confusion matrices, and learning
curves saved as PDF files.

---

## Results (Validation)

| Metric       | Score (Approx.) |
|-------------|------------------|
| Weighted F1 | ~0.64            |
| Macro F1    | ~0.11–0.14       |

---

## Error & Class Imbalance Analysis

- Label distribution is highly skewed toward the **O** class  
- Rare marker categories (Effect, Evidence, Victim) suffer from low recall  
- DAPT improves domain fluency but does **not sufficiently resolve class imbalance**  
- Most errors are false negatives for underrepresented markers  

These findings indicate the need for future work involving data augmentation or
cost-sensitive objectives.

---

## Assignment Contribution

**Member:** Hadia Ramzan  

**Responsibility:** Domain Adaptation & Analysis  

- Performed Domain-Adaptive Pretraining (DAPT) on Reddit conspiracy text  
- Integrated the adapted encoder with CRF-based sequence tagging  
- Conducted class imbalance and error analysis  
- Generated confusion matrices and learning curves  

---

## References

- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019  
- Gururangan et al., *Don’t Stop Pretraining*, ACL 2020  
- Lafferty et al., *Conditional Random Fields*, ICML 2001  
- SemEval 2026 Task 10  
- HuggingFace Transformers  
- pytorch-crf  
