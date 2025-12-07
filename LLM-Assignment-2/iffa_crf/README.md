# README — CRF Baseline for Psycholinguistic Marker Extraction

## 1. Overview

This folder contains the implementation of a **Conditional Random Field (CRF)** baseline for **SemEval 2026 Task 10 — Psycholinguistic Conspiracy Marker Extraction (Subtask 1)**.

The task aims to extract span-level markers such as:

* **Actor**
* **Action**
* **Effect**
* **Evidence**
* **Victim**

Each Reddit comment is converted into **BIO sequence tags** and evaluated using **sequence labeling metrics**.

---

## 2. Model Description

### Model: CRF (Conditional Random Field)

* Linear-chain CRF for BIO prediction
* Token-level input, no neural components
* Feature-based model with contextual window

### Token-level features:

| Category | Examples                                        |
| -------- | ----------------------------------------------- |
| Lexical  | token.lower(), token.isdigit(), token.isupper() |
| Shape    | prefix, suffix, word shape                      |
| Context  | prev/next token features                        |
| Flags    | contains digit, punctuation, uppercase          |

### Objective

Predict labels from tag set:

```
O
B-ACTOR, I-ACTOR
B-ACTION, I-ACTION
B-EFFECT, I-EFFECT
B-EVIDENCE, I-EVIDENCE
B-VICTIM, I-VICTIM
```

### Optimization

* L-BFGS
* L2 regularization
* max_iterations = 100

---

## 3. File Structure

```
.
├── data_loader.py
├── preprocess.py
├── train.py
├── evaluate.py
├── utils.py
├── models/
│   └── model_crf.py
├── plots/
│   └── baseline_crf_f1_per_label.png
└── results/
    ├── baseline_crf.pkl
    └── baseline_crf_metrics.json
```

---

## 4. Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

Download from Zenodo, place here:

```
data/train_redacted.jsonl
```

### 3. Rehydrate raw text

```bash
python rehydrate_data.py --input data/train_redacted.jsonl --output data/train_rehydrated.jsonl
```

---

## 5. Running the CRF Baseline

### Training

```bash
python train.py --train_path data/train_rehydrated.jsonl
```

This will:

* Preprocess text → BIO
* Train CRF
* Save model + metrics + plots

### Evaluation on validation set

```bash
python evaluate.py --train_path data/train_rehydrated.jsonl --model_path results/baseline_crf.pkl
```

This will print:

* Weighted F1
* Macro F1
* Per-label performance

---

## 6. Output Files

### Model Checkpoint

```
results/baseline_crf.pkl
```

### Metrics

```
results/baseline_crf_metrics.json
```

Contains:

```json
{
  "f1_weighted": 0.7110065455838496,
  "f1_macro": 0.32434026124811777,
  "per_label_f1": {
    "B-ACTION": 0.21237458193979933,
    "B-ACTOR": 0.3026706231454006,
    "B-EFFECT": 0.20704845814977973,
    "B-EVIDENCE": 0.18518518518518517,
    "B-VICTIM": 0.22759601706970128,
    "I-ACTION": 0.28158679202017883,
    "I-ACTOR": 0.32372281234193223,
    "I-EFFECT": 0.28981937602627256,
    "I-EVIDENCE": 0.3088180112570357,
    "I-VICTIM": 0.3682795698924731,
    "O": 0.8606414467015367
  }
}
```

### Plot

```
[plots/baseline_crf_f1_per_label.png
](https://github.com/hadiaramzan2199/SemEval2026-Task-10/blob/main/LLM-Assignment-2/iffa_crf/plots/baseline_crf_f1_per_label.png)```

---

## 7. Experimental Setup

### Data Split

| Set            | Size |
| -------------- | ---- |
| Train          | 80%  |
| Validation     | 20%  |
| Stratification | N/A  |

### Hyperparameters

| Parameter      | Value  |
| -------------- | ------ |
| Algorithm      | L-BFGS |
| L2             | 0.1    |
| Max Iter       | 100    |
| Context Window | ±1     |

---

## 8. Results

```markdown
### Validation Performance

| Metric | Score |
|---|---|
| Weighted F1 | 0.71 |
| Macro F1 | 0.32 |
```

### Per-label (example)

| Label      | F1   |
| ---------- | ---- |
| B-ACTION    | 0.21 |
| B-ACTOR    | 0.30 |
| B-EFFECT   | 0.20 |
| B-EVIDENCE | 0.18 |
| B-VICTIM   | 0.22 |
| I-ACTION   | 0.28 |
| I-ACTOR    | 0.32 |
| I-EFFECT   | 0.28 |
| I-EVIDENCE | 0.31 |
| I-VICTIM   | 0.36 |
| O          | 0.86 |

---

## 9. Known Limitations

* Tokenizer is whitespace-based → poor span alignment
* Hand-crafted features miss semantic cues
* Class imbalance heavily affects rare labels
* Struggles with multi-token spans

Overall performance is expected to be low, but the model is valuable as a **interpretable baseline**.

---

## 10. Future Improvements

Potential enhancements for later assignment:

* Character-level BiLSTM features
* Contextual embeddings (BERT)
* Joint span prediction
* Data augmentation
* CRF + transformer hybrid

---

## 11. References

Add these in your report and README:

```markdown
- Lafferty et al., "Conditional Random Fields", ICML 2001
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- SemEval 2026 Task 10 Official Website
- sklearn-crfsuite documentation
```

---

## 12. Author

```markdown
Student Name: __________
Model Responsibility: CRF Baseline
Course: ML/NLP — SemEval 2026 Project
Date: Dec 2025
```

---

