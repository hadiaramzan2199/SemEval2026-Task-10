
# README — RoBERTa Baseline for Psycholinguistic Marker Extraction

## 1. Overview

This folder contains the implementation of a **RoBERTa/XLM-RoBERTa** baseline for **SemEval 2026 Task 10 — Psycholinguistic Conspiracy Marker Extraction (Subtask 1)**.

The task aims to extract span-level markers such as:

* **Actor**
* **Action**
* **Effect**
* **Evidence**
* **Victim**

Each Reddit comment is converted into **BIO sequence tags** and evaluated using **sequence labeling metrics**.

---

## 2. Model Description

### Model: XLM-RoBERTa-base with Token Classification Head

* Base Model: XLM-RoBERTa-base (multilingual, 125M parameters)
* Architecture: Transformer encoder with linear classification head
* Task: Token-level sequence labeling (BIO prediction)
* Tokenizer: XLM-RoBERTa tokenizer with subword tokenization

### Model Architecture:

Input Text → Tokenization → XLM-RoBERTa Encoder → Linear Layer → BIO Predictions
                    ↑
             Subword-to-word alignment

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

* Optimizer: AdamW with weight decay
* Learning Rate: 2e-5 with linear warmup and decay
* Loss: Cross-entropy with ignore_index=-100 for special tokens
* Batch Size: 8 (adjustable based on GPU memory)

---

## 3. File Structure

```
.
├── model_roberta_ner.py
├── train_roberta.py 
├── model_baseline_B.py 
├── evaluate.py
├── data_loader.py 
├── preprocess.py 
├── requirements.txt
├── README.md
├── plots/
│   └── roberta_ner_f1_per_label.png
└── results/
    ├── roberta_ner_model.pth 
    ├── roberta_ner_metrics.json
    └── roberta_ner_model_tokenizer/ 
        ├── tokenizer_config.json
        ├── tokenizer.model
        ├── tokenizer.spiece

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
../data/train_rehydrated.jsonl
```

## 5. Running the RoBERTa Baseline

### Training

```bash
python train_roberta.py --train_path ../data/train_rehydrated.jsonl --epochs 10 --batch_size 8
```

### Arguments:

* --train_path: Path to training data (default: ../data/train_rehydrated.jsonl)
* --epochs: Number of training epochs (default: 3)
* --batch_size: Batch size for training (default: 8)
* --results_dir: Directory to save results (default: results)
* --plots_dir: Directory to save plots (default: plots)

This will:

* Load and preprocess data into BIO format
* Initialize XLM-RoBERTa model with classification head
* Train for specified epochs with validation monitoring
* Save model, metrics, and visualizations

### Evaluation on validation set

```bash
python evaluate.py --model_path results/roberta_ner_model.pth --train_path ../data/train_rehydrated.jsonl
```

This will:

* Load trained model
* Evaluate on validation set
* Print weighted F1, macro F1, and per-label performance

---

## 6. Output Files

### Model Checkpoint

```
results/roberta_ner_model.pth
```
```
results/roberta_ner_model_tokenizer/ 
```

### Metrics

```
results/roberta_ner_metrics.json
```

Contains:

```json
{
  "f1_weighted": 0.6309,
  "f1_macro": 0.1517,
  "per_label_f1": {
    "B-ACTION": 0.0469,
    "B-ACTOR": 0.1903,
    "B-EFFECT": 0.0098,
    "B-EVIDENCE": 0.0157,
    "B-VICTIM": 0.0271,
    "I-ACTION": 0.1282,
    "I-ACTOR": 0.1905,
    "I-EFFECT": 0.1212,
    "I-EVIDENCE": 0.0404,
    "I-VICTIM": 0.0673,
    "O": 0.8311
  }
}
```

### Plot

[plots/roberta_ner_f1_per_label.png](https://github.com/hadiaramzan2199/SemEval2026-Task-10/blob/main/LLM-Assignment-2/Hadia Roberta/plots/roberta_ner_f1_per_label.png)

---

## 7. Experimental Setup

### Data Split

| Set            | Size |
| -------------- | ---- |
| Train          | 80%  |
| Validation     | 20%  |
| Total Corpus   | 4,316|

### Hyperparameters

| Parameter              | Value               | Description                         |
|------------------------|---------------------|-------------------------------------|
| Base Model             | XLM-RoBERTa-base    | 125M parameters, multilingual       |
| Learning Rate          | 2e-5                | AdamW optimizer                     |
| Batch Size             | 8                   | Limited by GPU memory               |
| Epochs                 | 3                   | Training iterations                 |
| Max Sequence Length    | 256                 | Token limit                         |
| Dropout                | 0.1                 | Regularization                      |
| Warmup Steps           | 10%                 | Linear learning rate warmup         |

---

## 8. Results


### Validation Performance

| Metric         | RoBERTa Baseline |
|----------------|------------------|
| Weighted F1    | 0.6309           |
| Macro F1       | 0.1517           |
| Training Time  | 25 min (GPU)     |


### Per-label 

| Label        | F1 Score | Frequency |
|--------------|----------|-----------|
| O            | 0.8311   | 74.2%     |
| I-ACTOR      | 0.1905   | 2.1%      |
| B-ACTOR      | 0.1903   | 2.1%      |
| I-ACTION     | 0.1282   | 5.1%      |
| I-EFFECT     | 0.1212   | 5.3%      |
| I-VICTIM     | 0.0673   | 1.0%      |
| B-ACTION     | 0.0469   | 1.5%      |
| I-EVIDENCE   | 0.0404   | 5.3%      |
| B-VICTIM     | 0.0271   | 1.0%      |
| B-EVIDENCE   | 0.0157   | 1.2%      |
| B-EFFECT     | 0.0098   | 1.2%      |

---

## 9. Analysis & Discussion

### Strengths:

* Contextual Understanding: RoBERTa captures deep semantic relationships beyond local context
* Multilingual Support: XLM-RoBERTa handles English-Hindi code-mixed text natively
* Subword Tokenization: Better handling of out-of-vocabulary words and social media text
* No Feature Engineering: Learns representations automatically from data

### Limitations:

* Class Imbalance: Severe performance disparity between frequent (O: 0.83 F1) and rare classes (B-EFFECT: 0.01 F1)
* Training Efficiency: 3 epochs insufficient for transformer convergence; CRF trains faster
* Computational Cost: Requires GPU for practical training times
* Span Alignment: Subword tokenization complicates exact span boundary prediction


---


## 11. References

```markdown
- Lafferty et al., "Conditional Random Fields", ICML 2001
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- SemEval-2026 Task 10: Detection and Analysis of Conspiracy Theories.
Proceedings of the 15th International Workshop on Semantic Evaluation, 2026.
- Taubert, F., Meyer-Hoeven, G., Schmid, P., Gerdes, P., \& Betsch, C. (2024). Conspiracy narratives and vaccine hesitancy: a scoping review of prevalence, impact, and interventions
- sklearn-crfsuite documentation
```

---

## 12. Author

```markdown
Student Name: Hadia Ramzan
Model Responsibility: RoBERTa Baseline
Course: LLM — SemEval 2026 Project
Date: Dec 2025
```

---
