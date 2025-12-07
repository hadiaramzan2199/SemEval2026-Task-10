# CRF Baseline — Subtask 1

This is a baseline system using a Conditional Random Field (CRF) model to extract psycholinguistic markers:

- ACTOR
- ACTION
- EFFECT
- EVIDENCE
- VICTIM

## Files

- data_loader.py — load dataset
- preprocess.py — BIO conversion + split
- model_baseline_A.py — features + CRF model
- train.py — train model
- evaluate.py — run inference
