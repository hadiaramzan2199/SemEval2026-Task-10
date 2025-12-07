# NER Baseline Model for SemEval Task 10

## Project Overview
This project implements a BiLSTM-based Named Entity Recognition (NER) model for detecting markers (Actor, Action, Effect, Victim, Evidence) in human rights documentation text.

## Model Architecture
- **Embedding Layer**: 128-dimensional embeddings
- **BiLSTM Layer**: 256 hidden units, bidirectional
- **Linear Layer**: Output layer for 11 classes (B/I tags for 5 markers + O)

## Training Parameters
- **Epochs**: 7
- **Batch Size**: 16
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Max Sequence Length**: 128 tokens

## Performance Metrics
- **Overall Overlap F1**: ~0.43
- **Macro F1**: ~0.394
- **Weighted F1**: ~0.438

## Per-Class Performance (Overlap F1):
- Actor: 0.580
- Action: 0.455
- Effect: 0.297
- Victim: 0.346
- Evidence: 0.295

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Download and preprocess data using rehydrate_data.py
3. Train model: `python train.py`
4. Evaluate: `python evaluate.py`

## File Structure
- `data_loader.py`: Dataset and DataLoader classes
- `preprocess.py`: Text cleaning and tokenization functions
- `model_baseline.py`: BiLSTM model architecture
- `train.py`: Training loop and loss tracking

- `evaluate.py`: Evaluation metrics and visualization
