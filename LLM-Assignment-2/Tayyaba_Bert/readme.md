# BERT Analysis for SemEval Task 10

This repository contains the code for training and evaluating a BERT model for Token Classification (NER).

## Directory Structure

- `data_loader.py`: Handles dataset loading and tokenization.
- `model_baseline.py`: Defines the BERT model architecture.
- `train.py`: Main script to train the model.
- `evaluate.py`: Evaluation metrics and plotting functions.
- `preprocess.py`: Utility functions for text cleaning.
- `requirements.txt`: Python dependencies.
- `Plots/`: Directory where training loss and F1 plots will be saved.

## Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    Ensure `train_rehydrated.jsonl` is in the root directory.

3.  **Train**:
    ```bash
    python train.py
    ```

4.  **Results**:
    -   Trained model will be saved to `saved_model/`.
    -   Plots will be saved to `Plots/`.
