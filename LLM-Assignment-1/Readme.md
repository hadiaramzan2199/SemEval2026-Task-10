# PsyCoMark — EDA for SemEval Task 10 (Subtask 1)

This repository contains exploratory data analysis (EDA) for SemEval-2026 Task 10: Psycholinguistic Conspiracy Marker Extraction and Detection, specifically Subtask 1: Extraction of psycholinguistic conspiracy markers such as Actors, Victims, Effects, etc.

The dataset used is:

- PsyCoMark — Psycholinguistic Conspiracy Marker Dataset
- Source: Zenodo
- Format: JSONL with annotated text and marker spans

This EDA provides dataset statistics, token-level analysis, marker distribution, linguistic patterns, and domain-specific insights to support downstream model development.

---

## Project Structure

├── plots/ # Folder containing saved visualizations
├── LLM_proj_Assign_1_code(draft).ipynb
├── README.md
└── requirements.txt

Our analysis includes:
- Token length distribution  
- Label imbalance  
- POS tags  
- N-gram frequencies  
- Sentiment distribution  
- Language mix  
- Marker frequency and distribution  
- Marker co-occurrence heatmap  
- Missing-pair analysis (text vs markers)  
- Span-level consistency checks  

## Features Covered in EDA

### **1. Dataset Overview**

- Number of documents
- Marker type distribution
- Example annotated samples
- Token statistics (mean, median, max length)

### **2. Text-Level EDA**

- Token length distribution
- Most frequent tokens (stopwords removed)
- POS tag distribution (spaCy)
- N-gram (bi/tri-gram) frequency
- Sentiment distribution
- Language-mix detection (langdetect)

### **3. Marker-Level EDA**

- Marker type counts
- Marker span-length distribution
- Comparison of POS tags inside markers vs. normal text

### **4. Multimodal Checks**

_(Only if present)_

- Missing modality pairs
- Alignment between modalities
- Correlation statistics

### **5. Plot Saving**

All visualizations are saved automatically into `plots/`:

## Running the Notebook

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

Download the dataset from Zenodo and place it in the project root.

Open the notebook:

jupyter notebook LLM*proj_Assignment_1*.ipynb
