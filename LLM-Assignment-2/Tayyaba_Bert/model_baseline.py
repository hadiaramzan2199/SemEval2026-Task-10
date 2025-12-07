from transformers import AutoModelForTokenClassification
import torch

def initialize_model(tag2id, id2tag, model_name="bert-base-uncased", device="cpu"):
    print(f"Initializing Model {model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id
    ).to(device)
    return model
