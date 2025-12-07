import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from preprocess import clean_text_safe
import json
import os

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, tag2id, max_len=128):
        self.encodings = [self.tokenize_and_align_labels(item, tokenizer, tag2id, max_len) for item in data]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return {k: torch.tensor(v) for k, v in item.items()}

    def tokenize_and_align_labels(self, item, tokenizer, tag2id, max_len):
        text = item["text"]
        labels = ["O"] * len(text)
        for m in item.get("markers", []):
            start, end = m["startIndex"], m["endIndex"]
            if start < len(labels):
                labels[start] = "B-" + m["type"]
                for i in range(start+1, min(end, len(labels))):
                    labels[i] = "I-" + m["type"]
        
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_offsets_mapping=True
        )
        
        word_labels = []
        for start, end in encoding["offset_mapping"]:
            if start == end:   # Special tokens
                word_labels.append(tag2id["O"])
            elif start < len(labels):
                 word_labels.append(tag2id[labels[start]])
            else:
                 word_labels.append(tag2id["O"]) 

        encoding.pop("offset_mapping")
        encoding["labels"] = word_labels
        return encoding

def load_data(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                item["text"] = clean_text_safe(item["text"])
                data.append(item)
    return data

def get_tag_mappings(data):
    all_tags = set()
    for item in data:
        for m in item.get("markers", []):
            all_tags.add("B-" + m["type"])
            all_tags.add("I-" + m["type"])
    all_tags.add("O")
    
    tag2id = {t: i for i, t in enumerate(sorted(all_tags))}
    id2tag = {i: t for t, i in tag2id.items()}
    return tag2id, id2tag
