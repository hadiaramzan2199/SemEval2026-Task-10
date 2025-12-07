import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class NERDataset(Dataset):
    """Dataset class for NER task with marker annotations"""
    
    def __init__(self, data, tokenizer, tag2id, max_len=128):
        """
        Args:
            data: List of dictionaries with 'text' and 'markers'
            tokenizer: HuggingFace tokenizer
            tag2id: Dictionary mapping tags to IDs
            max_len: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
        self.encodings = self._process_data()
    
    def _process_data(self):
        """Tokenize and align labels for all examples"""
        encodings = []
        for item in self.data:
            encodings.append(self._tokenize_and_align_labels(item))
        return encodings
    
    def _tokenize_and_align_labels(self, item):
        """Tokenize text and create label alignment"""
        text = item["text"]
        labels = ["O"] * len(text)
        
        # Create character-level labels
        for m in item.get("markers", []):
            start, end = m["startIndex"], m["endIndex"]
            labels[start] = "B-" + m["type"]
            for i in range(start + 1, end):
                labels[i] = "I-" + m["type"]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_offsets_mapping=True
        )
        
        # Convert to token-level labels
        word_labels = []
        for start, end in encoding["offset_mapping"]:
            if start == end:  # Special tokens
                word_labels.append(self.tag2id["O"])
            else:
                word_labels.append(self.tag2id[labels[start]])
        
        encoding.pop("offset_mapping")
        encoding["labels"] = word_labels
        return encoding
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = self.encodings[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


def create_dataloader(data, tokenizer, tag2id, batch_size=16, shuffle=True, max_len=128):
    """Create DataLoader for training/evaluation"""
    dataset = NERDataset(data, tokenizer, tag2id, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_tag_mappings(data):
    """Extract all tag types from data and create mappings"""
    all_tags = set()
    for item in data:
        for m in item.get("markers", []):
            all_tags.add("B-" + m["type"])
            all_tags.add("I-" + m["type"])
    all_tags.add("O")
    
    tag2id = {t: i for i, t in enumerate(sorted(all_tags))}
    id2tag = {i: t for t, i in tag2id.items()}
    
    return tag2id, id2tag, all_tags