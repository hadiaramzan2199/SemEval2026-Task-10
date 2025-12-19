#model_roberta_crf.py

import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF  # <-- IMPORTANT: torchcrf


class RoBERTaCRFTagger(nn.Module):
    """
    RoBERTa encoder -> Linear emissions -> CRF decode
    Trains BIO tags. Handles pre-tokenized word list using add_prefix_space=True.
    """
    def __init__(self, model_name: str, label_list: List[str], max_length: int = 256):
        super().__init__()
        self.model_name = model_name
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for i, l in enumerate(label_list)}
        self.max_length = max_length

        # Roberta needs prefix space for pretokenized inputs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, len(label_list))

        # torchcrf supports batch_first=True
        self.crf = CRF(num_tags=len(label_list), batch_first=True)

        self.o_id = self.label2id["O"]

    def forward(self, input_ids, attention_mask, labels=None, crf_mask=None):
        """
        input_ids: (B,L)
        attention_mask: (B,L) 1 for real tokens
        labels: (B,L) label ids (O for specials/subwords we don't care)
        crf_mask: (B,L) bool mask for CRF (must have first timestep ON)
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = self.dropout(out.last_hidden_state)
        emissions = self.classifier(x)  # (B,L,C)

        if crf_mask is None:
            crf_mask = attention_mask.bool()

        # CRITICAL FIX: first timestep must be ON for all sequences
        crf_mask[:, 0] = True

        if labels is not None:
            # torchcrf expects tags within [0..C-1] everywhere, masked positions ignored.
            # so ensure labels are valid ints (already O where needed)
            log_likelihood = self.crf(emissions, labels, mask=crf_mask, reduction="mean")
            loss = -log_likelihood
            return {"loss": loss, "emissions": emissions}

        pred_paths = self.crf.decode(emissions, mask=crf_mask)  # list[list[int]]
        return {"pred_paths": pred_paths, "emissions": emissions}

    def encode_sentence(self, sample: Dict):
        """
        sample: {"tokens":[...], "labels":[...] (optional)}
        Returns tensors for one sentence:
          input_ids, attention_mask, labels, word_mask
        word_mask=1 for first-subword of each word token (plus CLS to satisfy CRF)
        """
        words = sample["tokens"]
        gold = sample.get("labels", None)

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)          # (L,)
        attn = enc["attention_mask"].squeeze(0)          # (L,)
        word_ids = enc.word_ids(batch_index=0)

        # Build labels over full sequence length
        labels_out = []
        word_mask = []

        prev_word = None
        for pos, widx in enumerate(word_ids):
            if widx is None:
                # special tokens ([CLS], [SEP], padding)
                labels_out.append(self.o_id)
                word_mask.append(1 if pos == 0 else 0)  # keep CLS word_mask ON for CRF first step
            elif widx == prev_word:
                # continuation subword: ignore from word-level eval -> mask off
                labels_out.append(self.o_id)
                word_mask.append(0)
            else:
                # first subword of a word
                word_mask.append(1)
                if gold is None:
                    labels_out.append(self.o_id)
                else:
                    lab = gold[widx] if widx < len(gold) else "O"
                    labels_out.append(self.label2id.get(lab, self.o_id))
            prev_word = widx

        labels_out = torch.tensor(labels_out, dtype=torch.long)
        word_mask = torch.tensor(word_mask, dtype=torch.bool)

        # CRF mask: we want to include CLS + first-subword positions only
        crf_mask = (attn.bool() & word_mask)
        crf_mask[0] = True  # ensure first timestep ON

        return input_ids, attn, labels_out, crf_mask, word_mask

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "model_name": self.model_name,
            "label_list": self.label_list,
            "max_length": self.max_length,
            "state_dict": self.state_dict(),
        }
        torch.save(ckpt, path)
        tok_dir = path.replace(".pth", "_tokenizer")
        self.tokenizer.save_pretrained(tok_dir)

    @classmethod
    def load(cls, path: str, device=None):
        ckpt = torch.load(path, map_location=device)
        model = cls(
            model_name=ckpt["model_name"],
            label_list=ckpt["label_list"],
            max_length=ckpt.get("max_length", 256),
        )
        model.load_state_dict(ckpt["state_dict"])
        tok_dir = path.replace(".pth", "_tokenizer")
        model.tokenizer = AutoTokenizer.from_pretrained(tok_dir, add_prefix_space=True)
        return model
