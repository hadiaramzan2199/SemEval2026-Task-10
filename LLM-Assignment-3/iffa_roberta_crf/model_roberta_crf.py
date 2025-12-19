#model_roberta_crf.py

import os
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF


class RoBERTaCRF(nn.Module):
    """
    RoBERTa encoder -> Linear emissions -> CRF decode.
    We train only on FIRST-subword positions (valid_mask).
    """
    def __init__(self, model_name: str, label_list: List[str], freeze_encoder: bool = False):
        super().__init__()
        self.model_name = model_name
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for i, l in enumerate(label_list)}

        # IMPORTANT for is_split_into_words=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, len(label_list))

        # torchcrf supports batch_first
        self.crf = CRF(num_tags=len(label_list), batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        labels: (B, L) with -100 for invalid positions
        valid_mask: (B, L) True only on FIRST subword of each word token
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = self.dropout(out.last_hidden_state)
        emissions = self.classifier(x)  # (B, L, C)

        if valid_mask is None:
            valid_mask = attention_mask.bool()

        if labels is not None:
            # Replace -100 with 0 (will be ignored by mask)
            labels_safe = labels.clone()
            labels_safe[labels_safe == -100] = 0

            # CRF returns log-likelihood; minimize -loglik
            log_likelihood = self.crf(emissions, labels_safe, mask=valid_mask.bool(), reduction="mean")
            loss = -log_likelihood
            return {"loss": loss}

        # decode
        paths = self.crf.decode(emissions, mask=valid_mask.bool())
        return {"predictions": paths}

    def save(self, path: str, max_length: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "model_name": self.model_name,
            "label_list": self.label_list,
            "state_dict": self.state_dict(),
            "max_length": max_length,
        }
        torch.save(ckpt, path)
        tok_dir = path.replace(".pth", "_tokenizer")
        self.tokenizer.save_pretrained(tok_dir)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        ckpt = torch.load(path, map_location=device)
        model = cls(ckpt["model_name"], ckpt["label_list"])
        model.load_state_dict(ckpt["state_dict"])
        tok_dir = path.replace(".pth", "_tokenizer")
        model.tokenizer = AutoTokenizer.from_pretrained(tok_dir, add_prefix_space=True)
        return model, ckpt.get("max_length", 128)

