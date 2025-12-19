import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class RoBERTaCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size, num_labels
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(
                emissions,
                labels,
                mask=attention_mask.bool(),
                reduction='mean'
            )
            return {"loss": loss}
        else:
            predictions = self.crf.decode(
                emissions, mask=attention_mask.bool()
            )
            return {"predictions": predictions}
