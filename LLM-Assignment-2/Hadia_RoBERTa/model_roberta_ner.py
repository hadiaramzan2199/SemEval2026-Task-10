import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, XLMRobertaModel
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import json
import os

class RoBERTaForTokenClassification(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=12, class_weights=None):
        super(RoBERTaForTokenClassification, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.class_weights = class_weights
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if self.class_weights is not None:
                weights = self.class_weights.to(labels.device)
                loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        return {"logits": logits, "loss": loss}

class RoBERTaSequenceTagger:
    def __init__(self, model_name="xlm-roberta-base", device=None):
        # Label list matching your CRF baseline
        self.label_list = [
            "O",
            "B-ACTION", "I-ACTION",
            "B-ACTOR", "I-ACTOR",
            "B-EFFECT", "I-EFFECT",
            "B-EVIDENCE", "I-EVIDENCE",
            "B-VICTIM", "I-VICTIM"
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RoBERTaForTokenClassification(
            model_name=model_name,
            num_labels=len(self.label_list)
        )
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.max_length = 256
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.num_epochs = 10
        
    def encode_samples(self, corpus):
        input_ids, attention_masks, label_ids = [], [], []
        
        for sample in corpus:
            tokens = sample["tokens"]
            bio_labels = sample["labels"]
            
            token_inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            word_ids = token_inputs.word_ids()
            previous_word_idx = None
            sample_label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    sample_label_ids.append(-100)
                elif word_idx == previous_word_idx:
                    sample_label_ids.append(-100)
                else:
                    label = bio_labels[word_idx] if word_idx < len(bio_labels) else "O"
                    sample_label_ids.append(self.label2id.get(label, self.label2id["O"]))
                previous_word_idx = word_idx
            
            input_ids.append(token_inputs["input_ids"].squeeze())
            attention_masks.append(token_inputs["attention_mask"].squeeze())
            label_ids.append(torch.tensor(sample_label_ids))
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(label_ids)
        }
    
    def create_dataloader(self, encoded_data, shuffle=True):
        dataset = torch.utils.data.TensorDataset(
            encoded_data["input_ids"],
            encoded_data["attention_mask"],
            encoded_data["labels"]
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(self, train_corpus, val_corpus=None):
        print(f"Training RoBERTa on {len(train_corpus)} samples")
        print(f"Using device: {self.device}")
        
        train_encoded = self.encode_samples(train_corpus)
        train_loader = self.create_dataloader(train_encoded, shuffle=True)
        
        if val_corpus:
            val_encoded = self.encode_samples(val_corpus)
            val_loader = self.create_dataloader(val_encoded, shuffle=False)
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch in progress_bar:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Average training loss: {avg_loss:.4f}")
            
            if val_corpus:
                val_results = self.evaluate(val_corpus)
                print(f"Validation F1 (weighted): {val_results['f1_weighted']:.4f}")
                print(f"Validation F1 (macro): {val_results['f1_macro']:.4f}")
        
        print("\nTraining completed!")
    
    def predict(self, corpus):
        self.model.eval()
        encoded_data = self.encode_samples(corpus)
        dataloader = self.create_dataloader(encoded_data, shuffle=False)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
                
                for i in range(predictions.shape[0]):
                    pred_seq = []
                    for j in range(predictions.shape[1]):
                        if encoded_data["labels"][i][j] != -100:
                            label_id = predictions[i][j].item()
                            pred_seq.append(self.id2label[label_id])
                    all_predictions.append(pred_seq)
        
        return all_predictions
    
    def evaluate(self, corpus):
        predictions = self.predict(corpus)
        
        true_labels = []
        pred_labels = []
        
        for i, sample in enumerate(corpus):
            true_seq = sample["labels"]
            pred_seq = predictions[i]
            
            min_len = min(len(true_seq), len(pred_seq))
            true_labels.extend(true_seq[:min_len])
            pred_labels.extend(pred_seq[:min_len])
        
        report_dict = classification_report(
            true_labels, 
            pred_labels, 
            output_dict=True,
            zero_division=0
        )
        report_str = classification_report(
            true_labels, 
            pred_labels, 
            zero_division=0
        )
        
        f1_weighted = report_dict["weighted avg"]["f1-score"]
        f1_macro = report_dict["macro avg"]["f1-score"]
        
        per_label_f1 = {}
        for label in self.label_list:
            if label in report_dict:
                per_label_f1[label] = report_dict[label]["f1-score"]
        
        return {
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "per_label_f1": per_label_f1,
            "report_str": report_str,
            "predictions": predictions
        }
    
    def save(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label2id': self.label2id,
            'id2label': self.id2label,
            'max_length': self.max_length
        }, model_path)
        
        tokenizer_path = model_path.replace(".pth", "_tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(device=device)
        model.label2id = checkpoint['label2id']
        model.id2label = checkpoint['id2label']
        model.max_length = checkpoint['max_length']
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        
        tokenizer_path = path.replace(".pth", "_tokenizer")
        model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        return model
