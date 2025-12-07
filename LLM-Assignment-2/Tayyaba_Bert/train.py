import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys

# Import local modules
from data_loader import load_data, get_tag_mappings, NERDataset
from model_baseline import initialize_model
from evaluate import overlap_f1, overlap_f1_per_class, plot_training_loss, plot_per_class_f1
from preprocess import get_spans

def main():
    # Configuration
    DATA_FILE = "train_rehydrated.jsonl"
    EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} not found. Please ensure data is rehydrated/available.")
        return

    # 1. Load Data
    print("Loading data...")
    data = load_data(DATA_FILE)
    tag2id, id2tag = get_tag_mappings(data)
    
    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = NERDataset(data, tokenizer, tag2id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model
    model = initialize_model(tag2id, id2tag, device=DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 4. Training
    loss_history = []
    print(f"Starting Training for {EPOCHS} epochs on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = total_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f}")
        
        # Checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'tag2id': tag2id,
            'id2tag': id2tag
        }, f'checkpoint_epoch_{epoch+1}.pt')

    plot_training_loss(loss_history)

    # 5. Evaluation
    print("Starting Evaluation...")
    all_preds_spans, all_labels_spans = [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(-1).cpu()

            for i in range(preds.shape[0]):
                length = (batch["attention_mask"][i] != 0).sum().item()
                pred_seq = preds[i].tolist()[:length]
                label_seq = labels[i].tolist()[:length]
                
                all_preds_spans.append(get_spans(pred_seq, id2tag))
                all_labels_spans.append(get_spans(label_seq, id2tag))

    precision, recall, f1 = overlap_f1(all_preds_spans, all_labels_spans)
    print(f"Overlap Precision: {precision}")
    print(f"Overlap Recall: {recall}")
    print(f"Overlap F1: {f1}")

    per_class_results = overlap_f1_per_class(all_preds_spans, all_labels_spans)
    plot_per_class_f1(per_class_results)
    
    # Save Final Model
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model saved.")

if __name__ == "__main__":
    main()
