import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
import json

from data_loader import create_dataloader, get_tag_mappings
from model_baseline import create_model
import preprocess

# Training configuration
CONFIG = {
    "epochs": 7,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "max_len": 128,
    "emb_dim": 128,
    "hidden_dim": 256,
    "dropout": 0.1,
    "model_name": "bert-base-uncased"
}

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        
        # Calculate loss
        loss = loss_fn(
            outputs.view(-1, outputs.shape[-1]),
            labels.view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, config, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def plot_training_history(loss_history, save_path=None):
    """Plot training loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o', linewidth=2)
    plt.title("Training Loss over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(loss_history)), range(1, len(loss_history) + 1))
    
    # Annotate the last loss value
    last_loss = loss_history[-1]
    plt.annotate(f'{last_loss:.4f}', 
                 xy=(len(loss_history)-1, last_loss),
                 xytext=(len(loss_history)-1.5, last_loss+0.05),
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")
    
    plt.show()


def main():
    """Main training function"""
    print("=" * 60)
    print("Starting NER Model Training")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data = preprocess.load_and_preprocess_data("train_rehydrated.jsonl")
    
    # Get tag mappings
    print("\n2. Creating tag mappings...")
    tag2id, id2tag, all_tags = get_tag_mappings(data)
    print(f"  Found {len(all_tags)} tags: {sorted(all_tags)}")
    
    # Initialize tokenizer
    print("\n3. Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    
    # Create data loader
    print("\n4. Creating data loader...")
    train_loader = create_dataloader(
        data, tokenizer, tag2id,
        batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"]
    )
    print(f"  Number of batches: {len(train_loader)}")
    
    # Create model
    print("\n5. Creating model...")
    model = create_model(
        vocab_size=len(tokenizer),
        device=device,
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_labels=len(tag2id),
        dropout=CONFIG["dropout"]
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n6. Starting training...")
    print("=" * 60)
    
    loss_history = []
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 40)
        
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        loss_history.append(epoch_loss)
        
        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss:.4f}")
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pt"
            save_checkpoint(
                model, optimizer, epoch + 1, epoch_loss,
                CONFIG, checkpoint_path
            )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Save final model
    final_model_path = "checkpoints/model_final.pt"
    save_checkpoint(model, optimizer, CONFIG["epochs"], loss_history[-1], CONFIG, final_model_path)
    
    # Plot and save training history
    plot_path = "plots/training_loss.png"
    plot_training_history(loss_history, plot_path)
    
    # Save training configuration and results
    results = {
        "config": CONFIG,
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "tag_mappings": {"tag2id": tag2id, "id2tag": id2tag}
    }
    
    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/training_results.json")
    print(f"Final loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    main()