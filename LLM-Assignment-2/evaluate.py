import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

from data_loader import create_dataloader, get_tag_mappings
from model_baseline import create_model
import preprocess

def get_spans(pred_seq, id2tag):
    """
    Convert predicted tag sequence to spans
    
    Args:
        pred_seq: List of predicted tag IDs
        id2tag: Mapping from ID to tag string
        
    Returns:
        List of (start, end, label) tuples
    """
    spans = []
    start = None
    label = None
    
    for i, tag_id in enumerate(pred_seq):
        tag = id2tag[tag_id]
        
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start = i
            label = tag[2:]  # Remove "B-" prefix
        elif tag.startswith("I-") and start is not None:
            # Continue with current span
            continue
        else:
            if start is not None:
                spans.append((start, i, label))
                start = None
                label = None
    
    # Add last span if exists
    if start is not None:
        spans.append((start, len(pred_seq), label))
    
    return spans


def overlap_f1_per_class(pred_spans, true_spans, marker_types=None):
    """
    Calculate overlap F1 scores per class
    
    Args:
        pred_spans: List of predicted spans for each example
        true_spans: List of true spans for each example
        marker_types: List of marker types to evaluate
        
    Returns:
        Dictionary with per-class metrics
    """
    if marker_types is None:
        marker_types = ['Actor', 'Action', 'Effect', 'Victim', 'Evidence']
    
    results = {}
    
    for m_type in marker_types:
        TP = FP = FN = 0
        
        for p, t in zip(pred_spans, true_spans):
            matched = set()
            
            # Check predicted spans of this type
            for ps in p:
                if ps[2] == m_type:
                    found = False
                    for i, ts in enumerate(t):
                        if ts[2] == m_type:
                            # Check for overlap
                            if not (ps[1] <= ts[0] or ps[0] >= ts[1]):
                                TP += 1
                                matched.add(i)
                                found = True
                                break
                    if not found:
                        FP += 1
            
            # Count false negatives for this type
            for i, ts in enumerate(t):
                if ts[2] == m_type and i not in matched:
                    FN += 1
        
        # Calculate metrics
        prec = TP / (TP + FP + 1e-8)
        rec = TP / (TP + FN + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0
        
        results[m_type] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': TP,
            'fp': FP,
            'fn': FN,
            'support': TP + FN
        }
    
    return results


def compute_macro_weighted_f1(per_class_results):
    """
    Compute macro and weighted F1 scores
    
    Args:
        per_class_results: Dictionary with per-class metrics
        
    Returns:
        macro_f1, weighted_f1
    """
    classes = list(per_class_results.keys())
    
    # Collect F1 scores and supports
    f1_scores = []
    supports = []
    
    for cls in classes:
        f1_scores.append(per_class_results[cls]["f1"])
        supports.append(per_class_results[cls]["support"])
    
    # Macro F1
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    # Weighted F1
    total_support = sum(supports)
    if total_support == 0:
        weighted_f1 = 0
    else:
        weighted_f1 = sum(f1 * s for f1, s in zip(f1_scores, supports)) / total_support
    
    return macro_f1, weighted_f1


def plot_per_class_metrics(per_class_results, save_path=None):
    """
    Plot per-class F1 scores and precision/recall
    
    Args:
        per_class_results: Dictionary with per-class metrics
        save_path: Path to save the plot
    """
    marker_types = list(per_class_results.keys())
    
    # Prepare data for plotting
    precisions = [per_class_results[m]['precision'] for m in marker_types]
    recalls = [per_class_results[m]['recall'] for m in marker_types]
    f1_scores = [per_class_results[m]['f1'] for m in marker_types]
    
    x = np.arange(len(marker_types))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Bar chart of F1 scores
    bars = ax1.bar(marker_types, f1_scores, color='skyblue', edgecolor='black')
    ax1.set_title("Per-Class Overlap F1 Scores", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Marker Type", fontsize=12)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{f1_scores[i]:.3f}', ha='center', fontweight='bold')
    
    # Plot 2: Precision and Recall comparison
    ax2.bar(x - width, precisions, width, label='Precision', color='lightcoral')
    ax2.bar(x, recalls, width, label='Recall', color='lightgreen')
    ax2.bar(x + width, f1_scores, width, label='F1', color='skyblue')
    
    ax2.set_xlabel("Marker Type", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Precision, Recall, and F1 by Class", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(marker_types)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    
    plt.show()


def evaluate_model(model, dataloader, id2tag, device):
    """
    Evaluate model on given dataloader
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        id2tag: Mapping from ID to tag
        device: Device to use
        
    Returns:
        Dictionary with all predictions and metrics
    """
    model.eval()
    all_preds_spans = []
    all_labels_spans = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]
            
            # Get predictions
            preds = model(input_ids).argmax(-1).cpu()
            
            # Process each example in batch
            for i in range(preds.shape[0]):
                # Get actual length (excluding padding)
                length = (batch["attention_mask"][i] != 0).sum().item()
                
                # Get sequences
                pred_seq = preds[i].tolist()[:length]
                label_seq = labels[i].tolist()[:length]
                
                # Convert to spans
                all_preds_spans.append(get_spans(pred_seq, id2tag))
                all_labels_spans.append(get_spans(label_seq, id2tag))
    
    return all_preds_spans, all_labels_spans


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    data = preprocess.load_and_preprocess_data("train_rehydrated.jsonl")
    
    # Load training results for tag mappings
    print("\n2. Loading tag mappings...")
    with open("results/training_results.json", "r") as f:
        training_results = json.load(f)
    
    tag2id = training_results["tag_mappings"]["tag2id"]
    id2tag = {int(k): v for k, v in training_results["tag_mappings"]["id2tag"].items()}
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create data loader
    print("\n3. Creating data loader...")
    eval_loader = create_dataloader(
        data, tokenizer, tag2id,
        batch_size=16,
        shuffle=False,
        max_len=128
    )
    
    # Load model
    print("\n4. Loading model...")
    model = create_model(
        vocab_size=len(tokenizer),
        device=device,
        num_labels=len(tag2id)
    )
    
    # Load checkpoint
    checkpoint_path = "checkpoints/model_final.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Model loaded from {checkpoint_path}")
        print(f"  Trained for {checkpoint['epoch']} epochs")
        print(f"  Final loss: {checkpoint['loss']:.4f}")
    else:
        print(f"  Warning: Checkpoint not found at {checkpoint_path}")
        return
    
    # Evaluate
    print("\n5. Evaluating model...")
    all_preds_spans, all_labels_spans = evaluate_model(
        model, eval_loader, id2tag, device
    )
    
    print(f"  Evaluated {len(all_preds_spans)} examples")
    
    # Calculate metrics
    print("\n6. Calculating metrics...")
    marker_types = ['Actor', 'Action', 'Effect', 'Victim', 'Evidence']
    per_class_results = overlap_f1_per_class(
        all_preds_spans, all_labels_spans, marker_types
    )
    
    macro_f1, weighted_f1 = compute_macro_weighted_f1(per_class_results)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nPer-Class Metrics:")
    print("-" * 40)
    print(f"{'Marker':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 40)
    
    for marker in marker_types:
        metrics = per_class_results[marker]
        print(f"{marker:<10} {metrics['precision']:.3f}      "
              f"{metrics['recall']:.3f}      "
              f"{metrics['f1']:.3f}      "
              f"{metrics['support']:<10}")
    
    print("-" * 40)
    print(f"\nMacro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Save results
    results = {
        "per_class_metrics": per_class_results,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_examples": len(all_preds_spans)
    }
    
    results_path = "results/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Create plots
    print("\n7. Generating plots...")
    plot_path = "plots/per_class_metrics.png"
    plot_per_class_metrics(per_class_results, plot_path)
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()