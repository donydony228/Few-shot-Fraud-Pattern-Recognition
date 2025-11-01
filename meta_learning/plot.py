#!/usr/bin/env python3
"""
Script to visualize Reptile training logs.

Usage:
    python plot.py logs/reptile_experiment_20240101_120000.json
    python plot.py  # Uses the most recent log file
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from datetime import datetime

def load_log(log_path):
    """Load training log from JSON file"""
    with open(log_path, 'r') as f:
        return json.load(f)

def find_latest_log(log_dir='logs'):
    """Find the most recent log file"""
    log_files = list(Path(log_dir).glob('*.json'))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")
    
    latest = max(log_files, key=lambda p: p.stat().st_mtime)
    return str(latest)

def plot_training_curves(log_data, save_dir='plots'):
    """
    Plot comprehensive training curves (2x2 layout).
    
    Plots:
    1. Loss curves (Train vs Validation)
    2. Accuracy curves (Train vs Validation) 
    3. F1 Score curves (Train vs Validation)
    4. Precision and Recall curves (Train vs Validation)
    
    Args:
        log_data: Dictionary containing training log
        save_dir: Directory to save plots
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = [e['epoch'] for e in log_data['epochs']]
    
    # Extract metrics with better error handling
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    train_f1 = []
    val_f1 = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    
    for e in log_data['epochs']:
        # Loss
        train_loss.append(e['train'].get('loss', 0))
        val_loss.append(e['val'].get('loss', 0))
        
        # Accuracy - 檢查多種可能的key並處理百分比轉換
        train_acc_val = e['train'].get('acc', e['train'].get('accuracy', 0))
        val_acc_val = e['val'].get('acc', e['val'].get('accuracy', 0))
        
        # 如果值已經是百分比形式（>1），直接使用；否則轉換為百分比
        if train_acc_val > 1:
            train_acc.append(train_acc_val)
        else:
            train_acc.append(train_acc_val * 100)
            
        if val_acc_val > 1:
            val_acc.append(val_acc_val)
        else:
            val_acc.append(val_acc_val * 100)
        
        # F1 Score - 使用默認值0如果不存在
        train_f1.append(e['train'].get('f1_macro', e['train'].get('f1', 0)))
        val_f1.append(e['val'].get('f1_macro', e['val'].get('f1', 0)))
        
        # Precision
        train_precision.append(e['train'].get('precision_macro', e['train'].get('precision', 0)))
        val_precision.append(e['val'].get('precision_macro', e['val'].get('precision', 0)))
        
        # Recall
        train_recall.append(e['train'].get('recall_macro', e['train'].get('recall', 0)))
        val_recall.append(e['val'].get('recall_macro', e['val'].get('recall', 0)))
    
    # 檢查數據有效性並提供警告
    if all(f == 0 for f in train_f1 + val_f1):
        print("⚠️  Warning: No F1 scores found in data. F1 plot will show zeros.")
    if all(p == 0 for p in train_precision + val_precision):
        print("⚠️  Warning: No precision scores found in data. Precision plot will show zeros.")
    if all(r == 0 for r in train_recall + val_recall):
        print("⚠️  Warning: No recall scores found in data. Recall plot will show zeros.")
    
    # Create figure with subplots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    exp_name = log_data.get('experiment_name', 'Training Experiment')
    fig.suptitle(f"Training Results: {exp_name}", fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax.axhline(y=np.log(3), color='gray', linestyle='--', label='Random (log(3)≈1.099)', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2, marker='s', markersize=4)
    ax.axhline(y=33.33, color='gray', linestyle='--', label='Random (33.33%)', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 3: F1 Score
    ax = axes[1, 0]
    if any(f > 0 for f in train_f1 + val_f1):  # 只有在有非零數據時才繪製
        ax.plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2, marker='s', markersize=4)
    else:
        ax.plot(epochs, [0] * len(epochs), 'b-', label='Train F1 (No data)', linewidth=2, alpha=0.5)
        ax.plot(epochs, [0] * len(epochs), 'r-', label='Val F1 (No data)', linewidth=2, alpha=0.5)
    
    ax.axhline(y=0.333, color='gray', linestyle='--', label='Random (≈0.333)', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('F1 Score (Macro Average)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: Precision vs Recall
    ax = axes[1, 1]
    if any(p > 0 for p in train_precision + val_precision + train_recall + val_recall):
        ax.plot(epochs, train_precision, 'b-', label='Train Precision', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, val_precision, 'r-', label='Val Precision', linewidth=2, marker='s', markersize=4)
        ax.plot(epochs, train_recall, 'b--', label='Train Recall', linewidth=2, marker='^', markersize=4)
        ax.plot(epochs, val_recall, 'r--', label='Val Recall', linewidth=2, marker='v', markersize=4)
    else:
        ax.plot(epochs, [0] * len(epochs), 'b-', label='Train Precision (No data)', linewidth=2, alpha=0.5)
        ax.plot(epochs, [0] * len(epochs), 'r-', label='Val Precision (No data)', linewidth=2, alpha=0.5)
        ax.plot(epochs, [0] * len(epochs), 'b--', label='Train Recall (No data)', linewidth=2, alpha=0.5)
        ax.plot(epochs, [0] * len(epochs), 'r--', label='Val Recall (No data)', linewidth=2, alpha=0.5)
    
    ax.axhline(y=0.333, color='gray', linestyle='--', label='Random (≈0.333)', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision and Recall (Macro Average)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    exp_name = log_data.get('experiment_name', 'Training Experiment')
    save_path = os.path.join(save_dir, f'{exp_name}_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved training curves to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(log_data, save_dir='plots', epoch=-1):
    """
    Plot confusion matrix for a specific epoch.
    
    Args:
        log_data: Dictionary containing training log
        save_dir: Directory to save plots
        epoch: Epoch to plot (-1 for last epoch)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epoch_data = log_data['epochs'][epoch]
    # Check if confusion matrix exists in validation data
    cm_data = epoch_data['val'].get('cm', epoch_data['val'].get('confusion_matrix', None))
    
    if cm_data is None:
        print("⚠️  No confusion matrix data found. Skipping confusion matrix plot.")
        return
    
    cm = np.array(cm_data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    class_names = ['Malware Type 1', 'Malware Type 2', 'Benign']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title=f'Confusion Matrix (Epoch {epoch_data["epoch"]})')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    exp_name = log_data.get('experiment_name', 'Training Experiment')
    save_path = os.path.join(save_dir, f'{exp_name}_confusion_matrix_epoch{epoch_data["epoch"]}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved confusion matrix to: {save_path}")
    
    plt.show()

def plot_per_class_metrics(log_data, save_dir='plots'):
    """Plot per-class F1 scores over epochs"""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = [e['epoch'] for e in log_data['epochs']]
    
    # Extract per-class F1 scores
    class_names = ['Malware Type 1', 'Malware Type 2', 'Benign']
    colors = ['red', 'orange', 'green']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for class_idx in range(3):
        train_f1_class = [e['train'].get('f1_per_class', [0, 0, 0])[class_idx] for e in log_data['epochs']]
        val_f1_class = [e['val'].get('f1_per_class', [0, 0, 0])[class_idx] for e in log_data['epochs']]
        
        ax.plot(epochs, train_f1_class, '--', color=colors[class_idx], 
                label=f'{class_names[class_idx]} (Train)', linewidth=1.5, alpha=0.7)
        ax.plot(epochs, val_f1_class, '-', color=colors[class_idx], 
                label=f'{class_names[class_idx]} (Val)', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    exp_name = log_data.get('experiment_name', 'Training Experiment')
    save_path = os.path.join(save_dir, f'{exp_name}_per_class_f1.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved per-class F1 scores to: {save_path}")
    
    plt.show()

def print_summary(log_data):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Hyperparameters
    print("\n Hyperparameters:")
    hyperparams = log_data.get('hyperparameters', log_data.get('config', {}))
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Final epoch results
    print("\n Final Epoch Results:")
    last_epoch = log_data['epochs'][-1]
    
    print(f"\n  Training:")
    print(f"    Loss: {last_epoch['train'].get('loss', 0):.4f}")
    print(f"    Accuracy: {last_epoch['train'].get('acc', last_epoch['train'].get('accuracy', 0))*100:.2f}%")
    print(f"    F1-Macro: {last_epoch['train'].get('f1_macro', 0):.4f}")
    print(f"    Precision: {last_epoch['train'].get('precision', last_epoch['train'].get('precision_macro', 0)):.4f}")
    print(f"    Recall: {last_epoch['train'].get('recall', last_epoch['train'].get('recall_macro', 0)):.4f}")
    
    print(f"\n  Validation:")
    print(f"    Loss: {last_epoch['val'].get('loss', 0):.4f}")
    print(f"    Accuracy: {last_epoch['val'].get('acc', last_epoch['val'].get('accuracy', 0))*100:.2f}%")
    print(f"    F1-Macro: {last_epoch['val'].get('f1_macro', 0):.4f}")
    print(f"    Precision: {last_epoch['val'].get('precision', last_epoch['val'].get('precision_macro', 0)):.4f}")
    print(f"    Recall: {last_epoch['val'].get('recall', last_epoch['val'].get('recall_macro', 0)):.4f}")
    
    # Best epoch
    print("\n Best Validation Accuracy:")
    val_accs = [e['val'].get('acc', e['val'].get('accuracy', 0)) for e in log_data['epochs']]
    best_epoch_idx = np.argmax(val_accs)
    best_epoch = log_data['epochs'][best_epoch_idx]
    print(f"  Epoch: {best_epoch['epoch']}")
    print(f"  Accuracy: {best_epoch['val'].get('acc', best_epoch['val'].get('accuracy', 0))*100:.2f}%")
    print(f"  F1-Macro: {best_epoch['val'].get('f1_macro', 0):.4f}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Visualize Reptile training logs')
    parser.add_argument('log_file', nargs='?', default=None,
                       help='Path to log JSON file (default: most recent in logs/)')
    parser.add_argument('--save-dir', default='plots',
                       help='Directory to save plots (default: plots/)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    # Find log file
    if args.log_file is None:
        try:
            log_path = find_latest_log()
            print(f" Using most recent log: {log_path}")
        except FileNotFoundError as e:
            print(f" Error: {e}")
            return
    else:
        log_path = args.log_file
        if not os.path.exists(log_path):
            print(f" Error: Log file not found: {log_path}")
            return
    
    # Load log
    print(f" Loading log from: {log_path}")
    log_data = load_log(log_path)
    
    # Print summary
    print_summary(log_data)
    
    # Generate plots
    print(f"\n Generating plots...")
    
    if args.no_show:
        plt.ioff()  # Turn off interactive mode
    
    plot_training_curves(log_data, args.save_dir)
    plot_confusion_matrix(log_data, args.save_dir)
    plot_per_class_metrics(log_data, args.save_dir)
    
    print(f"\n All plots saved to: {args.save_dir}/")
    print("\n Tips:")
    print("   - Check if accuracy is improving above 33.33% (random baseline)")
    print("   - Look for consistent improvement in F1 scores")
    print("   - Monitor precision-recall balance")

if __name__ == '__main__':
    main()