"""Reptile meta-learning algorithm for few-shot malware classification.

This module implements the Reptile algorithm for meta-learning on malware detection
tasks. It supports few-shot learning scenarios with N-way K-shot classification.

The module includes:
- Dataset loader for malware feature files
- Neural network model for malware classification
- Reptile meta-learning algorithm implementation
- Training and validation loops
- Logging and model checkpointing

Example:
    python reptile.py

Expected data structure:
    Data should be organized in features/{label}/{id}.npy format.
    label_split.csv should contain train/val/test splits.
    Each sample is a .npy file containing feature vectors (default 1280-dim).
"""

import os
import random
import json
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Add parent directory to path to import src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extraction.data_loader import (
    create_meta_learning_dataloaders,
    split_episode_to_support_query
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class CFG:
    """Configuration parameters for Reptile meta-learning.

    Attributes:
        n_way: Number of classes in each task (default: 3 for 2 malware + 1 benign).
        k_shot: Number of support samples per class (default: 1).
        q_query: Number of query samples per class (default: 5).
        input_dim: Dimensionality of input features (default: 1280).
        inner_lr: Learning rate for inner loop (task-specific adaptation).
        meta_lr: Learning rate for outer loop (meta-learning update).
        inner_steps: Number of gradient steps in inner loop.
        meta_batch_size: Number of tasks per meta-update.
        max_epoch: Maximum number of training epochs.
        eval_batches: Number of batches for evaluation.
        device: Computation device (cuda/cpu/mps).
        data_json: Path to JSON file containing dataset structure.
        log_dir: Directory for saving logs and model checkpoints.
    """
    n_way = 3
    k_shot = 1
    q_query = 5
    input_dim = 1280

    inner_lr = 0.05
    meta_lr = 0.1
    inner_steps = 5

    meta_batch_size = 8
    max_epoch = 200
    eval_batches = 10
    
    # Training/validation/test configuration
    val_k_shot = 1  # k_shot for validation (same as training)
    test_k_shot = 1  # k_shot for testing (same as training)
    train_episodes_per_epoch = 200
    val_episodes_per_epoch = 60
    test_episodes_per_epoch = 100
    
    # Early stopping and learning rate scheduling
    early_stopping_patience = 15  # Stop training if no improvement for 15 epochs
    lr_decay_factor = 0.5  # Learning rate decay factor
    lr_decay_patience = 10  # Reduce learning rate if no improvement for 10 epochs
    min_lr = 1e-6  # Minimum learning rate
    
    # Device selection with MPS support for MacBook
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # IO configuration
    # Paths relative to project root (parent directory of meta_learning/)
    features_dir = "../MalVis_dataset_small/features"
    split_csv_path = "../MalVis_dataset_small/label_split.csv"
    log_dir = "logs"
    
    # System configuration
    seed = 42
    num_workers = 0
    pin_memory = torch.cuda.is_available()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_label(n_way: int, num_per_class: int) -> torch.Tensor:
    """Create labels for N-way classification task.

    Args:
        n_way: Number of classes.
        num_per_class: Number of samples per class.

    Returns:
        Long tensor of shape (n_way * num_per_class,) containing class labels.
        Example: n_way=3, num_per_class=2 -> [0,0,1,1,2,2]
    """
    return torch.arange(n_way).repeat_interleave(num_per_class).long()


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate classification accuracy.

    Args:
        logits: Model predictions of shape (batch_size, n_classes).
        labels: Ground truth labels of shape (batch_size,).

    Returns:
        Accuracy as a float between 0 and 1.
    """
    return (torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy()).mean()


def calculate_metrics(preds: List[int], labels: List[int],
                     num_classes: int) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics.

    Args:
        preds: List of predicted class indices.
        labels: List of ground truth class indices.
        num_classes: Number of classes.

    Returns:
        Dictionary containing:
            - acc: Overall accuracy
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted F1 score
            - precision: Macro-averaged precision
            - recall: Macro-averaged recall
            - cm: Confusion matrix as nested list
    """
    preds = np.array(preds)
    labels = np.array(labels)
    
    return {
        "acc": (preds == labels).mean(),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "cm": confusion_matrix(labels, preds).tolist(),
    }


def timestamp() -> str:
    """Generate timestamp string.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Logger:
    """Logger for training statistics and model checkpointing.
    
    This class manages JSON logging and best model checkpointing during training.
    It saves epoch-wise metrics and automatically saves the best model based on
    validation accuracy.
    """
    
    def __init__(self):
        """Initialize logger with file paths and empty log structure."""
        os.makedirs(CFG.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.path = os.path.join(CFG.log_dir, f"reptile_{timestamp}.json")
        self.model_path = os.path.join(CFG.log_dir, f"reptile_best_{timestamp}.pth")
        self.experiment_name = f"reptile_experiment_{timestamp}"
        
        # Initialize log structure
        self.logs = {
            "experiment_name": self.experiment_name,
            "config": {
                k: v for k, v in vars(CFG).items()
                if not k.startswith("__") and isinstance(
                    v, (int, float, str, bool, list, type(None))
                )
            },
            "epochs": []
        }
        self.best_val = -1.0
        self.best_epoch = 0

    def add(self, epoch: int, train: Dict[str, Any], val: Dict[str, Any]) -> None:
        """Add epoch statistics to log file.

        Args:
            epoch: Current epoch number.
            train: Dictionary of training metrics.
            val: Dictionary of validation metrics.
        """
        self.logs["epochs"].append({
            "epoch": epoch,
            "train": train,
            "val": val
        })
        
        with open(self.path, "w") as f:
            json.dump(self.logs, f, indent=2)

    def should_save_best(self, val_acc: float) -> bool:
        """Check if current model is the best so far.

        Args:
            val_acc: Validation accuracy.

        Returns:
            True if this is the best model, False otherwise.
        """
        if val_acc > self.best_val:
            self.best_val = val_acc
            return True
        return False


# ============================================================================
# DATASET
# ============================================================================
# Dataset is now handled by src.extraction.data_loader
# Using create_meta_learning_dataloaders to create episode-based dataloaders


# ============================================================================
# MODEL
# ============================================================================

class MalwareNet(nn.Module):
    """Neural network for malware classification.
    
    Architecture:
        - LayerNorm(input_dim) -> Dropout(0.3)
        - Linear(input_dim, hidden) -> ReLU -> Dropout(0.3)
        - Linear(hidden, hidden) -> ReLU -> Dropout(0.3)
        - Linear(hidden, n_way)
    """
    
    def __init__(self, input_dim: int = 1280, hidden: int = 512, n_way: int = 3):
        """Initialize network.

        Args:
            input_dim: Dimensionality of input features.
            hidden: Hidden layer size.
            n_way: Number of output classes.
        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_way),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            Logits tensor of shape (batch_size, n_way).
        """
        # Handle single sample case
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.norm(x)
        return self.net(x)


# ============================================================================
# REPTILE ALGORITHM
# ============================================================================

def reptile_step(model: nn.Module, x: torch.Tensor, loss_fn: nn.Module,
                train: bool = True, k_shot: int = None) -> Tuple[float, float, List[int], List[int], List[float]]:
    """Perform one Reptile meta-learning step on a batch of tasks.

    Reptile algorithm:
    1. Initialize with base parameters θ
    2. For each task:
       a. Sample support and query loader
       b. Adapt on support set: θ' = θ - α∇L_support(θ)
       c. Evaluate on query set and compute meta-gradient
    3. Meta-update: θ = θ + β * mean(θ' - θ) across all tasks

    Args:
        model: PyTorch model with parameters to meta-learn.
        x: Batch of tasks, shape (meta_batch_size, n_way, k+q, feat_dim).
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        train: Whether to update model parameters (True for training).
        k_shot: Number of support samples per class (defaults to CFG.k_shot).

    Returns:
        Tuple of:
            - Mean query loss across all tasks
            - Mean query accuracy across all tasks
            - All predicted labels (flattened across tasks)
            - All true labels (flattened across tasks)
            - All inner step losses (for monitoring adaptation)
    """
    n_way = CFG.n_way
    k = k_shot if k_shot is not None else CFG.k_shot
    q = CFG.q_query
    
    # Store initial parameters (θ)
    theta0 = {n: p.data.clone() for n, p in model.named_parameters()}
    
    # Accumulate meta-gradient
    meta_delta = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    task_losses = []
    task_accs = []
    all_preds = []
    all_labels = []
    all_inner_losses = []  # Track inner loop losses for monitoring

    for task in x:
        # Reset to initial parameters for each task
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(theta0[n])

        # Split into support and query sets using helper function
        # task shape: [n_way, k+q, feat_dim]
        episode = task.squeeze(0) if task.dim() == 4 else task
        support, query, y_s, y_q = split_episode_to_support_query(
            episode, k_shot=k, q_query=q
        )
        support = support.to(CFG.device)
        query = query.to(CFG.device)
        y_s = y_s.to(CFG.device)
        y_q = y_q.to(CFG.device)

        # Inner loop: adapt on support set (skip if k_shot=0)
        inner_losses = []  # Track losses for this task's inner loop
        if k > 0:
            model.train()
            for _ in range(CFG.inner_steps):
                out = model(support)
                loss = loss_fn(out, y_s)
                inner_losses.append(loss.item())  # Track inner step loss
                model.zero_grad()
                loss.backward()
                
                # Gradient step
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.data -= CFG.inner_lr * p.grad
        
        # Store inner losses for monitoring
        all_inner_losses.extend(inner_losses)

        # Store adapted parameters (θ')
        adapted = {n: p.data.clone() for n, p in model.named_parameters()}

        # Evaluate on query set
        model.eval()
        with torch.no_grad():
            q_out = model(query)
            q_loss = loss_fn(q_out, y_q)
            acc = calculate_accuracy(q_out, y_q)
            preds = torch.argmax(q_out, -1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y_q.cpu().numpy())

        task_losses.append(q_loss.item())
        task_accs.append(acc)
        
        # Accumulate meta-gradient (θ' - θ)
        with torch.no_grad():
            for n, p in model.named_parameters():
                meta_delta[n] += adapted[n] - theta0[n]

    # Outer loop: meta-update
    if train:
        with torch.no_grad():
            for n, p in model.named_parameters():
                # θ = θ + β * mean(θ' - θ)
                p.data = theta0[n] + CFG.meta_lr * meta_delta[n] / len(x)

    return np.mean(task_losses), np.mean(task_accs), all_preds, all_labels, all_inner_losses


# ============================================================================
# TRAINING
# ============================================================================

def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             train: bool, k_shot: int = None) -> Dict[str, Any]:
    """Run one training or validation epoch.

    Args:
        model: Model to train/evaluate.
        loader: DataLoader for the dataset.
        loss_fn: Loss function.
        train: Whether to train the model (True) or evaluate (False).
        k_shot: Number of support samples per class (defaults to CFG.k_shot).

    Returns:
        Dictionary of metrics (loss, acc, f1_macro, precision, recall, cm, inner_loss).
    """
    losses = []
    preds = []
    labels = []
    inner_losses = []  # Track inner loop losses
    
    # Determine number of batches
    if train:
        num_batches = CFG.train_episodes_per_epoch // CFG.meta_batch_size
    else:
        num_batches = CFG.eval_batches
    
    k = k_shot if k_shot is not None else CFG.k_shot

    # Collect meta-batch of tasks
    meta_batch = []
    iterator = iter(loader)
    
    for _ in tqdm(range(num_batches), desc="Train" if train else "Val"):
        # Collect meta_batch_size tasks
        meta_batch = []
        for _ in range(CFG.meta_batch_size):
            try:
                episode_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                episode_batch = next(iterator)
            
            # episode_batch shape: [1, n_way, k+q, feat_dim]
            # Remove batch dimension and move to device
            episode = episode_batch.squeeze(0).to(CFG.device)
            meta_batch.append(episode)
        
        # Stack to form meta-batch: [meta_batch_size, n_way, k+q, feat_dim]
        x = torch.stack(meta_batch)
        
        # Perform Reptile step
        loss, acc, p, y, inner_loss = reptile_step(model, x, loss_fn, train=train, k_shot=k)
        
        losses.append(loss)
        preds.extend(p)
        labels.extend(y)
        inner_losses.extend(inner_loss)  # Collect inner loop losses
    
    # Compute overall metrics
    metrics = calculate_metrics(preds, labels, CFG.n_way)
    metrics["loss"] = float(np.mean(losses))
    metrics["inner_loss"] = float(np.mean(inner_losses)) if len(inner_losses) > 0 else 0.0
    
    return metrics


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main training loop for Reptile meta-learning."""
    print(f"[Reptile] device={CFG.device}  "
          f"n_way={CFG.n_way}  k_shot={CFG.k_shot}  q_query={CFG.q_query}")
    
    # Initialize model and loss
    model = MalwareNet(CFG.input_dim, hidden=512, n_way=CFG.n_way).to(CFG.device)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    # Create data loaders using new meta-learning dataloader
    # Convert relative paths to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.abspath(os.path.join(script_dir, CFG.features_dir))
    split_csv_path = os.path.abspath(os.path.join(script_dir, CFG.split_csv_path))
    
    print(f"\n[DataLoader] Loading from: {features_dir}")
    print(f"[DataLoader] Split CSV: {split_csv_path}")
    
    # Get configuration
    val_k_shot = getattr(CFG, 'val_k_shot', CFG.k_shot)
    test_k_shot = getattr(CFG, 'test_k_shot', CFG.k_shot)
    
    print(f"\n[Config] Training/Validation/Test Setup:")
    print(f"  Task: {CFG.n_way}-way, {CFG.k_shot}-shot, {CFG.q_query}-query")
    print(f"  Train: {CFG.k_shot}-shot")
    print(f"  Val:   {val_k_shot}-shot")
    print(f"  Test:  {test_k_shot}-shot")
    
    # Create training and validation dataloaders
    dataloaders_train = create_meta_learning_dataloaders(
        features_dir=features_dir,
        split_csv_path=split_csv_path,
        n_way=CFG.n_way,
        k_shot=CFG.k_shot,
        q_query=CFG.q_query,
        train_episodes_per_epoch=CFG.train_episodes_per_epoch,
        val_episodes_per_epoch=CFG.val_episodes_per_epoch,
        test_episodes_per_epoch=CFG.test_episodes_per_epoch,
        normalize=True,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        seed=CFG.seed
    )
    
    train_loader = dataloaders_train['train']
    
    # Create validation dataloader (use val_k_shot if different)
    if val_k_shot != CFG.k_shot:
        dataloaders_val = create_meta_learning_dataloaders(
            features_dir=features_dir,
            split_csv_path=split_csv_path,
            n_way=CFG.n_way,
            k_shot=val_k_shot,
            q_query=CFG.q_query,
            train_episodes_per_epoch=CFG.train_episodes_per_epoch,
            val_episodes_per_epoch=CFG.val_episodes_per_epoch,
            test_episodes_per_epoch=CFG.test_episodes_per_epoch,
            normalize=True,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
            seed=CFG.seed
        )
        val_loader = dataloaders_val['val']
        print(f"[DataLoader] Validation uses separate {val_k_shot}-shot dataloader")
    else:
        val_loader = dataloaders_train['val']
        print(f"[DataLoader] Validation uses same {CFG.k_shot}-shot dataloader as training")
    
    print(f"\n[DataLoader] Created dataloaders:")
    print(f"  Train: {CFG.k_shot}-shot, {CFG.train_episodes_per_epoch} episodes/epoch")
    print(f"  Val:   {val_k_shot}-shot, {CFG.val_episodes_per_epoch} episodes/epoch")

    # Training loop with early stopping and learning rate decay
    best_val = -1.0
    save_path = os.path.join(CFG.log_dir, f"reptile_best_{timestamp()}.pth")
    
    # Early stopping and learning rate scheduling variables
    patience_counter = 0
    lr_patience_counter = 0
    best_epoch = 0
    current_lr = CFG.meta_lr

    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Train and validate
        train_metrics = run_epoch(model, train_loader, loss_fn, train=True, k_shot=CFG.k_shot)
        val_metrics = run_epoch(model, val_loader, loss_fn, train=False, k_shot=val_k_shot)

        # Print progress
        print(f"Train: acc={train_metrics['acc']*100:.2f}%  "
              f"loss={train_metrics['loss']:.4f}  "
              f"inner_loss={train_metrics.get('inner_loss', 0.0):.4f}")
        print(f"Val  : acc={val_metrics['acc']*100:.2f}%  "
              f"loss={val_metrics['loss']:.4f}")
        
        # Log metrics
        logger.add(epoch, train_metrics, val_metrics)
        
        # Check if validation improved
        improved = False
        if val_metrics['acc'] > best_val:
            best_val = val_metrics['acc']
            best_epoch = epoch
            patience_counter = 0
            lr_patience_counter = 0
            improved = True
            
            cfg_dict = {
                k: v for k, v in vars(CFG).items()
                if not k.startswith("__") and isinstance(
                    v, (int, float, str, bool, list, type(None))
                )
            }
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg_dict,
                "epoch": epoch,
                "val_acc": best_val,
            }, save_path)
            
            print(f"✓ Saved best model (epoch {epoch}, val_acc={best_val*100:.2f}%)")
        else:
            patience_counter += 1
            lr_patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CFG.early_stopping_patience})")
        
        # Learning rate decay (using meta_lr as base)
        # Note: Reptile doesn't use an optimizer, so we adjust CFG.meta_lr directly
        if lr_patience_counter >= CFG.lr_decay_patience:
            current_lr = max(current_lr * CFG.lr_decay_factor, CFG.min_lr)
            CFG.meta_lr = current_lr  # Update the learning rate used in reptile_step
            lr_patience_counter = 0
            print(f"  ⚠ Learning rate decayed to: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= CFG.early_stopping_patience:
            print(f"\n⚠ Early stopping triggered! No improvement for {CFG.early_stopping_patience} epochs.")
            print(f"Best validation accuracy: {best_val*100:.2f}% at epoch {best_epoch}")
            break

    # Training complete
    print(f"\n✅ Training done. Logs saved at: {logger.path}")
    print(f"✅ Best model saved at: {save_path}")


if __name__ == "__main__":
    main()
