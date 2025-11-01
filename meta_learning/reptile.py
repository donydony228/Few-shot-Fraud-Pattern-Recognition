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
    Data should be organized in JSON format with train/val/test splits.
    Each sample is a .npy file containing 1280-dimensional feature vectors.
"""

import os
import random
import json
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

# Import helper functions from data_loader (without modifying data_loader.py)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.extraction.data_loader import get_label_splits, collect_file_paths


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
    
    # Device selection with MPS support for MacBook
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    data_json = "../malware_data_structure.json"  # For JSON-based loading
    features_dir = "../MalVis_dataset_small/features"  # For CSV-based loading
    split_csv_path = "../MalVis_dataset_small/label_split.csv"  # For CSV-based loading
    use_csv_loader = True  # Set to True to use data_loader.py, False to use JSON
    log_dir = "logs"


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

class CSVBasedMalwareDataset(Dataset):
    """Dataset class for meta-learning using CSV label splits.
    
    This dataset uses helper functions from data_loader.py to load data
    based on CSV label splits and dynamically generates few-shot tasks.
    """
    
    def __init__(
        self,
        features_dir: str,
        split_csv_path: str,
        split: str,
        k_shot: int,
        q_query: int,
        n_way: int = 3,
        input_dim: int = 1280,
        seed: int = 42,
        normal_label: str = "benign"
    ):
        """Initialize CSV-based dataset.
        
        Args:
            features_dir: Path to features directory
            split_csv_path: Path to label_split.csv
            split: 'train', 'val', or 'test'
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
            n_way: Number of classes per task
            input_dim: Feature dimensionality
            seed: Random seed
            normal_label: Name of normal/benign class
        """
        # Use helper functions from data_loader.py
        seen_labels, unseen_labels = get_label_splits(split_csv_path)
        
        # Determine which labels to use based on split
        if split == "train" or split == "val":
            self.labels = seen_labels
        else:  # test
            self.labels = unseen_labels
        
        # Collect file paths for each label using helper function
        all_files = collect_file_paths(features_dir, self.labels)
        
        # Organize files by label
        self.label_to_files = {}
        for filepath in all_files:
            label = os.path.basename(os.path.dirname(filepath))
            if label not in self.label_to_files:
                self.label_to_files[label] = []
            self.label_to_files[label].append(filepath)
        
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_way = n_way
        self.input_dim = input_dim
        self.seed = seed
        self.normal_label = normal_label
        self.classes = list(self.label_to_files.keys())
        
        if len(self.classes) < n_way:
            raise ValueError(
                f"Not enough classes ({len(self.classes)}) for {n_way}-way tasks. "
                f"Available: {self.classes}"
            )
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a single few-shot task."""
        np.random.seed(self.seed + idx)
        
        # Select classes for this task
        frauds = [c for c in self.classes if c != self.normal_label]
        if len(frauds) < (self.n_way - 1):
            raise ValueError(
                f"Not enough non-normal classes ({len(frauds)}) for {self.n_way}-way task"
            )
        
        selected_frauds = np.random.choice(frauds, self.n_way - 1, replace=False).tolist()
        
        if self.normal_label not in self.classes:
            raise ValueError(f"Normal class '{self.normal_label}' not found")
        
        selected = selected_frauds + [self.normal_label]
        task = []
        need = self.k_shot + self.q_query
        
        for cls in selected:
            files = self.label_to_files[cls]
            chosen = np.random.choice(files, need, replace=(len(files) < need))
            
            cls_features = []
            for filepath in chosen:
                try:
                    arr = np.load(filepath)
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    
                    # Ensure correct dimension
                    if arr.shape[0] != self.input_dim:
                        if arr.shape[0] < self.input_dim:
                            arr = np.pad(arr, (0, self.input_dim - arr.shape[0]))
                        else:
                            arr = arr[:self.input_dim]
                    
                    # Z-score normalization
                    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
                except Exception:
                    arr = np.zeros(self.input_dim, dtype=np.float32)
                
                cls_features.append(arr.astype(np.float32))
            
            task.append(torch.tensor(np.stack(cls_features), dtype=torch.float32))
        
        return torch.stack(task)  # Shape: [n_way, k+q, feat_dim]
    
    def __len__(self) -> int:
        return 100000


class MalwareDataset(Dataset):
    """Dataset class for malware feature vectors with few-shot task generation.

    This dataset loads malware features from JSON structure and dynamically
    generates few-shot tasks. Each task contains:
    - 2 randomly selected malware families (abnormal classes)
    - 1 benign class (normal class)
    - Total of 3 classes (n_way=3)

    Features are loaded from .npy files and normalized using Z-score normalization.
    """
    
    def __init__(self, json_path: str, split: str, k_shot: int, q_query: int):
        """Initialize dataset.

        Args:
            json_path: Path to JSON file containing dataset structure.
            split: Data split ('train', 'val', or 'test').
            k_shot: Number of support samples per class.
            q_query: Number of query samples per class.
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)[split]
        
        self.classes = list(self.data.keys())
        self.k_shot = k_shot
        self.q_query = q_query
        self.normal = "benign"

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a single few-shot task.

        Args:
            idx: Task index.

        Returns:
            Tensor of shape (n_way, k_shot + q_query, input_dim) containing
            features for one task. The order is: [malware1, malware2, benign].
        """
        # Set seed for reproducibility based on task index
        np.random.seed(42 + idx)
        
        # Select 2 malware families randomly
        frauds = [c for c in self.classes if c != self.normal]
        selected = np.random.choice(frauds, 2, replace=False).tolist() + [self.normal]

        task = []
        for cls in selected:
            # Get file list for this class
            files = self.data[cls]
            need = self.k_shot + self.q_query
            
            # Sample files (with replacement if necessary)
            chosen = np.random.choice(files, need, replace=(len(files) < need))
            
            cls_features = []
            for f in chosen:
                # Try to load and normalize features
                f = self._fix_path(f)
                try:
                    arr = np.load(f)
                    arr = arr.flatten() if arr.ndim > 1 else arr
                    # Z-score normalization per sample
                    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
                except Exception:
                    # Fallback to zero vector if loading fails
                    arr = np.zeros(CFG.input_dim)
                
                cls_features.append(arr)
            
            task.append(torch.tensor(np.stack(cls_features), dtype=torch.float32))
        
        return torch.stack(task)  # Shape: [n_way, k+q, feat_dim]

    def _fix_path(self, path: str) -> str:
        """Attempt to fix file path by trying common prefixes.

        Args:
            path: Original file path.

        Returns:
            Corrected absolute path if file exists, original path otherwise.
        """
        if os.path.exists(path):
            return path
        
        # Try common path prefixes
        for prefix in ["../", "../../", "./"]:
            candidate = os.path.join(prefix, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        
        return path

    def __len__(self) -> int:
        """Return dataset length (virtual, used by DataLoader).

        Returns:
            Fixed length of 200 (actual number of tasks is controlled by
            training loop iterations).
        """
        return 200


def get_meta_batch(loader: DataLoader, iterator: iter) -> Tuple[torch.Tensor, iter]:
    """Collect a meta-batch of tasks.

    Args:
        loader: DataLoader for the dataset.
        iterator: Current iterator over the loader.

    Returns:
        Tuple of (batch_tensor, updated_iterator).
        batch_tensor has shape (meta_batch_size, n_way, k+q, feat_dim).
    """
    batch = []
    for _ in range(CFG.meta_batch_size):
        try:
            task = next(iterator)
        except StopIteration:
            # Reset iterator if exhausted
            iterator = iter(loader)
            task = next(iterator)
        
        batch.append(task.squeeze(0).to(CFG.device))
    
    return torch.stack(batch), iterator


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
                train: bool = True) -> Tuple[float, float, List[int], List[int]]:
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

    Returns:
        Tuple of:
            - Mean query loss across all tasks
            - Mean query accuracy across all tasks
            - All predicted labels (flattened across tasks)
            - All true labels (flattened across tasks)
    """
    n_way, k, q = CFG.n_way, CFG.k_shot, CFG.q_query
    
    # Store initial parameters (θ)
    theta0 = {n: p.data.clone() for n, p in model.named_parameters()}
    
    # Accumulate meta-gradient
    meta_delta = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    task_losses = []
    task_accs = []
    all_preds = []
    all_labels = []

    for task in x:
        # Reset to initial parameters for each task
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(theta0[n])

        # Split into support and query sets
        support = task[:, :k, :].reshape(n_way * k, -1)
        query = task[:, k:, :].reshape(n_way * q, -1)
        
        # Create labels
        y_s = create_label(n_way, k).to(CFG.device)
        y_q = create_label(n_way, q).to(CFG.device)

        # Inner loop: adapt on support set
        model.train()
        for _ in range(CFG.inner_steps):
            out = model(support)
            loss = loss_fn(out, y_s)
            model.zero_grad()
            loss.backward()
            
            # Gradient step
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= CFG.inner_lr * p.grad

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

    return np.mean(task_losses), np.mean(task_accs), all_preds, all_labels


# ============================================================================
# TRAINING
# ============================================================================

def run_epoch(model: nn.Module, loader: DataLoader, iterator: iter,
             loss_fn: nn.Module, train: bool) -> Dict[str, Any]:
    """Run one training or validation epoch.

    Args:
        model: Model to train/evaluate.
        loader: DataLoader for the dataset.
        iterator: Iterator over the loader.
        loss_fn: Loss function.
        train: Whether to train the model (True) or evaluate (False).

    Returns:
        Dictionary of metrics (loss, acc, f1_macro, precision, recall, cm).
    """
    losses = []
    preds = []
    labels = []
    
    # Determine number of batches
    if train:
        num_batches = len(loader) // CFG.meta_batch_size
    else:
        num_batches = CFG.eval_batches

    for _ in tqdm(range(num_batches), desc="Train" if train else "Val"):
        # Get meta-batch of tasks
        x, iterator = get_meta_batch(loader, iterator)
        
        # Perform Reptile step
        loss, acc, p, y = reptile_step(model, x, loss_fn, train=train)
        
        losses.append(loss)
        preds.extend(p)
        labels.extend(y)
    
    # Compute overall metrics
    metrics = calculate_metrics(preds, labels, CFG.n_way)
    metrics["loss"] = float(np.mean(losses))
    
    return metrics


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main training loop for Reptile meta-learning."""
    print(f"Using device: {CFG.device}")
    
    # Initialize model and loss
    model = MalwareNet(CFG.input_dim, hidden=512, n_way=CFG.n_way).to(CFG.device)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    # Create datasets and data loaders
    if CFG.use_csv_loader:
        # Use CSV-based dataloader (using helper functions from data_loader.py)
        print(f"Using CSV-based dataloader (via data_loader.py helpers)")
        print(f"Features dir: {CFG.features_dir}")
        print(f"Split CSV: {CFG.split_csv_path}")
        
        train_ds = CSVBasedMalwareDataset(
            features_dir=CFG.features_dir,
            split_csv_path=CFG.split_csv_path,
            split="train",
            k_shot=CFG.k_shot,
            q_query=CFG.q_query,
            n_way=CFG.n_way,
            input_dim=CFG.input_dim,
            seed=42
        )
        val_ds = CSVBasedMalwareDataset(
            features_dir=CFG.features_dir,
            split_csv_path=CFG.split_csv_path,
            split="val",
            k_shot=CFG.k_shot,
            q_query=CFG.q_query,
            n_way=CFG.n_way,
            input_dim=CFG.input_dim,
            seed=42
        )
        
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    else:
        # Use JSON-based dataloader (original method)
        print(f"Using JSON-based dataloader from {CFG.data_json}")
        train_ds = MalwareDataset(CFG.data_json, "train", CFG.k_shot, CFG.q_query)
        val_ds = MalwareDataset(CFG.data_json, "val", CFG.k_shot, CFG.q_query)
        
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Training loop
    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        
        # Train and validate
        train_metrics = run_epoch(model, train_loader, train_iter, loss_fn, train=True)
        val_metrics = run_epoch(model, val_loader, val_iter, loss_fn, train=False)

        # Print progress
        print(f"Train Acc: {train_metrics['acc']*100:.2f}% | "
              f"Val Acc: {val_metrics['acc']*100:.2f}%")
        
        # Log metrics
        logger.add(epoch, train_metrics, val_metrics)
        
        # Save best model
        if logger.should_save_best(val_metrics['acc']):
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
                "val_acc": val_metrics['acc'],
            }, logger.model_path)
            
            print(f" Saved best model: {logger.model_path} "
                  f"(val_acc={val_metrics['acc']*100:.2f}%)")

    # Training complete
    print(f"\n Training done. Logs saved at: {logger.path}")
    print(f" Best model saved at: {logger.model_path}")


if __name__ == "__main__":
    main()
