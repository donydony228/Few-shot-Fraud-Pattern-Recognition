"""MI-MAML meta-learning algorithm for few-shot malware classification.

This module implements MAML-Inspired (MI-MAML) meta-learning algorithm, specifically
the FOMAML (First-Order MAML) variant. MI-MAML is designed for few-shot learning
scenarios with N-way K-shot classification tasks.

The module includes:
- Dataset loader for malware feature files (using src.extraction.data_loader)
- Lightweight MLP model with functional forward pass
- MI-MAML/FOMAML meta-learning implementation
- Cyclic inner learning rate scheduling
- Training and validation loops with gradient accumulation
- Logging and model checkpointing

Example:
    python mi_maml.py

Expected data structure:
    Data should be organized in features/{label}/{id}.npy format.
    label_split.csv should contain train/val/test splits.
    Each sample is a .npy file containing feature vectors (default 1280-dim).
"""

import math
import os
import random
import json
import sys
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
    """Configuration parameters for MI-MAML meta-learning.

    Attributes:
        n_way: Number of classes in each task (default: 3 for 2 malware + 1 benign).
        k_shot: Number of support samples per class (default: 1).
        q_query: Number of query samples per class (default: 5).
        input_dim: Dimensionality of input features (default: 1280).
        inner_lr: Base learning rate for inner loop (will be cycled).
        inner_steps: Number of gradient steps in inner loop.
        meta_lr: Learning rate for outer loop (Adam optimizer).
        meta_batch_size: Number of tasks per meta-update (gradient accumulation).
        max_epoch: Maximum number of training epochs.
        train_episodes_per_epoch: Number of tasks per training epoch.
        val_episodes_per_epoch: Number of tasks per validation epoch.
        seed: Random seed for reproducibility.
        device: Computation device (cuda/cpu/mps).
        data_json: Path to JSON file containing dataset structure.
        log_dir: Directory for saving logs and model checkpoints.
        num_workers: Number of workers for DataLoader.
        pin_memory: Whether to pin memory for faster GPU transfer.
    """
    # Task setup
    n_way = 3
    k_shot = 1  # Number of support samples per class during training
    q_query = 5
    input_dim = 1280
    
    # Training/validation/test configuration
    # Unified config: train=1-shot, val=1-shot, test=1-shot (consistent)
    val_k_shot = 1  # k_shot for validation (same as training)
    test_k_shot = 1  # k_shot for testing (same as training)

    # Inner / outer loop hyperparameters
    inner_lr = 0.01
    inner_steps = 5
    meta_lr = 1e-3
    meta_batch_size = 8

    # Training configuration
    max_epoch = 200
    train_episodes_per_epoch = 200
    val_episodes_per_epoch = 60
    test_episodes_per_epoch = 100  # Number of episodes for testing
    
    # Early stopping and learning rate scheduling
    early_stopping_patience = 15  # Stop training if no improvement for 15 epochs
    lr_decay_factor = 0.5  # Learning rate decay factor
    lr_decay_patience = 10  # Reduce learning rate if no improvement for 10 epochs
    min_lr = 1e-6  # Minimum learning rate

    # IO configuration
    # Paths relative to project root (parent directory of meta_learning/)
    features_dir = "../MalVis_dataset_small/features"
    split_csv_path = "../MalVis_dataset_small/label_split.csv"
    log_dir = "logs_mimaml"

    # System configuration
    seed = 42
    
    # Device selection with MPS support for MacBook
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    num_workers = 0
    pin_memory = torch.cuda.is_available()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_label(n_way: int, num_per_class: int) -> torch.Tensor:
    """Create labels for N-way classification task.

    Args:
        n_way: Number of classes.
        num_per_class: Number of samples per class.

    Returns:
        Long tensor of shape (n_way * num_per_class,) containing class labels.
    """
    return torch.arange(n_way).repeat_interleave(num_per_class).long()


def calculate_metrics(preds: List[int], labels: List[int],
                     num_classes: int) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics.

    Args:
        preds: List of predicted class indices.
        labels: List of ground truth class indices.
        num_classes: Number of classes.

    Returns:
        Dictionary containing accuracy, F1 scores, precision, recall, and
        confusion matrix.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    
    return {
        "acc": (preds == labels).mean().item() if preds.size else 0.0,
        "f1_macro": float(
            f1_score(labels, preds, average="macro", zero_division=0)
        ) if preds.size else 0.0,
        "f1_weighted": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ) if preds.size else 0.0,
        "precision": float(
            precision_score(labels, preds, average="macro", zero_division=0)
        ) if preds.size else 0.0,
        "recall": float(
            recall_score(labels, preds, average="macro", zero_division=0)
        ) if preds.size else 0.0,
        "cm": confusion_matrix(labels, preds).tolist() if preds.size else [[0]],
    }


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """Generate timestamp string.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cyclic_inner_lr(base_lr: float, step_idx: int, total_steps: int) -> float:
    """Calculate cyclic learning rate for inner loop.

    Implements 1-cycle cosine annealing schedule:
        lr = base_lr * 0.5 * (1 + cos(2π * step / total_steps))

    Args:
        base_lr: Base learning rate.
        step_idx: Current step index.
        total_steps: Total number of steps.

    Returns:
        Learning rate for current step.
    """
    return float(
        base_lr * 0.5 * (1.0 + math.cos(2.0 * math.pi * (step_idx / max(1, total_steps))))
    )


class Logger:
    """Logger for training statistics and model checkpointing.
    
    This class manages JSON logging and best model checkpointing during training.
    """
    
    def __init__(self, experiment_name: str = "mi_maml") -> None:
        """Initialize logger.

        Args:
            experiment_name: Base name for experiment files.
        """
        ensure_dir(CFG.log_dir)
        timestamp_str = timestamp()
        
        self.path = os.path.join(CFG.log_dir, f"{experiment_name}_{timestamp_str}.json")
        self.model_path = os.path.join(CFG.log_dir, f"mi_maml_best_{timestamp_str}.pth")
        self.experiment_name = f"{experiment_name}_experiment_{timestamp_str}"
        
        self.logs = {
            "experiment_name": self.experiment_name,
            "config": {
                k: v for k, v in CFG.__dict__.items()
                if not k.startswith("__") and isinstance(v, (int, float, str, bool))
            },
            "epochs": []
        }
        self.best_val = -1.0

    def add(self, epoch: int, train_stats: Dict[str, float],
            val_stats: Dict[str, float]) -> None:
        """Add epoch statistics to log.

        Args:
            epoch: Current epoch number.
            train_stats: Training metrics dictionary.
            val_stats: Validation metrics dictionary.
        """
        self.logs["epochs"].append({
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats
        })
        
        with open(self.path, "w") as f:
            json.dump(self.logs, f, indent=2)

    def should_save_best(self, val_acc: float) -> bool:
        """Check if current model is the best so far.

        Args:
            val_acc: Validation accuracy.

        Returns:
            True if this is the best model.
        """
        if val_acc > self.best_val:
            self.best_val = val_acc
            return True
        return False


# Initialize random seeds
set_seed(CFG.seed)


# ============================================================================
# DATASET
# ============================================================================
# 
# Note: Old MalwareDataset and make_loader have been removed
# Now using create_meta_learning_dataloaders from src.extraction.data_loader
#


# ============================================================================
# MODEL
# ============================================================================

class MalwareHead(nn.Module):
    """Lightweight MLP model for malware classification.
    
    Architecture:
        - LayerNorm(input_dim)
        - Linear(input_dim, hidden) -> ReLU -> Dropout
        - Linear(hidden, hidden) -> ReLU -> Dropout
        - Linear(hidden, n_way)
    
    The model supports both standard forward pass and functional forward pass
    for meta-learning algorithms that need to work with temporary parameters.
    """
    
    def __init__(self, input_dim: int = 1280, hidden: int = 512,
                 n_way: int = 3, p_drop: float = 0.4):
        """Initialize model.

        Args:
            input_dim: Dimensionality of input features.
            hidden: Hidden layer size.
            n_way: Number of output classes.
            p_drop: Dropout probability.
        """
        
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_way)
        self.p_drop = p_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            Logits tensor of shape (batch_size, n_way).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.p_drop, training=self.training)
        x = torch.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.p_drop, training=self.training)
        return self.fc3(x)

    def ordered_params(self) -> List[torch.Tensor]:
        """Return parameters in a stable order for functional forward.

        Returns:
            List of parameters: [norm.weight, norm.bias, fc1.weight, fc1.bias,
            fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        """
        return [
            self.norm.weight,
            self.norm.bias,
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.fc3.weight,
            self.fc3.bias,
        ]


def functional_forward(model: MalwareHead, x: torch.Tensor,
                      fast_params: List[torch.Tensor]) -> torch.Tensor:
    """Functional forward pass using custom parameters.

    This is used for meta-learning where we need to compute forward passes
    with temporary parameters during inner loop adaptation.

    Args:
        model: Model instance (used for p_drop attribute).
        x: Input tensor.
        fast_params: List of parameter tensors in the order returned by
            model.ordered_params().

    Returns:
        Logits tensor.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    param_idx = 0
    
    # LayerNorm
    w_ln, b_ln = fast_params[param_idx], fast_params[param_idx + 1]
    param_idx += 2
    x = nn.functional.layer_norm(x, (w_ln.shape[0],), w_ln, b_ln, eps=1e-5)
    
    # FC1
    w, b = fast_params[param_idx], fast_params[param_idx + 1]
    param_idx += 2
    x = nn.functional.linear(x, w, b)
    x = nn.functional.relu(x)
    x = nn.functional.dropout(x, p=model.p_drop, training=True)
    
    # FC2
    w, b = fast_params[param_idx], fast_params[param_idx + 1]
    param_idx += 2
    x = nn.functional.linear(x, w, b)
    x = nn.functional.relu(x)
    x = nn.functional.dropout(x, p=model.p_drop, training=True)
    
    # FC3 (logits)
    w, b = fast_params[param_idx], fast_params[param_idx + 1]
    param_idx += 2
    x = nn.functional.linear(x, w, b)
    
    return x


# ============================================================================
# MI-MAML ALGORITHM
# ============================================================================

def maml_inner_adapt(model: MalwareHead, task_batch: torch.Tensor,
                     n_way: int, k_shot: int, q_query: int,
                     inner_steps: int, inner_base_lr: float, device: str,
                     loss_fn: nn.Module = None) -> Tuple[torch.Tensor, float,
                                                          List[torch.Tensor]]:
    """Perform one MI-MAML inner adaptation for a single task.
    
    Supports 0-shot learning: when k_shot=0, skip inner loop adaptation and use base parameters directly.

    MI-MAML algorithm (FOMAML variant):
    1. Initialize fast_params = clone of base parameters
    2. For K inner steps (if k_shot > 0):
       a. Compute loss on support set using fast_params
       b. Update: fast_params = fast_params - α_t * grad
       c. Use cyclic learning rate α_t
    3. Compute query loss using adapted fast_params (or base params for 0-shot)
    4. Return gradients w.r.t. fast_params (approximate gradient w.r.t. base)

    Args:
        model: Model instance (used for ordered_params and functional_forward).
        task_batch: Batch of one task, shape (1, n_way, k+q, feat_dim) or (1, n_way, q, feat_dim) for 0-shot.
        n_way: Number of classes.
        k_shot: Number of support samples per class (can be 0).
        q_query: Number of query samples per class.
        inner_steps: Number of inner loop steps.
        inner_base_lr: Base learning rate for inner loop.
        device: Implementing device.
        loss_fn: Loss function (default: CrossEntropyLoss).

    Returns:
        Tuple of (query_loss, query_accuracy, gradients_wrt_fast).
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    task = task_batch.squeeze(0).to(device)  # [n_way, k+q, feat] or [n_way, q, feat] for 0-shot
    
    # Split support and query (supports 0-shot)
    if k_shot > 0:
        support = task[:, :k_shot, :].reshape(n_way * k_shot, -1)
        query = task[:, k_shot:, :].reshape(n_way * q_query, -1)
        y_s = create_label(n_way, k_shot).to(device)
    else:
        # 0-shot: all samples are query
        support = None
        query = task.reshape(n_way * q_query, -1)
        y_s = None
    
    y_q = create_label(n_way, q_query).to(device)

    # Initialize fast parameters from base
    base_params = model.ordered_params()
    fast = [p.clone().detach().requires_grad_(True) for p in base_params]

    # Inner loop: adapt on support set (only if k_shot > 0)
    if k_shot > 0 and inner_steps > 0:
        for step in range(inner_steps):
            # Compute cyclic learning rate
            lr_t = cyclic_inner_lr(inner_base_lr, step, inner_steps)
            
            # Forward pass on support set
            logits_s = functional_forward(model, support, fast)
            loss_s = loss_fn(logits_s, y_s)
            
            # Compute gradients
            grads = torch.autograd.grad(loss_s, fast, create_graph=False)
            
            # Update fast parameters: θ' ← θ' - lr_t * grad
            fast = [
                w - lr_t * g if g is not None else w
                for w, g in zip(fast, grads)
            ]
            
            # Re-enable gradients for next step
            fast = [w.detach().requires_grad_(True) for w in fast]
    # else: 0-shot case, fast params remain as copies of base params (no adaptation)

    # Compute meta-loss on query set
    logits_q = functional_forward(model, query, fast)
    loss_q = loss_fn(logits_q, y_q)
    acc_q = (logits_q.argmax(-1) == y_q).float().mean().item()

    # Compute gradients w.r.t. fast parameters (approximate base gradients)
    # For 0-shot, this is equivalent to computing gradients w.r.t. base parameters directly
    grads_q = torch.autograd.grad(loss_q, fast, create_graph=False)

    return loss_q.detach(), acc_q, grads_q


def accumulate_grads(model: MalwareHead,
                     grads_accum: List[torch.Tensor]) -> None:
    """Write accumulated gradients back into model parameters.

    Args:
        model: Model instance.
        grads_accum: List of gradient tensors in same order as model.ordered_params().
    """
    model_params = model.ordered_params()
    with torch.no_grad():
        for param, grad in zip(model_params, grads_accum):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad


# ============================================================================
# TRAINING
# ============================================================================

def run_epoch(train: bool, model: MalwareHead, loader: DataLoader,
              cfg: CFG, test_k_shot: int = None) -> Dict[str, float]:
    """Run one training or validation epoch.

    Args:
        train: Whether to train the model (True) or evaluate (False).
        model: Model to train/evaluate.
        loader: DataLoader for the dataset.
        cfg: Configuration object.
        test_k_shot: k_shot to use for validation/testing (if None, uses cfg.k_shot).

    Returns:
        Dictionary of metrics (loss, acc, etc.).
    """
    device = cfg.device
    loss_fn = nn.CrossEntropyLoss()
    model.train() if train else model.eval()

    # Get or create optimizer
    meta_opt = getattr(run_epoch, "_opt", None)
    if meta_opt is None:
        meta_opt = torch.optim.Adam(model.parameters(), lr=cfg.meta_lr)
        run_epoch._opt = meta_opt

    episodes = cfg.train_episodes_per_epoch if train else cfg.val_episodes_per_epoch
    iterator = iter(loader)

    # Determine k_shot to use
    k_shot_to_use = cfg.k_shot if train else (test_k_shot if test_k_shot is not None else cfg.k_shot)

    total_loss = 0.0
    total_acc = 0.0
    meta_batch_grads = None

    for episode in tqdm(range(episodes), desc="Train" if train else "Val"):
        try:
            task = next(iterator)  # [1, n_way, k+q, feat] or [1, n_way, q, feat] for 0-shot
        except StopIteration:
            iterator = iter(loader)
            task = next(iterator)

        # Perform inner adaptation for one task
        # Use cfg.k_shot for training, test_k_shot for validation/testing (can be 0)
        q_loss, q_acc, grads_q = maml_inner_adapt(
            model, task, cfg.n_way, k_shot_to_use, cfg.q_query,
            cfg.inner_steps, cfg.inner_lr, device, loss_fn
        )

        total_loss += q_loss.item()
        total_acc += q_acc

        # Accumulate gradients for meta-batch
        if meta_batch_grads is None:
            meta_batch_grads = [g.clone() for g in grads_q]
        else:
            for i in range(len(meta_batch_grads)):
                meta_batch_grads[i] += grads_q[i]

        # Perform meta-update every meta_batch_size tasks
        if (episode + 1) % cfg.meta_batch_size == 0:
            avg_grads = [g / cfg.meta_batch_size for g in meta_batch_grads]
            meta_opt.zero_grad(set_to_none=True)
            accumulate_grads(model, avg_grads)
            meta_opt.step()
            meta_batch_grads = None

    # Handle remaining gradients if not evenly divisible
    if meta_batch_grads is not None and train:
        remaining = (episodes % cfg.meta_batch_size) or cfg.meta_batch_size
        avg_grads = [g / remaining for g in meta_batch_grads]
        meta_opt.zero_grad(set_to_none=True)
        accumulate_grads(model, avg_grads)
        meta_opt.step()

    # Compute average metrics
    avg_loss = total_loss / max(1, episodes)
    avg_acc = total_acc / max(1, episodes)
    
    return {"loss": float(avg_loss), "acc": float(avg_acc)}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main training loop for MI-MAML meta-learning."""
    ensure_dir(CFG.log_dir)
    print(
        f"[MI-MAML] device={CFG.device}  "
        f"n_way={CFG.n_way}  k_shot={CFG.k_shot}  q_query={CFG.q_query}"
    )

    # Initialize logger
    logger = Logger(experiment_name="mi_maml")

    # Create data loaders using new meta-learning dataloader
    # Convert relative paths to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.abspath(os.path.join(script_dir, CFG.features_dir))
    split_csv_path = os.path.abspath(os.path.join(script_dir, CFG.split_csv_path))
    
    print(f"\n[DataLoader] Loading from: {features_dir}")
    print(f"[DataLoader] Split CSV: {split_csv_path}")
    
    # Get configuration
    val_k_shot = getattr(CFG, 'val_k_shot', CFG.k_shot)  # k_shot for validation
    test_k_shot = getattr(CFG, 'test_k_shot', CFG.k_shot)  # k_shot for testing (default same as training)
    
    print(f"\n[Config] Training/Validation/Test Setup:")
    print(f"  Task: {CFG.n_way}-way, {CFG.k_shot}-shot, {CFG.q_query}-query")
    print(f"  Train: {CFG.k_shot}-shot")
    print(f"  Val:   {val_k_shot}-shot")
    print(f"  Test:  {test_k_shot}-shot")
    
    # Create training and validation dataloaders (use same k_shot for consistency)
    dataloaders_train = create_meta_learning_dataloaders(
        features_dir=features_dir,
        split_csv_path=split_csv_path,
        n_way=CFG.n_way,
        k_shot=CFG.k_shot,  # Training uses 1-shot
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
    
    # Create validation dataloader (use val_k_shot)
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
    
    # Create test dataloaders (use test_k_shot)
    if test_k_shot != CFG.k_shot:
        dataloaders_test = create_meta_learning_dataloaders(
            features_dir=features_dir,
            split_csv_path=split_csv_path,
            n_way=CFG.n_way,
            k_shot=test_k_shot,  # Testing uses specified k_shot
            q_query=CFG.q_query,
            train_episodes_per_epoch=CFG.train_episodes_per_epoch,
            val_episodes_per_epoch=CFG.val_episodes_per_epoch,
            test_episodes_per_epoch=CFG.test_episodes_per_epoch,
            normalize=True,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
            seed=CFG.seed
        )
        test_loaders = {
            'test_unseen': dataloaders_test['test_unseen'],
            'test_generalized': dataloaders_test['test_generalized']
        }
        print(f"[DataLoader] Testing uses {test_k_shot}-shot dataloaders")
    else:
        # If test and training use same k_shot, use train dataloaders directly
        test_loaders = {
            'test_unseen': dataloaders_train['test_unseen'],
            'test_generalized': dataloaders_train['test_generalized']
        }
        print(f"[DataLoader] Testing uses same {CFG.k_shot}-shot dataloaders as training")
    
    print(f"\n[DataLoader] Created dataloaders:")
    print(f"  Train: {CFG.k_shot}-shot, {CFG.train_episodes_per_epoch} episodes/epoch")
    print(f"  Val:   {val_k_shot}-shot, {CFG.val_episodes_per_epoch} episodes/epoch")
    print(f"  Test:  {test_k_shot}-shot, {CFG.test_episodes_per_epoch} episodes/split")

    # Initialize model
    model = MalwareHead(CFG.input_dim, hidden=512, n_way=CFG.n_way).to(CFG.device)

    # Training loop with early stopping and learning rate decay
    best_val = -1.0
    save_path = os.path.join(CFG.log_dir, f"mi_maml_{timestamp()}.pth")
    
    # Early stopping and learning rate scheduling variables
    patience_counter = 0
    lr_patience_counter = 0
    best_epoch = 0
    
    # Get optimizer (created in run_epoch, need to initialize here)
    _ = run_epoch(train=True, model=model, loader=train_loader, cfg=CFG)
    meta_opt = run_epoch._opt
    current_lr = CFG.meta_lr

    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        print(f"Current learning rate: {current_lr:.6f}")
        
        train_stats = run_epoch(train=True, model=model, loader=train_loader,
                               cfg=CFG)
        val_stats = run_epoch(train=False, model=model, loader=val_loader,
                             cfg=CFG, test_k_shot=val_k_shot)  # Use val_k_shot for validation

        print(
            f"Train: acc={train_stats['acc']*100:.2f}%  "
            f"loss={train_stats['loss']:.4f}"
        )
        print(
            f"Val  : acc={val_stats['acc']*100:.2f}%  "
            f"loss={val_stats['loss']:.4f}"
        )

        # Log epoch statistics
        logger.add(epoch, train_stats, val_stats)

        # Check if validation improved
        improved = False
        if val_stats["acc"] > best_val:
            best_val = val_stats["acc"]
            best_epoch = epoch
            patience_counter = 0
            lr_patience_counter = 0
            improved = True
            
            cfg_dict = {
                k: v for k, v in CFG.__dict__.items()
                if not k.startswith("__") and isinstance(
                    v, (int, float, str, bool, list)
                )
            }
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg_dict,
                "epoch": epoch,
                "val_acc": best_val,
            }, save_path)
            
            print(
                f"✓ Saved best model (epoch {epoch}, val_acc={best_val*100:.2f}%)"
            )
        else:
            patience_counter += 1
            lr_patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CFG.early_stopping_patience})")
        
        # Learning rate decay
        if lr_patience_counter >= CFG.lr_decay_patience:
            current_lr = max(current_lr * CFG.lr_decay_factor, CFG.min_lr)
            for param_group in meta_opt.param_groups:
                param_group['lr'] = current_lr
            lr_patience_counter = 0
            print(f"  ⚠ Learning rate decayed to: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= CFG.early_stopping_patience:
            print(f"\n⚠ Early stopping triggered! No improvement for {CFG.early_stopping_patience} epochs.")
            print(f"Best validation accuracy: {best_val*100:.2f}% at epoch {best_epoch}")
            break

    print(f"\n Training done. Logs saved at: {logger.path}")
    print(f" Best model saved at: {save_path}")
    
    # ============================================================================
    # Testing phase: Evaluate on best model
    # ============================================================================
    print("\n" + "=" * 60)
    print("Starting Testing Phase")
    print("=" * 60)
    
    # Load best model
    print(f"\n[Test] Loading best model: {save_path}")
    checkpoint = torch.load(save_path, map_location=CFG.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Test on three test sets
    test_results = {}
    for test_name, test_loader in test_loaders.items():
        print(f"\n[Test] Testing {test_name}...")
        test_stats = run_epoch(
            train=False,
            model=model,
            loader=test_loader,
            cfg=CFG,
            test_k_shot=test_k_shot  # Use test_k_shot for testing
        )
        test_results[test_name] = test_stats
        
        print(
            f"{test_name}: acc={test_stats['acc']*100:.2f}%  "
            f"loss={test_stats['loss']:.4f}"
        )
    
    # Save test results to log
    logger.logs["test_results"] = test_results
    logger.logs["test_config"] = {
        "test_k_shot": test_k_shot,
        "test_episodes_per_epoch": CFG.test_episodes_per_epoch
    }
    with open(logger.path, "w") as f:
        json.dump(logger.logs, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
    print("\nTest Results Summary:")
    for test_name, stats in test_results.items():
        print(f"  {test_name}: {stats['acc']*100:.2f}%")
    print(f"\nAll results saved to: {logger.path}")


if __name__ == "__main__":
    main()