"""MI-MAML meta-learning algorithm for few-shot malware classification.

This module implements MAML-Inspired (MI-MAML) meta-learning algorithm, specifically
the FOMAML (First-Order MAML) variant. MI-MAML is designed for few-shot learning
scenarios with N-way K-shot classification tasks.

The module includes:
- Dataset loader for malware feature files
- Lightweight MLP model with functional forward pass
- MI-MAML/FOMAML meta-learning implementation
- Cyclic inner learning rate scheduling
- Training and validation loops with gradient accumulation
- Logging and model checkpointing

Example:
    python mi_maml.py

Expected data structure:
    Data should be organized in JSON format with train/val/test splits.
    Each sample is a .npy file containing 1280-dimensional feature vectors.
"""

import math
import os
import random
import json
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
    k_shot = 1
    q_query = 5
    input_dim = 1280

    # Inner / outer loop hyperparameters
    inner_lr = 0.01
    inner_steps = 5
    meta_lr = 1e-3
    meta_batch_size = 8

    # Training configuration
    max_epoch = 200
    train_episodes_per_epoch = 200
    val_episodes_per_epoch = 60

    # IO configuration
    data_json = "../malware_data_structure.json"
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

class MalwareDataset(Dataset):
    """Dataset class for malware feature vectors with few-shot task generation.

    This dataset loads malware features from JSON structure and dynamically
    generates few-shot tasks. Each task contains:
    - 2 randomly selected malware families (abnormal classes)
    - 1 benign class (normal class)
    - Total of 3 classes (n_way=3)

    Features are loaded from .npy files and normalized using Z-score normalization.
    """
    
    def __init__(self, json_path: str, split: str, k_shot: int, q_query: int,
                 input_dim: int = 1280):
        """Initialize dataset.

        Args:
            json_path: Path to JSON file containing dataset structure.
            split: Data split ('train', 'val', or 'test').
            k_shot: Number of support samples per class.
            q_query: Number of query samples per class.
            input_dim: Dimensionality of input features.
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)[split]
        
        self.classes = list(self.data.keys())
        self.k_shot = k_shot
        self.q_query = q_query
        self.input_dim = input_dim
        self.normal = "benign"

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a single few-shot task.

        Args:
            idx: Task index.

        Returns:
            Tensor of shape (n_way, k_shot + q_query, input_dim) containing
            features for one task.
        """
        np.random.seed(CFG.seed + idx)
        
        # Select 2 malware families + 1 benign
        frauds = [c for c in self.classes if c != self.normal]
        if len(frauds) < 2:
            raise ValueError(
                "Not enough malware families in JSON for a 3-way task."
            )

        selected = (
            np.random.choice(frauds, CFG.n_way - 1, replace=False).tolist()
            + [self.normal]
        )

        task = []
        for cls in selected:
            files = self.data[cls]
            need = self.k_shot + self.q_query
            chosen = np.random.choice(files, need, replace=(len(files) < need))
            
            cls_feats = []
            for filepath in chosen:
                path = self._fix_path(filepath)
                
                try:
                    arr = np.load(path)
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    # Per-sample Z-score normalization
                    mu, std = arr.mean(), arr.std()
                    arr = (arr - mu) / (std + 1e-6)
                except Exception:
                    # Fallback to zero vector if loading fails
                    arr = np.zeros(self.input_dim, dtype=np.float32)
                
                cls_feats.append(arr.astype(np.float32))
            
            task.append(torch.tensor(np.stack(cls_feats), dtype=torch.float32))
        
        return torch.stack(task)  # [n_way, k+q, feat]

    def _fix_path(self, path: str) -> str:
        """Attempt to fix file path by trying common prefixes.

        Args:
            path: Original file path.

        Returns:
            Corrected absolute path if file exists, original path otherwise.
        """
        if os.path.exists(path):
            return path
        
        for prefix in ["../", "../../", "./"]:
            candidate = os.path.join(prefix, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        
        return path

    def __len__(self) -> int:
        """Return virtual dataset length.

        Returns:
            Large fixed number (actual number of tasks controlled by epochs).
        """
        return 100000


def make_loader(json_path: str, split: str) -> DataLoader:
    """Create DataLoader for a given split.

    Args:
        json_path: Path to JSON file containing dataset structure.
        split: Data split ('train', 'val', or 'test').

    Returns:
        DataLoader instance for the specified split.
    """
    dataset = MalwareDataset(
        json_path, split, CFG.k_shot, CFG.q_query, CFG.input_dim
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory
    )


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
                 n_way: int = 3, p_drop: float = 0.3):
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

    MI-MAML algorithm (FOMAML variant):
    1. Initialize fast_params = clone of base parameters
    2. For K inner steps:
       a. Compute loss on support set using fast_params
       b. Update: fast_params = fast_params - α_t * grad
       c. Use cyclic learning rate α_t
    3. Compute query loss using adapted fast_params
    4. Return gradients w.r.t. fast_params (approximate gradient w.r.t. base)

    Args:
        model: Model instance (used for ordered_params and functional_forward).
        task_batch: Batch of one task, shape (1, n_way, k+q, feat_dim).
        n_way: Number of classes.
        k_shot: Number of support samples per class.
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

    task = task_batch.squeeze(0).to(device)  # [n_way, k+q, feat]
    support = task[:, :k_shot, :].reshape(n_way * k_shot, -1)
    query = task[:, k_shot:, :].reshape(n_way * q_query, -1)

    y_s = create_label(n_way, k_shot).to(device)
    y_q = create_label(n_way, q_query).to(device)

    # Initialize fast parameters from base
    base_params = model.ordered_params()
    fast = [p.clone().detach().requires_grad_(True) for p in base_params]

    # Inner loop: adapt on support set
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

    # Compute meta-loss on query set
    logits_q = functional_forward(model, query, fast)
    loss_q = loss_fn(logits_q, y_q)
    acc_q = (logits_q.argmax(-1) == y_q).float().mean().item()

    # Compute gradients w.r.t. fast parameters (approximate base gradients)
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
              cfg: CFG) -> Dict[str, float]:
    """Run one training or validation epoch.

    Args:
        train: Whether to train the model (True) or evaluate (False).
        model: Model to train/evaluate.
        loader: DataLoader for the dataset.
        cfg: Configuration object.

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

    total_loss = 0.0
    total_acc = 0.0
    meta_batch_grads = None

    for episode in tqdm(range(episodes), desc="Train" if train else "Val"):
        try:
            task = next(iterator)  # [1, n_way, k+q, feat]
        except StopIteration:
            iterator = iter(loader)
            task = next(iterator)

        # Perform inner adaptation for one task
        q_loss, q_acc, grads_q = maml_inner_adapt(
            model, task, cfg.n_way, cfg.k_shot, cfg.q_query,
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

    # Create data loaders
    train_loader = make_loader(CFG.data_json, "train")
    val_loader = make_loader(CFG.data_json, "val")

    # Initialize model
    model = MalwareHead(CFG.input_dim, hidden=512, n_way=CFG.n_way).to(CFG.device)

    # Training loop
    best_val = -1.0
    save_path = os.path.join(CFG.log_dir, f"mi_maml_{timestamp()}.pth")

    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        
        train_stats = run_epoch(train=True, model=model, loader=train_loader,
                               cfg=CFG)
        val_stats = run_epoch(train=False, model=model, loader=val_loader,
                             cfg=CFG)

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

        # Save best model
        if val_stats["acc"] > best_val:
            best_val = val_stats["acc"]
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
                f" Saved best model: {save_path} "
                f"(val_acc={best_val*100:.2f}%)"
            )

    print(f"\n Training done. Logs saved at: {logger.path}")
    print(f" Best model saved at: {save_path}")


if __name__ == "__main__":
    main()