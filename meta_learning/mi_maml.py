# """MI-MAML meta-learning algorithm for few-shot malware classification.

# This module implements MAML-Inspired (MI-MAML) meta-learning algorithm, specifically
# the FOMAML (First-Order MAML) variant. MI-MAML is designed for few-shot learning
# scenarios with N-way K-shot classification tasks.

# The module includes:
# - Dataset loader for malware feature files
# - Lightweight MLP model with functional forward pass
# - MI-MAML/FOMAML meta-learning implementation
# - Cyclic inner learning rate scheduling
# - Training and validation loops with gradient accumulation
# - Logging and model checkpointing

# Example:
#     python mi_maml.py

# Expected data structure:
#     Data should be organized in JSON format with train/val/test splits.
#     Each sample is a .npy file containing 1280-dimensional feature vectors.
# """

# import math
import os
# import random
# import json
# from datetime import datetime
# from typing import Dict, List, Tuple, Any

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from tqdm.auto import tqdm
# from sklearn.metrics import (
#     f1_score,
#     precision_score,
#     recall_score,
#     confusion_matrix,
# )

# Import helper functions from data_loader (without modifying data_loader.py)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.extraction.data_loader import get_label_splits, collect_file_paths


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class CFG:
#     """Configuration parameters for MI-MAML meta-learning.

#     Attributes:
#         n_way: Number of classes in each task (default: 3 for 2 malware + 1 benign).
#         k_shot: Number of support samples per class (default: 1).
#         q_query: Number of query samples per class (default: 5).
#         input_dim: Dimensionality of input features (default: 1280).
#         inner_lr: Base learning rate for inner loop (will be cycled).
#         inner_steps: Number of gradient steps in inner loop.
#         meta_lr: Learning rate for outer loop (Adam optimizer).
#         meta_batch_size: Number of tasks per meta-update (gradient accumulation).
#         max_epoch: Maximum number of training epochs.
#         train_episodes_per_epoch: Number of tasks per training epoch.
#         val_episodes_per_epoch: Number of tasks per validation epoch.
#         seed: Random seed for reproducibility.
#         device: Computation device (cuda/cpu/mps).
#         data_json: Path to JSON file containing dataset structure.
#         log_dir: Directory for saving logs and model checkpoints.
#         num_workers: Number of workers for DataLoader.
#         pin_memory: Whether to pin memory for faster GPU transfer.
#     """
#     # Task setup
#     n_way = 3
#     k_shot = 1
#     q_query = 5
#     input_dim = 1280

#     # Inner / outer loop hyperparameters
#     inner_lr = 0.01
#     inner_steps = 5
#     meta_lr = 1e-3
#     meta_batch_size = 8

#     # Training configuration
#     max_epoch = 200
#     train_episodes_per_epoch = 200
#     val_episodes_per_epoch = 60

#     # IO configuration
#     # data_json = "../malware_data_structure.json"  # For JSON-based loading
#     features_dir = "../MalVis_dataset_small/features"  # For CSV-based loading
#     split_csv_path = "../MalVis_dataset_small/label_split.csv"  # For CSV-based loading
#     use_csv_loader = True  # Set to True to use data_loader.py, False to use JSON
#     log_dir = "logs_mimaml"

#     # System configuration
#     seed = 42
    
#     # Device selection with MPS support for MacBook
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
    
#     num_workers = 0
#     pin_memory = torch.cuda.is_available()


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def set_seed(seed: int = 42) -> None:
#     """Set random seeds for reproducibility.

#     Args:
#         seed: Random seed value.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def create_label(n_way: int, num_per_class: int) -> torch.Tensor:
#     """Create labels for N-way classification task.

#     Args:
#         n_way: Number of classes.
#         num_per_class: Number of samples per class.

#     Returns:
#         Long tensor of shape (n_way * num_per_class,) containing class labels.
#     """
#     return torch.arange(n_way).repeat_interleave(num_per_class).long()


# def calculate_metrics(preds: List[int], labels: List[int],
#                      num_classes: int) -> Dict[str, Any]:
#     """Calculate comprehensive classification metrics.

#     Args:
#         preds: List of predicted class indices.
#         labels: List of ground truth class indices.
#         num_classes: Number of classes.

#     Returns:
#         Dictionary containing accuracy, F1 scores, precision, recall, and
#         confusion matrix.
#     """
#     preds = np.asarray(preds)
#     labels = np.asarray(labels)
    
#     return {
#         "acc": (preds == labels).mean().item() if preds.size else 0.0,
#         "f1_macro": float(
#             f1_score(labels, preds, average="macro", zero_division=0)
#         ) if preds.size else 0.0,
#         "f1_weighted": float(
#             f1_score(labels, preds, average="weighted", zero_division=0)
#         ) if preds.size else 0.0,
#         "precision": float(
#             precision_score(labels, preds, average="macro", zero_division=0)
#         ) if preds.size else 0.0,
#         "recall": float(
#             recall_score(labels, preds, average="macro", zero_division=0)
#         ) if preds.size else 0.0,
#         "cm": confusion_matrix(labels, preds).tolist() if preds.size else [[0]],
#     }


# def ensure_dir(path: str) -> None:
#     """Create directory if it doesn't exist.

#     Args:
#         path: Directory path to create.
#     """
#     os.makedirs(path, exist_ok=True)


# def timestamp() -> str:
#     """Generate timestamp string.

#     Returns:
#         Timestamp string in format YYYYMMDD_HHMMSS.
#     """
#     return datetime.now().strftime("%Y%m%d_%H%M%S")


# def cyclic_inner_lr(base_lr: float, step_idx: int, total_steps: int) -> float:
#     """Calculate cyclic learning rate for inner loop.

#     Implements 1-cycle cosine annealing schedule:
#         lr = base_lr * 0.5 * (1 + cos(2π * step / total_steps))

#     Args:
#         base_lr: Base learning rate.
#         step_idx: Current step index.
#         total_steps: Total number of steps.

#     Returns:
#         Learning rate for current step.
#     """
#     return float(
#         base_lr * 0.5 * (1.0 + math.cos(2.0 * math.pi * (step_idx / max(1, total_steps))))
#     )


# class Logger:
#     """Logger for training statistics and model checkpointing.
    
#     This class manages JSON logging and best model checkpointing during training.
#     """
    
#     def __init__(self, experiment_name: str = "mi_maml") -> None:
#         """Initialize logger.

#         Args:
#             experiment_name: Base name for experiment files.
#         """
#         ensure_dir(CFG.log_dir)
#         timestamp_str = timestamp()
        
#         self.path = os.path.join(CFG.log_dir, f"{experiment_name}_{timestamp_str}.json")
#         self.model_path = os.path.join(CFG.log_dir, f"mi_maml_best_{timestamp_str}.pth")
#         self.experiment_name = f"{experiment_name}_experiment_{timestamp_str}"
        
#         self.logs = {
#             "experiment_name": self.experiment_name,
#             "config": {
#                 k: v for k, v in CFG.__dict__.items()
#                 if not k.startswith("__") and isinstance(v, (int, float, str, bool))
#             },
#             "epochs": []
#         }
#         self.best_val = -1.0

#     def add(self, epoch: int, train_stats: Dict[str, float],
#             val_stats: Dict[str, float]) -> None:
#         """Add epoch statistics to log.

#         Args:
#             epoch: Current epoch number.
#             train_stats: Training metrics dictionary.
#             val_stats: Validation metrics dictionary.
#         """
#         self.logs["epochs"].append({
#             "epoch": epoch,
#             "train": train_stats,
#             "val": val_stats
#         })
        
#         with open(self.path, "w") as f:
#             json.dump(self.logs, f, indent=2)

#     def should_save_best(self, val_acc: float) -> bool:
#         """Check if current model is the best so far.

#         Args:
#             val_acc: Validation accuracy.

#         Returns:
#             True if this is the best model.
#         """
#         if val_acc > self.best_val:
#             self.best_val = val_acc
#             return True
#         return False


# # Initialize random seeds
# set_seed(CFG.seed)


# # ============================================================================
# # DATASET
# # ============================================================================

# class CSVBasedMalwareDataset(Dataset):
#     """Dataset class for meta-learning using CSV label splits.
    
#     This dataset uses helper functions from data_loader.py to load data
#     based on CSV label splits and dynamically generates few-shot tasks.
#     """
    
#     def __init__(
#         self,
#         features_dir: str,
#         split_csv_path: str,
#         split: str,
#         k_shot: int,
#         q_query: int,
#         n_way: int = 3,
#         input_dim: int = 1280,
#         seed: int = 42,
#         normal_label: str = "benign"
#     ):
#         """Initialize CSV-based dataset.
        
#         Args:
#             features_dir: Path to features directory
#             split_csv_path: Path to label_split.csv
#             split: 'train', 'val', or 'test'
#             k_shot: Number of support samples per class
#             q_query: Number of query samples per class
#             n_way: Number of classes per task
#             input_dim: Feature dimensionality
#             seed: Random seed
#             normal_label: Name of normal/benign class
#         """
#         # Use helper functions from data_loader.py
#         seen_labels, unseen_labels = get_label_splits(split_csv_path)
        
#         # Determine which labels to use based on split
#         if split == "train" or split == "val":
#             self.labels = seen_labels
#         else:  # test
#             self.labels = unseen_labels
        
#         # Collect file paths for each label using helper function
#         all_files = collect_file_paths(features_dir, self.labels)
        
#         # Organize files by label
#         self.label_to_files: Dict[str, List[str]] = {}
#         for filepath in all_files:
#             label = os.path.basename(os.path.dirname(filepath))
#             if label not in self.label_to_files:
#                 self.label_to_files[label] = []
#             self.label_to_files[label].append(filepath)
        
#         self.k_shot = k_shot
#         self.q_query = q_query
#         self.n_way = n_way
#         self.input_dim = input_dim
#         self.seed = seed
#         self.normal_label = normal_label
#         self.classes = list(self.label_to_files.keys())
        
#         if len(self.classes) < n_way:
#             raise ValueError(
#                 f"Not enough classes ({len(self.classes)}) for {n_way}-way tasks. "
#                 f"Available: {self.classes}"
#             )
    
#     def __getitem__(self, idx: int) -> torch.Tensor:
#         """Generate a single few-shot task."""
#         np.random.seed(self.seed + idx)
        
#         # Select classes for this task
#         frauds = [c for c in self.classes if c != self.normal_label]
#         if len(frauds) < (self.n_way - 1):
#             raise ValueError(
#                 f"Not enough non-normal classes ({len(frauds)}) for {self.n_way}-way task"
#             )
        
#         selected_frauds = np.random.choice(frauds, self.n_way - 1, replace=False).tolist()
        
#         if self.normal_label not in self.classes:
#             raise ValueError(f"Normal class '{self.normal_label}' not found")
        
#         selected = selected_frauds + [self.normal_label]
#         task = []
#         need = self.k_shot + self.q_query
        
#         for cls in selected:
#             files = self.label_to_files[cls]
#             chosen = np.random.choice(files, need, replace=(len(files) < need))
            
#             cls_features = []
#             for filepath in chosen:
#                 try:
#                     arr = np.load(filepath)
#                     if arr.ndim > 1:
#                         arr = arr.flatten()
                    
#                     # Ensure correct dimension
#                     if arr.shape[0] != self.input_dim:
#                         if arr.shape[0] < self.input_dim:
#                             arr = np.pad(arr, (0, self.input_dim - arr.shape[0]))
#                         else:
#                             arr = arr[:self.input_dim]
                    
#                     # Z-score normalization
#                     arr = (arr - arr.mean()) / (arr.std() + 1e-6)
#                 except Exception:
#                     arr = np.zeros(self.input_dim, dtype=np.float32)
                
#                 cls_features.append(arr.astype(np.float32))
            
#             task.append(torch.tensor(np.stack(cls_features), dtype=torch.float32))
        
#         return torch.stack(task)  # Shape: [n_way, k+q, feat_dim]
    
#     def __len__(self) -> int:
#         return 100000


# class MalwareDataset(Dataset):
#     """Dataset class for malware feature vectors with few-shot task generation.

#     This dataset loads malware features from JSON structure and dynamically
#     generates few-shot tasks. Each task contains:
#     - 2 randomly selected malware families (abnormal classes)
#     - 1 benign class (normal class)
#     - Total of 3 classes (n_way=3)

#     Features are loaded from .npy files and normalized using Z-score normalization.
#     """
    
#     def __init__(self, json_path: str, split: str, k_shot: int, q_query: int,
#                  input_dim: int = 1280):
#         """Initialize dataset.

#         Args:
#             json_path: Path to JSON file containing dataset structure.
#             split: Data split ('train', 'val', or 'test').
#             k_shot: Number of support samples per class.
#             q_query: Number of query samples per class.
#             input_dim: Dimensionality of input features.
#         """
#         with open(json_path, "r") as f:
#             self.data = json.load(f)[split]
        
#         self.classes = list(self.data.keys())
#         self.k_shot = k_shot
#         self.q_query = q_query
#         self.input_dim = input_dim
#         self.normal = "benign"

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         """Generate a single few-shot task.

#         Args:
#             idx: Task index.

#         Returns:
#             Tensor of shape (n_way, k_shot + q_query, input_dim) containing
#             features for one task.
#         """
#         np.random.seed(CFG.seed + idx)
        
#         # Select 2 malware families + 1 benign
#         frauds = [c for c in self.classes if c != self.normal]
#         if len(frauds) < 2:
#             raise ValueError(
#                 "Not enough malware families in JSON for a 3-way task."
#             )

#         selected = (
#             np.random.choice(frauds, CFG.n_way - 1, replace=False).tolist()
#             + [self.normal]
#         )

#         task = []
#         for cls in selected:
#             files = self.data[cls]
#             need = self.k_shot + self.q_query
#             chosen = np.random.choice(files, need, replace=(len(files) < need))
            
#             cls_feats = []
#             for filepath in chosen:
#                 path = self._fix_path(filepath)
                
#                 try:
#                     arr = np.load(path)
#                     if arr.ndim > 1:
#                         arr = arr.flatten()
#                     # Per-sample Z-score normalization
#                     mu, std = arr.mean(), arr.std()
#                     arr = (arr - mu) / (std + 1e-6)
#                 except Exception:
#                     # Fallback to zero vector if loading fails
#                     arr = np.zeros(self.input_dim, dtype=np.float32)
                
#                 cls_feats.append(arr.astype(np.float32))
            
#             task.append(torch.tensor(np.stack(cls_feats), dtype=torch.float32))
        
#         return torch.stack(task)  # [n_way, k+q, feat]

#     def _fix_path(self, path: str) -> str:
#         """Attempt to fix file path by trying common prefixes.

#         Args:
#             path: Original file path.

#         Returns:
#             Corrected absolute path if file exists, original path otherwise.
#         """
#         if os.path.exists(path):
#             return path
        
#         for prefix in ["../", "../../", "./"]:
#             candidate = os.path.join(prefix, path)
#             if os.path.exists(candidate):
#                 return os.path.abspath(candidate)
        
#         return path

#     def __len__(self) -> int:
#         """Return virtual dataset length.

#         Returns:
#             Large fixed number (actual number of tasks controlled by epochs).
#         """
#         return 100000


# def make_loader(json_path: str, split: str) -> DataLoader:
#     """Create DataLoader for a given split.

#     Args:
#         json_path: Path to JSON file containing dataset structure.
#         split: Data split ('train', 'val', or 'test').

#     Returns:
#         DataLoader instance for the specified split.
#     """
#     dataset = MalwareDataset(
#         json_path, split, CFG.k_shot, CFG.q_query, CFG.input_dim
#     )
#     return DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=True,
#         num_workers=CFG.num_workers,
#         pin_memory=CFG.pin_memory
#     )


# # ============================================================================
# # MODEL
# # ============================================================================

# class MalwareHead(nn.Module):
#     """Lightweight MLP model for malware classification.
    
#     Architecture:
#         - LayerNorm(input_dim)
#         - Linear(input_dim, hidden) -> ReLU -> Dropout
#         - Linear(hidden, hidden) -> ReLU -> Dropout
#         - Linear(hidden, n_way)
    
#     The model supports both standard forward pass and functional forward pass
#     for meta-learning algorithms that need to work with temporary parameters.
#     """
    
#     def __init__(self, input_dim: int = 1280, hidden: int = 512,
#                  n_way: int = 3, p_drop: float = 0.3):
#         """Initialize model.

#         Args:
#             input_dim: Dimensionality of input features.
#             hidden: Hidden layer size.
#             n_way: Number of output classes.
#             p_drop: Dropout probability.
#         """
        
#         super().__init__()
#         self.norm = nn.LayerNorm(input_dim)
#         self.fc1 = nn.Linear(input_dim, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, n_way)
#         self.p_drop = p_drop

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Standard forward pass.

#         Args:
#             x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

#         Returns:
#             Logits tensor of shape (batch_size, n_way).
#         """
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
        
#         x = self.norm(x)
#         x = torch.relu(self.fc1(x))
#         x = nn.functional.dropout(x, p=self.p_drop, training=self.training)
#         x = torch.relu(self.fc2(x))
#         x = nn.functional.dropout(x, p=self.p_drop, training=self.training)
#         return self.fc3(x)

#     def ordered_params(self) -> List[torch.Tensor]:
#         """Return parameters in a stable order for functional forward.

#         Returns:
#             List of parameters: [norm.weight, norm.bias, fc1.weight, fc1.bias,
#             fc2.weight, fc2.bias, fc3.weight, fc3.bias]
#         """
#         return [
#             self.norm.weight,
#             self.norm.bias,
#             self.fc1.weight,
#             self.fc1.bias,
#             self.fc2.weight,
#             self.fc2.bias,
#             self.fc3.weight,
#             self.fc3.bias,
#         ]


# def functional_forward(model: MalwareHead, x: torch.Tensor,
#                       fast_params: List[torch.Tensor]) -> torch.Tensor:
#     """Functional forward pass using custom parameters.

#     This is used for meta-learning where we need to compute forward passes
#     with temporary parameters during inner loop adaptation.

#     Args:
#         model: Model instance (used for p_drop attribute).
#         x: Input tensor.
#         fast_params: List of parameter tensors in the order returned by
#             model.ordered_params().

#     Returns:
#         Logits tensor.
#     """
#     if x.dim() == 1:
#         x = x.unsqueeze(0)
    
#     param_idx = 0
    
#     # LayerNorm
#     w_ln, b_ln = fast_params[param_idx], fast_params[param_idx + 1]
#     param_idx += 2
#     x = nn.functional.layer_norm(x, (w_ln.shape[0],), w_ln, b_ln, eps=1e-5)
    
#     # FC1
#     w, b = fast_params[param_idx], fast_params[param_idx + 1]
#     param_idx += 2
#     x = nn.functional.linear(x, w, b)
#     x = nn.functional.relu(x)
#     x = nn.functional.dropout(x, p=model.p_drop, training=True)
    
#     # FC2
#     w, b = fast_params[param_idx], fast_params[param_idx + 1]
#     param_idx += 2
#     x = nn.functional.linear(x, w, b)
#     x = nn.functional.relu(x)
#     x = nn.functional.dropout(x, p=model.p_drop, training=True)
    
#     # FC3 (logits)
#     w, b = fast_params[param_idx], fast_params[param_idx + 1]
#     param_idx += 2
#     x = nn.functional.linear(x, w, b)
    
#     return x


# # ============================================================================
# # MI-MAML ALGORITHM
# # ============================================================================

# def maml_inner_adapt(model: MalwareHead, task_batch: torch.Tensor,
#                      n_way: int, k_shot: int, q_query: int,
#                      inner_steps: int, inner_base_lr: float, device: str,
#                      loss_fn: nn.Module = None) -> Tuple[torch.Tensor, float,
#                                                           List[torch.Tensor]]:
#     """Perform one MI-MAML inner adaptation for a single task.

#     MI-MAML algorithm (FOMAML variant):
#     1. Initialize fast_params = clone of base parameters
#     2. For K inner steps:
#        a. Compute loss on support set using fast_params
#        b. Update: fast_params = fast_params - α_t * grad
#        c. Use cyclic learning rate α_t
#     3. Compute query loss using adapted fast_params
#     4. Return gradients w.r.t. fast_params (approximate gradient w.r.t. base)

#     Args:
#         model: Model instance (used for ordered_params and functional_forward).
#         task_batch: Batch of one task, shape (1, n_way, k+q, feat_dim).
#         n_way: Number of classes.
#         k_shot: Number of support samples per class.
#         q_query: Number of query samples per class.
#         inner_steps: Number of inner loop steps.
#         inner_base_lr: Base learning rate for inner loop.
#         device: Implementing device.
#         loss_fn: Loss function (default: CrossEntropyLoss).

#     Returns:
#         Tuple of (query_loss, query_accuracy, gradients_wrt_fast).
#     """
#     if loss_fn is None:
#         loss_fn = nn.CrossEntropyLoss()

#     task = task_batch.squeeze(0).to(device)  # [n_way, k+q, feat]
#     support = task[:, :k_shot, :].reshape(n_way * k_shot, -1)
#     query = task[:, k_shot:, :].reshape(n_way * q_query, -1)

#     y_s = create_label(n_way, k_shot).to(device)
#     y_q = create_label(n_way, q_query).to(device)

#     # Initialize fast parameters from base
#     base_params = model.ordered_params()
#     fast = [p.clone().detach().requires_grad_(True) for p in base_params]

#     # Inner loop: adapt on support set
#     for step in range(inner_steps):
#         # Compute cyclic learning rate
#         lr_t = cyclic_inner_lr(inner_base_lr, step, inner_steps)
        
#         # Forward pass on support set
#         logits_s = functional_forward(model, support, fast)
#         loss_s = loss_fn(logits_s, y_s)
        
#         # Compute gradients
#         grads = torch.autograd.grad(loss_s, fast, create_graph=False)
        
#         # Update fast parameters: θ' ← θ' - lr_t * grad
#         fast = [
#             w - lr_t * g if g is not None else w
#             for w, g in zip(fast, grads)
#         ]
        
#         # Re-enable gradients for next step
#         fast = [w.detach().requires_grad_(True) for w in fast]

#     # Compute meta-loss on query set
#     logits_q = functional_forward(model, query, fast)
#     loss_q = loss_fn(logits_q, y_q)
#     acc_q = (logits_q.argmax(-1) == y_q).float().mean().item()

#     # Compute gradients w.r.t. fast parameters (approximate base gradients)
#     grads_q = torch.autograd.grad(loss_q, fast, create_graph=False)

#     return loss_q.detach(), acc_q, grads_q


# def accumulate_grads(model: MalwareHead,
#                      grads_accum: List[torch.Tensor]) -> None:
#     """Write accumulated gradients back into model parameters.

#     Args:
#         model: Model instance.
#         grads_accum: List of gradient tensors in same order as model.ordered_params().
#     """
#     model_params = model.ordered_params()
#     with torch.no_grad():
#         for param, grad in zip(model_params, grads_accum):
#             if param.grad is None:
#                 param.grad = grad.clone()
#             else:
#                 param.grad += grad


# # ============================================================================
# # TRAINING
# # ============================================================================

# def run_epoch(train: bool, model: MalwareHead, loader: DataLoader,
#               cfg: CFG) -> Dict[str, float]:
#     """Run one training or validation epoch.

#     Args:
#         train: Whether to train the model (True) or evaluate (False).
#         model: Model to train/evaluate.
#         loader: DataLoader for the dataset.
#         cfg: Configuration object.

#     Returns:
#         Dictionary of metrics (loss, acc, etc.).
#     """
#     device = cfg.device
#     loss_fn = nn.CrossEntropyLoss()
#     model.train() if train else model.eval()

#     # Get or create optimizer
#     meta_opt = getattr(run_epoch, "_opt", None)
#     if meta_opt is None:
#         meta_opt = torch.optim.Adam(model.parameters(), lr=cfg.meta_lr)
#         run_epoch._opt = meta_opt

#     episodes = cfg.train_episodes_per_epoch if train else cfg.val_episodes_per_epoch
#     iterator = iter(loader)

#     total_loss = 0.0
#     total_acc = 0.0
#     meta_batch_grads = None

#     for episode in tqdm(range(episodes), desc="Train" if train else "Val"):
#         try:
#             task = next(iterator)  # [1, n_way, k+q, feat]
#         except StopIteration:
#             iterator = iter(loader)
#             task = next(iterator)

#         # Perform inner adaptation for one task
#         q_loss, q_acc, grads_q = maml_inner_adapt(
#             model, task, cfg.n_way, cfg.k_shot, cfg.q_query,
#             cfg.inner_steps, cfg.inner_lr, device, loss_fn
#         )

#         total_loss += q_loss.item()
#         total_acc += q_acc

#         # Accumulate gradients for meta-batch
#         if meta_batch_grads is None:
#             meta_batch_grads = [g.clone() for g in grads_q]
#         else:
#             for i in range(len(meta_batch_grads)):
#                 meta_batch_grads[i] += grads_q[i]

#         # Perform meta-update every meta_batch_size tasks
#         if (episode + 1) % cfg.meta_batch_size == 0:
#             avg_grads = [g / cfg.meta_batch_size for g in meta_batch_grads]
#             meta_opt.zero_grad(set_to_none=True)
#             accumulate_grads(model, avg_grads)
#             meta_opt.step()
#             meta_batch_grads = None

#     # Handle remaining gradients if not evenly divisible
#     if meta_batch_grads is not None and train:
#         remaining = (episodes % cfg.meta_batch_size) or cfg.meta_batch_size
#         avg_grads = [g / remaining for g in meta_batch_grads]
#         meta_opt.zero_grad(set_to_none=True)
#         accumulate_grads(model, avg_grads)
#         meta_opt.step()

#     # Compute average metrics
#     avg_loss = total_loss / max(1, episodes)
#     avg_acc = total_acc / max(1, episodes)
    
#     return {"loss": float(avg_loss), "acc": float(avg_acc)}


# # ============================================================================
# # MAIN ENTRY POINT
# # ============================================================================

# def main() -> None:
#     """Main training loop for MI-MAML meta-learning."""
#     ensure_dir(CFG.log_dir)
#     print(
#         f"[MI-MAML] device={CFG.device}  "
#         f"n_way={CFG.n_way}  k_shot={CFG.k_shot}  q_query={CFG.q_query}"
#     )

#     # Initialize logger
#     logger = Logger(experiment_name="mi_maml")

#     # Create data loaders
#     if CFG.use_csv_loader:
#         # Use CSV-based dataloader (using helper functions from data_loader.py)
#         print(f"Using CSV-based dataloader (via data_loader.py helpers)")
#         print(f"Features dir: {CFG.features_dir}")
#         print(f"Split CSV: {CFG.split_csv_path}")
        
#         train_dataset = CSVBasedMalwareDataset(
#             features_dir=CFG.features_dir,
#             split_csv_path=CFG.split_csv_path,
#             split="train",
#             k_shot=CFG.k_shot,
#             q_query=CFG.q_query,
#             n_way=CFG.n_way,
#             input_dim=CFG.input_dim,
#             seed=CFG.seed
#         )
#         val_dataset = CSVBasedMalwareDataset(
#             features_dir=CFG.features_dir,
#             split_csv_path=CFG.split_csv_path,
#             split="val",
#             k_shot=CFG.k_shot,
#             q_query=CFG.q_query,
#             n_way=CFG.n_way,
#             input_dim=CFG.input_dim,
#             seed=CFG.seed
#         )
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=1,
#             shuffle=True,
#             num_workers=CFG.num_workers,
#             pin_memory=CFG.pin_memory
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=1,
#             shuffle=False,
#             num_workers=CFG.num_workers,
#             pin_memory=CFG.pin_memory
#         )
#     else:
#         # Use JSON-based dataloader (original method)
#         print(f"Using JSON-based dataloader from {CFG.data_json}")
#         train_loader = make_loader(CFG.data_json, "train")
#         val_loader = make_loader(CFG.data_json, "val")

#     # Initialize model
#     model = MalwareHead(CFG.input_dim, hidden=512, n_way=CFG.n_way).to(CFG.device)

#     # Training loop
#     best_val = -1.0
#     save_path = os.path.join(CFG.log_dir, f"mi_maml_{timestamp()}.pth")

#     for epoch in range(1, CFG.max_epoch + 1):
#         print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        
#         train_stats = run_epoch(train=True, model=model, loader=train_loader,
#                                cfg=CFG)
#         val_stats = run_epoch(train=False, model=model, loader=val_loader,
#                              cfg=CFG)

#         print(
#             f"Train: acc={train_stats['acc']*100:.2f}%  "
#             f"loss={train_stats['loss']:.4f}"
#         )
#         print(
#             f"Val  : acc={val_stats['acc']*100:.2f}%  "
#             f"loss={val_stats['loss']:.4f}"
#         )

#         # Log epoch statistics
#         logger.add(epoch, train_stats, val_stats)

#         # Save best model
#         if val_stats["acc"] > best_val:
#             best_val = val_stats["acc"]
#             cfg_dict = {
#                 k: v for k, v in CFG.__dict__.items()
#                 if not k.startswith("__") and isinstance(
#                     v, (int, float, str, bool, list)
#                 )
#             }
            
#             torch.save({
#                 "model_state_dict": model.state_dict(),
#                 "cfg": cfg_dict,
#                 "epoch": epoch,
#                 "val_acc": best_val,
#             }, save_path)
            
#             print(
#                 f" Saved best model: {save_path} "
#                 f"(val_acc={best_val*100:.2f}%)"
#             )

#     print(f"\n Training done. Logs saved at: {logger.path}")
#     print(f" Best model saved at: {save_path}")


# if __name__ == "__main__":
#     main()
# mi_maml.py
# -*- coding: utf-8 -*-
"""
MI-MAML meta-learning (few-shot, ZSL, GZSL).
Compatible with episode_loader + your colleague's create_dataloaders.
"""

import os, math, random, json
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# from data_loader import create_dataloaders
from eposide_loader import EpisodeIterator
from src.extraction.data_loader import create_dataloaders, get_label_to_idx
# ===========================
# CONFIG
# ===========================
class CFG:
    mode = "gzsl"     # "fewshot" | "zsl" | "gzsl"
    n_way = 3
    k_shot = 1
    q_query = 5
    input_dim = 1280
    inner_steps = 5
    inner_lr = 0.01
    meta_lr = 1e-3
    meta_batch_size = 8
    max_epoch = 50
    train_episodes_per_epoch = 200
    val_episodes_per_epoch = 60
    features_dir = "../MalVis_dataset_small/features"
    split_csv_path = "../MalVis_dataset_small/label_split.csv"
    log_dir = "logs_meta"
    seed = 42
    # Device selection with MPS support for MacBook
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
            else:
        device = "cpu"


# ===========================
# UTILITIES
# ===========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cyclic_inner_lr(base_lr, step, total_steps):
    return float(base_lr * 0.5 * (1 + math.cos(2 * math.pi * step / max(1, total_steps))))


# ===========================
# MODEL
# ===========================
class MLPHead(nn.Module):
    def __init__(self, dim=1280, hidden=512, n_way=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_way)
        self.drop = 0.3

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.drop, training=self.training)
        x = torch.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.drop, training=self.training)
        return self.fc3(x)

    def ordered_params(self):
        return [self.norm.weight, self.norm.bias,
                self.fc1.weight, self.fc1.bias,
                self.fc2.weight, self.fc2.bias,
                self.fc3.weight, self.fc3.bias]


def functional_forward(model, x, fast_params):
    i = 0
    w_ln, b_ln = fast_params[i], fast_params[i+1]; i += 2
    x = nn.functional.layer_norm(x, (w_ln.shape[0],), w_ln, b_ln)
    w, b = fast_params[i], fast_params[i+1]; i += 2
    x = torch.relu(nn.functional.linear(x, w, b))
    x = nn.functional.dropout(x, p=model.drop, training=True)
    w, b = fast_params[i], fast_params[i+1]; i += 2
    x = torch.relu(nn.functional.linear(x, w, b))
    x = nn.functional.dropout(x, p=model.drop, training=True)
    w, b = fast_params[i], fast_params[i+1]; i += 2
    return nn.functional.linear(x, w, b)


# ===========================
# MAML ADAPTATION
# ===========================
def maml_inner_adapt(model, episode, n_way, k_shot, q_query, cfg):
    loss_fn = nn.CrossEntropyLoss()
    device = cfg.device
    # 数据已经在 EpisodeIterator 中移动到 device 了
    support_x, query_x = episode.support_x, episode.query_x
    y_s, y_q = episode.support_y, episode.query_y
    fast = [p.clone().detach().requires_grad_(True) for p in model.ordered_params()]

    for step in range(cfg.inner_steps):
        lr_t = cyclic_inner_lr(cfg.inner_lr, step, cfg.inner_steps)
        logits_s = functional_forward(model, support_x, fast)
        loss_s = loss_fn(logits_s, y_s)
        grads = torch.autograd.grad(loss_s, fast, create_graph=False)
        fast = [w - lr_t * g for w, g in zip(fast, grads)]
        fast = [w.detach().requires_grad_(True) for w in fast]

    logits_q = functional_forward(model, query_x, fast)
    loss_q = loss_fn(logits_q, y_q)
    acc_q = (logits_q.argmax(-1) == y_q).float().mean().item()
    grads_q = torch.autograd.grad(loss_q, fast, create_graph=False)
    return loss_q.item(), acc_q, grads_q


def accumulate_grads(model, grads):
    params = model.ordered_params()
    with torch.no_grad():
        for p, g in zip(params, grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad += g


# ===========================
# TRAIN LOOP
# ===========================
def run_epoch(train, model, iterator, cfg):
    model.train() if train else model.eval()
    meta_opt = getattr(run_epoch, "_opt", None)
    if meta_opt is None:
        meta_opt = torch.optim.Adam(model.parameters(), lr=cfg.meta_lr)
        run_epoch._opt = meta_opt
    total_loss, total_acc, meta_batch = 0, 0, None

    # 获取 episodes 数量（如果是 EpisodeIterator）
    total_episodes = getattr(iterator, 'episodes_per_epoch', None)
    if total_episodes is None:
        # 如果不知道总数，使用 tqdm 但不显示总数
        pbar = tqdm(iterator, desc="Training" if train else "Validating", ncols=100)
    else:
        pbar = tqdm(iterator, total=total_episodes, desc="Training" if train else "Validating", ncols=100)

    episode_count = 0
    for epi in pbar:
        episode_count += 1
        loss_q, acc_q, grads = maml_inner_adapt(model, epi, cfg.n_way, cfg.k_shot, cfg.q_query, cfg)
        total_loss += loss_q
        total_acc += acc_q
        
        # 累积梯度
        if meta_batch is None:
            meta_batch = [g.clone() for g in grads]
        else:
            for j in range(len(meta_batch)):
                meta_batch[j] += grads[j]
        
        # Meta-batch 更新
        if train and episode_count % cfg.meta_batch_size == 0:
            avg = [g / cfg.meta_batch_size for g in meta_batch]
            meta_opt.zero_grad(set_to_none=True)
            accumulate_grads(model, avg)
            meta_opt.step()
            meta_batch = None
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_q:.4f}',
            'acc': f'{acc_q*100:.2f}%',
            'avg_loss': f'{total_loss/episode_count:.4f}',
            'avg_acc': f'{total_acc/episode_count*100:.2f}%'
        })

    # 处理最后一个不完整的 meta-batch（仅在训练时）
    if train and meta_batch is not None and episode_count > 0:
        remaining = episode_count % cfg.meta_batch_size
        if remaining > 0:
            avg = [g / remaining for g in meta_batch]
        meta_opt.zero_grad(set_to_none=True)
            accumulate_grads(model, avg)
        meta_opt.step()

    n = max(1, episode_count)
    return {"loss": total_loss / n, "acc": total_acc / n}


# ===========================
# MAIN
# ===========================
def main():
    set_seed(CFG.seed)
    ensure_dir(CFG.log_dir)
    print(f"[MODE={CFG.mode}] n_way={CFG.n_way} k_shot={CFG.k_shot}")
    print(f"Using device: {CFG.device}")
    if CFG.device == "mps":
        print("✓ MPS (Metal Performance Shaders) enabled for MacBook GPU acceleration")
    elif CFG.device == "cuda":
        print("✓ CUDA enabled")
    else:
        print("⚠ Using CPU (slower)")

    dls, label_to_idx = create_dataloaders(
        features_dir=CFG.features_dir,
        split_csv_path=CFG.split_csv_path,
        batch_size=256,
        val_ratio=0.1,
        test_ratio=0.1,
        generalized=False,
        num_workers=0
    )
    
    # 获取 benign 类别的 ID
    normal_class_id = label_to_idx.get("benign", None)
    if normal_class_id is None:
        print("⚠️ Warning: 'benign' label not found in label_to_idx")
        print(f"   Available labels: {label_to_idx}")
        # 尝试找可能的 benign（通常样本数最多的类别）
        normal_class_id = 2  # 假设
        print(f"   Using default normal_class_id={normal_class_id}")
    else:
        print(f"✓ Found benign class ID: {normal_class_id}")
        print(f"  Label mapping: {label_to_idx}")

    # episode builders - 传入 device 和 normal_class_id
    print(f"\nBuilding episode iterators with device={CFG.device}...")
    train_iter = EpisodeIterator(
        dls["train"], CFG.n_way, CFG.k_shot, CFG.q_query, 
        CFG.train_episodes_per_epoch, seed=CFG.seed, device=CFG.device, 
        normalize=True, normal_class_id=normal_class_id
    )
    val_seen = EpisodeIterator(
        dls["test_seen"], CFG.n_way, CFG.k_shot, CFG.q_query, 
        CFG.val_episodes_per_epoch, seed=CFG.seed, device=CFG.device, 
        normalize=True, normal_class_id=normal_class_id
    )
    # 对于 test_unseen，如果没有 benign，从 train 数据借用
    val_unseen = EpisodeIterator(
        dls["test_unseen"], CFG.n_way, CFG.k_shot, CFG.q_query, 
        CFG.val_episodes_per_epoch, seed=CFG.seed, device=CFG.device, 
        normalize=True, normal_class_id=normal_class_id,
        normal_class_loader=dls["train"]  # 从 train 借用 benign 样本
    )
    print("Episode iterators ready!\n")

    model = MLPHead(CFG.input_dim, 512, CFG.n_way).to(CFG.device)

    # Epoch 循环，使用 tqdm 显示 epoch 进度
    epoch_pbar = tqdm(range(1, CFG.max_epoch + 1), desc="Epochs", ncols=100)
    for ep in epoch_pbar:
        train_stats = run_epoch(True, model, iter(train_iter), CFG)
        seen_stats = run_epoch(False, model, iter(val_seen), CFG)
        unseen_stats = run_epoch(False, model, iter(val_unseen), CFG)
        
        H = 2 * (seen_stats["acc"] * unseen_stats["acc"]) / (seen_stats["acc"] + unseen_stats["acc"] + 1e-8)
        
        # 更新 epoch 进度条
        epoch_pbar.set_postfix({
            'Train': f'{train_stats["acc"]*100:.1f}%',
            'ZSL': f'{unseen_stats["acc"]*100:.1f}%',
            'GZSL_seen': f'{seen_stats["acc"]*100:.1f}%',
            'H': f'{H*100:.1f}%'
        })
        
        print(f"\nEpoch {ep:03d}/{CFG.max_epoch} | Train {train_stats['acc']*100:.1f}% | "
              f"ZSL {unseen_stats['acc']*100:.1f}% | "
              f"GZSL_seen {seen_stats['acc']*100:.1f}% | H={H*100:.1f}%")


if __name__ == "__main__":
    main()
