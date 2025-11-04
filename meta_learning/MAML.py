# python -m meta_learning.MAML
import os
print("Test")
import sys
import random
import json
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)

from dotenv import load_dotenv

# Add src to path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_path)

try:
    from extraction.data_loader import create_dataloaders
    from extraction.downloader import download_dataset
except ImportError as e:
    print(f"Import error: {e}")
    # Alternative import paths
    alternative_paths = [
        os.path.join(current_dir, '..', 'src'),
        os.path.join(current_dir, 'src'),
        'src'
    ]
    
    imported = False
    for alt_path in alternative_paths:
        try:
            if alt_path not in sys.path:
                sys.path.append(alt_path)
            from extraction.data_loader import create_dataloaders
            from extraction.downloader import download_dataset
            print(f"Successfully imported from path: {alt_path}")
            imported = True
            break
        except ImportError:
            continue
    
    if not imported:
        print("Unable to import required modules, please check project structure")
        sys.exit(1)

# CONFIGURATION
class CFG:
    """Configuration parameters for MAML meta-learning."""
    n_way = 3
    k_shot = 1  
    q_query = 15
    input_dim = 1280

    inner_lr = 0.01  
    meta_lr = 0.001
    inner_steps_train = 1  
    inner_steps_val = 1

    meta_batch_size = 16  
    max_epoch = 100  
    eval_batches = 20
    grad_clip = 10.0

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dataset_dir = '../dataset'
    log_dir = "logs"

    # Random seed for reproducibility
    random_seed = 42

if not hasattr(CFG, 'eps'):
    CFG.eps = 1e-6

# UTILITY FUNCTIONS
def set_random_seeds(seed: int = CFG.random_seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_label(n_way: int, num_per_class: int) -> torch.Tensor:
    """Create labels for N-way classification task."""
    return torch.arange(n_way).repeat_interleave(num_per_class).long()

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    return (torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy()).mean()

def calculate_metrics(preds: List[int], labels: List[int],
                     num_classes: int) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
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
    """Logger for training statistics and model checkpointing."""
    
    def __init__(self):
        """Initialize logger with file paths and empty log structure."""
        os.makedirs(CFG.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.path = os.path.join(CFG.log_dir, f"maml_{timestamp}.json")
        self.model_path = os.path.join(CFG.log_dir, f"maml_best_{timestamp}.pth")
        self.experiment_name = f"maml_experiment_{timestamp}"
        
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
        """Add epoch statistics to log file."""
        self.logs["epochs"].append({
            "epoch": epoch,
            "train": train,
            "val": val
        })
        
        with open(self.path, "w") as f:
            json.dump(self.logs, f, indent=2)

    def should_save_best(self, val_acc: float) -> bool:
        """Check if current model is the best so far."""
        if val_acc > self.best_val:
            self.best_val = val_acc
            self.best_epoch = len(self.logs["epochs"])
            return True
        return False

# DATASET WRAPPER - Using standardized dataloader
class MAMLDatasetWrapper(Dataset):
    """Wrapper to adapt the standardized dataset for MAML's episodic training."""
    
    def __init__(self, dataloader, n_way=3, k_shot=1, q_query=5, num_tasks=1000):
        """Initialize MAML dataset wrapper."""
        self.dataloader = dataloader
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks
        
        # Extract all data and organize by class
        self.class_data = self._organize_data_by_class()
        self.available_classes = list(self.class_data.keys())
        
        print(f"Available classes: {len(self.available_classes)}")
        print(f"Class distribution: {[(cls, len(samples)) for cls, samples in self.class_data.items()]}")
        
        # Check if we have enough classes
        if len(self.available_classes) < self.n_way:
            raise ValueError(f"Need at least {self.n_way} classes for {self.n_way}-way classification, but only found {len(self.available_classes)}")
        
        # Validate we have enough samples per class
        min_samples_needed = self.k_shot + self.q_query
        for cls, samples in self.class_data.items():
            if len(samples) < min_samples_needed:
                print(f"Warning: Class {cls} has only {len(samples)} samples, need {min_samples_needed}. Will use replacement sampling.")
    
    def _organize_data_by_class(self):
        """Organize dataset samples by class label"""
        class_data = {}
        
        for features, label in self.dataloader:
            label_item = label.item()
            if label_item not in class_data:
                class_data[label_item] = []
            class_data[label_item].append(features.squeeze(0))
        
        return class_data
    
    def _sample_task_classes(self):
        """Sample classes for an n-way task"""
        # Randomly sample n_way classes from available classes
        sampled_classes = np.random.choice(self.available_classes, self.n_way, replace=False)
        return list(sampled_classes)
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        """Generate a single N-way K-shot task"""
        # Set seed for reproducibility based on index
        np.random.seed(CFG.random_seed + idx)
        
        # Sample classes for this task
        task_classes = self._sample_task_classes()
        
        task_data = []
        
        for cls in task_classes:
            class_samples = self.class_data[cls]
            
            # Sample support + query samples
            total_needed = self.k_shot + self.q_query
            
            if len(class_samples) >= total_needed:
                selected_indices = np.random.choice(len(class_samples), total_needed, replace=False)
            else:
                # Sample with replacement if not enough samples
                selected_indices = np.random.choice(len(class_samples), total_needed, replace=True)
            
            selected_samples = [class_samples[i] for i in selected_indices]
            task_data.append(torch.stack(selected_samples))
        
        # Stack all class data: [n_way, k_shot + q_query, feature_dim]
        task_tensor = torch.stack(task_data)
        
        # Reshape to [n_way * (k_shot + q_query), feature_dim]
        task_tensor = task_tensor.view(-1, task_tensor.size(-1))
        
        return task_tensor

def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    """Get meta batch function adapted for standardized dataloader"""
    data = []
    for _ in range(meta_batch_size):
        try:
            task_data = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            task_data = next(iterator)
        
        # task_data shape: [1, n_way * (k_shot + q_query), feature_dim]
        # Remove the batch dimension
        task_data = task_data.squeeze(0)  # [n_way * (k_shot + q_query), feature_dim]
        data.append(task_data)
    
    return torch.stack(data).to(CFG.device), iterator

# MODEL
class MalwareClassifier(nn.Module):
    """Neural network for malware classification with functional forward pass for MAML."""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        """Initialize malware classifier."""
        super(MalwareClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.5),           
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # nn.Linear(hidden_dim//2, hidden_dim//4),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def functional_forward(self, x, params):
        """Functional forward pass using provided parameters."""
        for i, module in enumerate(self.network):
            module_name = f'network.{i}'
            if isinstance(module, nn.Linear):
                weight = params[f'{module_name}.weight']
                bias = params[f'{module_name}.bias']
                x = F.linear(x, weight, bias)
            elif isinstance(module, nn.LayerNorm):
                weight = params[f'{module_name}.weight']
                bias = params[f'{module_name}.bias']
                x = F.layer_norm(x, module.normalized_shape, weight, bias, module.eps)
            elif isinstance(module, nn.BatchNorm1d):
                weight = params[f'{module_name}.weight']
                bias = params[f'{module_name}.bias']
                x = F.batch_norm(
                    x, 
                    running_mean=module.running_mean,
                    running_var=module.running_var,
                    weight=weight,
                    bias=bias,
                    training=False, 
                    momentum=1.0
                )
            elif isinstance(module, nn.ReLU):
                x = F.relu(x)
            elif isinstance(module, nn.Dropout):
                x = F.dropout(x, p=module.p, training=False) 
        return x

class EarlyStopping:
    """Early stopping utility to halt training when validation accuracy plateaus."""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_acc = 0
        
    def __call__(self, val_acc):
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# MAML ALGORITHM
def maml_step_with_preds(model: nn.Module,
                         criterion: nn.Module,
                         task_batch_normalized: torch.Tensor,
                         train: bool
                         ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Perform a single MAML step (inner and outer loop) for one task.
    """
    
    # task_batch_normalized Shape: [1, N*(K+Q), D]
    task_tensor = task_batch_normalized.squeeze(0) # Shape: [N*(K+Q), D]

    support_indices = []
    query_indices = []
    for n in range(CFG.n_way):
        class_start_idx = n * (CFG.k_shot + CFG.q_query)
        # K-shot support
        support_indices.extend(range(class_start_idx, class_start_idx + CFG.k_shot))
        # Q-query query
        query_indices.extend(range(class_start_idx + CFG.k_shot, class_start_idx + CFG.k_shot + CFG.q_query))
    
    support_x = task_tensor[support_indices] # [N*K, D]
    query_x = task_tensor[query_indices]     # [N*Q, D]

    # Create labels
    support_y = torch.arange(CFG.n_way).repeat_interleave(CFG.k_shot).to(CFG.device) # [N*K]
    query_y = torch.arange(CFG.n_way).repeat_interleave(CFG.q_query).to(CFG.device)   # [N*Q]
    
    # Initialize fast weights (θ') for inner loop
    fast_weights = OrderedDict(model.named_parameters())
    inner_steps = CFG.inner_steps_train if train else CFG.inner_steps_val
    
    for inner_step in range(inner_steps):
        # Calculate support logits and loss with current fast weights
        support_logits = model.functional_forward(support_x, fast_weights)
        support_loss = criterion(support_logits, support_y)
        
        # Calculate gradients w.r.t. fast weights
        grads = torch.autograd.grad(support_loss, 
                                    fast_weights.values(), 
                                    create_graph=False) # <--- FO-MAML

        # Update fast weights (θ')
        fast_weights = OrderedDict(
            (name, param - CFG.inner_lr * grad)
            for ((name, param), grad) in zip(fast_weights.items(), grads)
        )

    # Calculate query logits and loss with adapted fast weights
    query_logits = model.functional_forward(query_x, fast_weights)
    query_loss = criterion(query_logits, query_y)
    meta_loss = query_loss 
    
    # Calculate task accuracy
    with torch.no_grad():
        preds = torch.argmax(query_logits, dim=1)
        labels = query_y
        task_acc = (preds == labels).sum().item() / len(labels)

    # FO-MAML Outer Loop Update
    if train:
        # Calculate meta gradients w.r.t. original model parameters (θ)
        meta_grads = torch.autograd.grad(meta_loss, 
                                         fast_weights.values())

        # Get original model parameters (θ)
        params = model.parameters()

        # Accumulate gradients into model parameters (θ)
        for param, meta_grad in zip(params, meta_grads):
            if param.grad is None:
                param.grad = meta_grad.detach()
            else:
                # Accumulate gradients
                param.grad += meta_grad.detach()

    return meta_loss, task_acc, preds.detach().cpu(), labels.detach().cpu()

def get_task_predictions(model, task_tensor, loss_fn, cfg, inner_steps):
    """
    Get predictions for the query set after inner loop adaptation.
    """
    
    # Task tensor shape: [1, N*(K+Q), D]
    task_tensor = task_tensor.view(
        cfg.n_way, cfg.k_shot + cfg.q_query, -1
    ).squeeze(0) # [N_way, K+Q, D] -> [N*(K+Q), D]

    # Split Features (x)
    support_x = task_tensor[:, :cfg.k_shot, :].contiguous().view(-1, cfg.input_dim)
    query_x = task_tensor[:, cfg.k_shot:, :].contiguous().view(-1, cfg.input_dim)

    # Generate Labels (y)
    support_y = create_label(cfg.n_way, cfg.k_shot).to(cfg.device)
    query_y = create_label(cfg.n_way, cfg.q_query).to(cfg.device)

    # Initialize fast weights (θ') for inner loop
    fast_weights = OrderedDict()
    for name, param in model.named_parameters():
            if param.requires_grad:  
                fast_weights[name] = param.clone()

    # Update fast weights through inner loop
    current_fast_weights = fast_weights
    
    for k in range(inner_steps):
        support_logits = model.functional_forward(support_x, current_fast_weights)
        support_loss = loss_fn(support_logits, support_y.long())
        
        grad = torch.autograd.grad(
            support_loss, 
            current_fast_weights.values(), 
            create_graph=False, 
            # allow_unused=True
        )

        # Update fast weights (θ')
        current_fast_weights = OrderedDict(
            (name, param - cfg.inner_lr * g if g is not None else param)
            for ((name, param), g) in zip(current_fast_weights.items(), grad)
        )

    # Outer Loop Prediction on Query Set
    model.eval()
    with torch.no_grad():
        query_logits = model.functional_forward(query_x, current_fast_weights)
        query_pred = torch.argmax(query_logits, dim=1).cpu().numpy()
        
    # Return predictions and true labels
    return query_pred.tolist(), query_y.cpu().numpy().tolist()

def run_epoch(
    model: nn.Module,
    meta_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    mode: str,
    train: bool = True
) -> Dict[str, Any]:
    """
    Run a single epoch of MAML training or evaluation.
    """
    
    if train:
        meta_optimizer.zero_grad()

    epoch_meta_loss = 0.0
    all_preds = []
    all_labels = []
    
    try:
        diag_batch = next(iter(dataloader))
        # print(f"  Type of diag_batch: {type(diag_batch)}")
        
        support_x = None
        
        if isinstance(diag_batch, (list, tuple)):
            # print(f"  Length of batch: {len(diag_batch)}")
            if len(diag_batch) == 5:
                support_x = diag_batch[0].to(CFG.device)
            elif len(diag_batch) > 0:
                support_x = diag_batch[0].to(CFG.device)
            else:
                print("Error: Batch is empty.")
        else:
            # print("  Batch is not a list/tuple. Assuming it's the tensor itself.")
            support_x = diag_batch.to(CFG.device)

    except Exception as e:
        print(f"Error: Data diagnostics failed: {e}")
    print("="*30 + "\n")

    # Iterate tqdm for progress bar
    epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.max_epoch} [{mode}]", leave=False)
    
    for i, task_batch in enumerate(epoch_iterator):
        # Set task batch to device
        task_batch = task_batch.to(CFG.device) # Shape [1, N*(K+Q), D]
        
        # Check for NaN/Inf in raw task batch before normalization
        if torch.isnan(task_batch).any() or torch.isinf(task_batch).any():
            if hasattr(CFG, 'verbose') and CFG.verbose:
                print(f"\nWarning: Skipping task {i} due to NaN/Inf in raw data.")
            continue

        # Squeeze to [N*(K+Q), D] for normalization
        task_tensor_raw = task_batch.squeeze(0)

        # Get Support and Query indices
        support_indices = []
        query_indices = []
        for n in range(CFG.n_way):
            class_start_idx = n * (CFG.k_shot + CFG.q_query)
            support_indices.extend(range(class_start_idx, class_start_idx + CFG.k_shot))
            query_indices.extend(range(class_start_idx + CFG.k_shot, class_start_idx + CFG.k_shot + CFG.q_query))

        support_x_raw = task_tensor_raw[support_indices] # [N*K, D]
        query_x_raw = task_tensor_raw[query_indices]     # [N*Q, D]
        
        # Calculate mean and std from Support set
        mean = torch.mean(support_x_raw, dim=0) # Shape: [D]
        std = torch.std(support_x_raw, dim=0)   # Shape: [D]
        
        # Normalize Support and Query sets
        support_x_norm = (support_x_raw - mean) / (std + CFG.eps)
        query_x_norm = (query_x_raw - mean) / (std + CFG.eps)
        
        # Replace NaN values resulted from zero std with zeros
        support_x_norm = torch.nan_to_num(support_x_norm, nan=0.0)
        query_x_norm = torch.nan_to_num(query_x_norm, nan=0.0)

        # Combine back into a single task tensor
        task_tensor_normalized = torch.zeros_like(task_tensor_raw)
        
        # Replace support and query parts with normalized data
        task_tensor_normalized[support_indices] = support_x_norm
        task_tensor_normalized[query_indices] = query_x_norm
        task_batch_normalized = task_tensor_normalized.unsqueeze(0)
        
        # Invoke MAML step
        meta_loss, task_acc, preds, labels = maml_step_with_preds(
            model, criterion, task_batch_normalized, train=train
        )
        
        # Convert meta_loss to float for accumulation
        meta_loss_float = meta_loss.item()

        if np.isnan(meta_loss_float):
            print(f"\nWarning: Detected 'nan' loss in epoch {epoch+1} [{mode}] at task {i}. Stopping epoch early.")
            epoch_meta_loss = np.nan
            break 
        
        epoch_meta_loss += meta_loss_float
        all_preds.extend(preds)
        all_labels.extend(labels)

        if train and (i + 1) % CFG.meta_batch_size == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.grad_clip)
            meta_optimizer.step()
            meta_optimizer.zero_grad()
        
        epoch_iterator.set_postfix(
            meta_loss=f"{epoch_meta_loss / (i + 1):.4f}", 
            task_acc=f"{task_acc:.4f}"
        )

    # Deal with remaining gradients if any
    if train and (i + 1) % CFG.meta_batch_size != 0 and not np.isnan(epoch_meta_loss):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.grad_clip)
        meta_optimizer.step()
        meta_optimizer.zero_grad()

    # Calculate average epoch loss
    if np.isnan(epoch_meta_loss):
        avg_epoch_loss = np.nan
    else:
        avg_epoch_loss = epoch_meta_loss / len(dataloader)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    if len(all_labels) == 0 or len(all_preds) == 0:
        print(f"Warning: No labels or predictions found for epoch {epoch+1} [{mode}]. Skipping metrics.")
        return {
            "loss": float(avg_epoch_loss),
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "confusion_matrix": None
        }

    # Calculate metrics
    avg_accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    all_labels_int = all_labels.astype(int)
    all_preds_int = all_preds.astype(int)
    
    cm = confusion_matrix(all_labels_int, all_preds_int)

    return {
        "loss": float(avg_epoch_loss),
        "accuracy": float(avg_accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "confusion_matrix": cm.tolist() if cm is not None else None
    }

def test_model(model, test_loader, num_test_tasks=20):
    """Test the trained model and return detailed results."""
    test_iter = iter(test_loader)
    
    all_predicted_labels = []
    all_true_labels = []
    task_accuracies = []

    print("Starting testing (collecting predictions for detailed report)...")
    device = CFG.device 

    for batch_idx in tqdm(range(num_test_tasks), desc="Testing"):
        x, test_iter = get_meta_batch(1, CFG.k_shot, CFG.q_query, test_loader, test_iter)
        
        # x shape: [1, N*(K+Q), D]
        task_tensor_raw = x[0] # [N*(K+Q), D]
   
        task_tensor_raw = task_tensor_raw.to(device)
        
        if torch.isnan(task_tensor_raw).any() or torch.isinf(task_tensor_raw).any():
            if hasattr(CFG, 'verbose') and CFG.verbose:
                print(f"\nWarning: Skipping test task {batch_idx} due to NaN/Inf in raw data.")
            continue
            
        support_indices = []
        query_indices = []
        for n in range(CFG.n_way):
            class_start_idx = n * (CFG.k_shot + CFG.q_query)
            support_indices.extend(range(class_start_idx, class_start_idx + CFG.k_shot))
            query_indices.extend(range(class_start_idx + CFG.k_shot, class_start_idx + CFG.k_shot + CFG.q_query))
        
        support_x_raw = task_tensor_raw[support_indices]
        query_x_raw = task_tensor_raw[query_indices]

        mean = torch.mean(support_x_raw, dim=0)
        std = torch.std(support_x_raw, dim=0)

        support_x_norm = (support_x_raw - mean) / (std + CFG.eps)
        query_x_norm = (query_x_raw - mean) / (std + CFG.eps)
        
        support_x_norm = torch.nan_to_num(support_x_norm, nan=0.0)
        query_x_norm = torch.nan_to_num(query_x_norm, nan=0.0)

        task_tensor_normalized = torch.zeros_like(task_tensor_raw)
        task_tensor_normalized[support_indices] = support_x_norm
        task_tensor_normalized[query_indices] = query_x_norm
        
        predicted_labels, true_labels = get_task_predictions(
            model,
            task_tensor_normalized, 
            nn.CrossEntropyLoss(),
            CFG,
            CFG.inner_steps_val
        )

        task_true = np.array(true_labels)
        task_pred = np.array(predicted_labels)
        task_acc = (task_true == task_pred).mean()
        task_accuracies.append(task_acc)

        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(true_labels)

    return all_predicted_labels, all_true_labels, task_accuracies

def main():
    """Main training loop for MAML meta-learning."""
    print(f"🔥 MAML Meta-Learning for Malware Classification (FIXED VERSION)")
    print(f"Using device: {CFG.device}")
    
    # Set random seeds
    set_random_seeds()
    
    # Load environment variables
    load_dotenv()

    # Download dataset if not exists
    if not os.path.exists(CFG.dataset_dir):
        download_dataset(dir=CFG.dataset_dir)

    features_dir = os.path.join(CFG.dataset_dir, 'features')
    split_csv_path = os.path.join(CFG.dataset_dir, 'label_split.csv')

    # Create standardized dataloaders
    dataloaders = create_dataloaders(
        features_dir=features_dir,
        split_csv_path=split_csv_path,
        batch_size=1,  
        val_ratio=0.1,
        test_ratio=0.1,
        generalized=False,
        num_workers=0  # Set to 0 for compatibility
    )

    print(f"Available dataloaders: {list(dataloaders.keys())}")
    print(f"Train dataset size: {len(dataloaders['train'].dataset)}")
    print(f"Validation dataset size: {len(dataloaders['val'].dataset)}")
    print(f"Test unseen dataset size: {len(dataloaders['test_unseen'].dataset)}")

    # Check available classes and adjust n_way if needed
    temp_loader = dataloaders['train']
    temp_classes = set()
    sample_count = 0
    for _, label in temp_loader:
        temp_classes.add(label.item())
        sample_count += 1
        if sample_count >= 100:  # Sample enough to get all classes
            break
    actual_num_classes = len(temp_classes)
    print(f"Actual number of classes found: {actual_num_classes}")
    print(f"Available classes: {sorted(list(temp_classes))}")

    # Adjust n_way to fit available classes
    if actual_num_classes < CFG.n_way:
        print(f"WARNING: Only {actual_num_classes} classes available, reducing n_way from {CFG.n_way} to {actual_num_classes}")
        CFG.n_way = actual_num_classes
    print(f"Using {CFG.n_way}-way classification")

    # Create MAML-compatible datasets from standardized dataloaders
    train_maml_dataset = MAMLDatasetWrapper(
        dataloaders['train'], 
        n_way=CFG.n_way, 
        k_shot=CFG.k_shot, 
        q_query=CFG.q_query, 
        num_tasks=500  # Reduced for faster training
    )

    val_maml_dataset = MAMLDatasetWrapper(
        dataloaders['val'], 
        n_way=CFG.n_way, 
        k_shot=CFG.k_shot, 
        q_query=CFG.q_query, 
        num_tasks=100  # Reduced for faster validation
    )

    test_maml_dataset = MAMLDatasetWrapper(
        dataloaders['test_unseen'], 
        n_way=CFG.n_way, 
        k_shot=CFG.k_shot, 
        q_query=CFG.q_query, 
        num_tasks=200
    )

    # Create DataLoaders for MAML
    train_loader = DataLoader(train_maml_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_maml_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_maml_dataset, batch_size=1, shuffle=False, num_workers=0)

    sample_task = next(iter(train_loader))
    actual_input_dim = sample_task.shape[-1]
    
    
    if actual_input_dim != CFG.input_dim:
        CFG.input_dim = actual_input_dim

    # Initialize model, optimizer, and logger
    model = MalwareClassifier(CFG.input_dim, output_dim=CFG.n_way).to(CFG.device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=CFG.meta_lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss().to(CFG.device)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    # Create iterators
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Early stopping setup
    early_stopping = EarlyStopping(patience=15, min_delta=0.01)
    initial_inner_lr = CFG.inner_lr
    best_val_acc = 0
    no_improve_epochs = 0

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        meta_optimizer, mode='max', factor=0.8, patience=10
    )

    # Training loop
    print("\n" + "="*60)
    print("STARTING MAML TRAINING")
    print("="*60)

    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        
        # Train and validate
        train_metrics = run_epoch(
            model=model,
            meta_optimizer=meta_optimizer,
            criterion=criterion,
            dataloader=train_loader,
            epoch=epoch,
            mode="Train", 
            train=True
        )

        # Log training metrics
        train_loss = train_metrics["loss"]
        train_acc = train_metrics["accuracy"]
        train_f1 = train_metrics["f1_macro"]
        val_metrics = run_epoch(
            model=model,
            meta_optimizer=meta_optimizer, 
            criterion=criterion,
            dataloader=val_loader,
            epoch=epoch,
            mode="Validation", 
            train=False
        )
        
        # Log validation metrics
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_macro"]

        # Print progress with more detailed info 
        print(f"Train Loss: {train_metrics['loss']:.3f} | Train Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.3f} | Val Acc: {val_metrics['accuracy']*100:.2f}%")
        print(f"Train F1: {train_metrics.get('f1_macro', 0):.3f} | Val F1: {val_metrics.get('f1_macro', 0):.3f}")
        
        # Adjust inner learning rate every 20 epochs
        if epoch % 20 == 0 and epoch > 0:
            CFG.inner_lr = max(CFG.inner_lr * 0.8, 0.001)
            # print(f"Reduced inner learning rate to: {CFG.inner_lr:.4f}")

        # Check for overfitting
        acc_gap = train_metrics['accuracy'] - val_metrics['accuracy']
        if acc_gap > 0.3:
            print(f"Warning: Possible overfitting! Gap: {acc_gap*100:.1f}%")

        # Update learning rate scheduler
        scheduler.step(val_metrics['accuracy'])
        
        # Log metrics
        logger.add(epoch, train_metrics, val_metrics)
        
        # Save best model
        if logger.should_save_best(val_metrics['accuracy']):
            best_val_acc = val_metrics['accuracy']
            no_improve_epochs = 0
            
            cfg_dict = {
                k: v for k, v in vars(CFG).items()
                if not k.startswith("__") and isinstance(
                    v, (int, float, str, bool, list, type(None))
                )
            }
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": meta_optimizer.state_dict(),
                "cfg": cfg_dict,
                "epoch": epoch,
                "val_accuracy": val_metrics['accuracy'],
                "hyperparameters": {
                    'n_way': CFG.n_way,
                    'k_shot': CFG.k_shot,
                    'q_query': CFG.q_query,
                    'input_dim': CFG.input_dim,
                    'inner_lr': CFG.inner_lr,
                    'meta_lr': CFG.meta_lr
                },
            }, logger.model_path)
            
            print(f"Saved best model: {logger.model_path} "
                f"(val_accuracy={val_metrics['accuracy']*100:.2f}%)")
        else:
            no_improve_epochs += 1
        
        # Early stopping check
        if early_stopping(val_metrics['accuracy']):
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {early_stopping.best_val_acc*100:.2f}%")
            break
        
        # Hard stop after 25 epochs of no improvement
        if no_improve_epochs >= 25:
            print(f"Stopping: No improvement for {no_improve_epochs} epochs")
            break

    # Testing phase
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    test_predicted_labels, test_true_labels, test_task_accuracies = test_model(
        model, test_loader, num_test_tasks=20
    )
    
    average_test_accuracy = np.mean(test_task_accuracies)
    print(f"Average Test Task Accuracy: {average_test_accuracy*100:.3f}%")
    print(f"Total test samples: {len(test_predicted_labels)}")

    # Save test results
    import pandas as pd
    results_df = pd.DataFrame({
        'id': range(len(test_predicted_labels)),
        'predicted_class': test_predicted_labels,
        'true_class': test_true_labels
    })

    results_csv_path = os.path.join(CFG.log_dir, 'maml_test_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Test results saved as {results_csv_path}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_true_labels, test_predicted_labels, 
                              target_names=[f'Class_{i}' for i in range(CFG.n_way)]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_true_labels, test_predicted_labels))

    # Training complete
    print("MAML Training completed!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Logs saved at: {logger.path}")
    print(f"Best model saved at: {logger.model_path}")
    print(f"Test results saved at: {results_csv_path}")


if __name__ == "__main__":
    main()