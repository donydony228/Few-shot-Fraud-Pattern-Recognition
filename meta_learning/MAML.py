#!/usr/bin/env python3
"""
MAML for Few-shot Fraud Pattern Recognition - GPU Optimized Version
Optimized for NYU HPC Cloud Bursting with checkpointing and memory management
"""

import os
import sys
import random
import json
import argparse
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
    classification_report
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
    print("Trying alternative import paths...")
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


# ============================================================================
# CONFIGURATION
# ============================================================================

class CFG:
    """Configuration parameters for MAML meta-learning."""
    n_way = 3
    k_shot = 1
    q_query = 5
    input_dim = 1280

    inner_lr = 0.001
    meta_lr = 0.001
    inner_steps_train = 5
    inner_steps_val = 5

    meta_batch_size = 8  # Optimized for GPU memory
    max_epoch = 100
    eval_batches = 20
    
    # GPU-optimized device selection
    if torch.cuda.is_available():
        device = "cuda"
        # Enable memory optimization for GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        device = "cpu"

    dataset_dir = '/scratch/$USER/dataset'  # Use scratch for better I/O
    log_dir = '/scratch/$USER/logs'
    checkpoint_dir = '/scratch/$USER/checkpoints'

    # Random seed for reproducibility
    random_seed = 42
    
    # Checkpointing settings
    save_every_epochs = 10
    resume_from_checkpoint = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    """Logger with checkpointing support for training statistics."""
    
    def __init__(self):
        """Initialize logger with file paths and empty log structure."""
        os.makedirs(CFG.log_dir, exist_ok=True)
        os.makedirs(CFG.checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.path = os.path.join(CFG.log_dir, f"maml_{timestamp}.json")
        self.model_path = os.path.join(CFG.log_dir, f"maml_best_{timestamp}.pth")
        self.checkpoint_path = os.path.join(CFG.checkpoint_dir, f"maml_checkpoint_{timestamp}.pth")
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
    
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       train_iter_state: int, val_iter_state: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "logs": self.logs,
            "train_iter_state": train_iter_state,
            "val_iter_state": val_iter_state,
            "random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f"✅ Checkpoint saved: {self.checkpoint_path}")


# ============================================================================
# DATASET WRAPPER
# ============================================================================

class MAMLDatasetWrapper(Dataset):
    """Memory-efficient wrapper for MAML's episodic training."""
    
    def __init__(self, dataloader, n_way=3, k_shot=1, q_query=5, num_tasks=1000):
        """Initialize MAML dataset wrapper with memory optimization."""
        self.dataloader = dataloader
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks
        
        # Extract all data and organize by class (memory optimized)
        self.class_data = self._organize_data_by_class()
        self.available_classes = list(self.class_data.keys())
        
        print(f"Available classes: {len(self.available_classes)}")
        print(f"Class distribution: {[(cls, len(samples)) for cls, samples in self.class_data.items()]}")
        
        if len(self.available_classes) < self.n_way:
            raise ValueError(f"Need at least {self.n_way} classes for {self.n_way}-way classification")
        
        # Validate sample requirements
        min_samples_needed = self.k_shot + self.q_query
        for cls, samples in self.class_data.items():
            if len(samples) < min_samples_needed:
                print(f"Warning: Class {cls} has only {len(samples)} samples, need {min_samples_needed}")
    
    def _organize_data_by_class(self):
        """Organize dataset samples by class label with memory optimization."""
        class_data = {}
        
        # Use iterator to avoid loading all data at once
        for features, label in self.dataloader:
            label_item = label.item()
            if label_item not in class_data:
                class_data[label_item] = []
            # Move to CPU to save GPU memory
            class_data[label_item].append(features.squeeze(0).cpu())
        
        return class_data
    
    def _sample_task_classes(self):
        """Sample classes for an n-way task."""
        return np.random.choice(self.available_classes, self.n_way, replace=False).tolist()
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        """Generate a single N-way K-shot task."""
        np.random.seed(CFG.random_seed + idx)
        
        task_classes = self._sample_task_classes()
        task_data = []
        
        for cls in task_classes:
            class_samples = self.class_data[cls]
            total_needed = self.k_shot + self.q_query
            
            if len(class_samples) >= total_needed:
                selected_indices = np.random.choice(len(class_samples), total_needed, replace=False)
            else:
                selected_indices = np.random.choice(len(class_samples), total_needed, replace=True)
            
            selected_samples = [class_samples[i] for i in selected_indices]
            task_data.append(torch.stack(selected_samples))
        
        # Stack and return tensor
        task_tensor = torch.stack(task_data)
        return task_tensor.view(-1, task_tensor.size(-1))


def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    """Get meta batch with memory management."""
    data = []
    for _ in range(meta_batch_size):
        try:
            task_data = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            task_data = next(iterator)
        
        # Move to device efficiently
        task_data = task_data.squeeze(0).to(CFG.device, non_blocking=True)
        data.append(task_data)
    
    return torch.stack(data), iterator


# ============================================================================
# MODEL
# ============================================================================

class MalwareClassifier(nn.Module):
    """GPU-optimized neural network for malware classification."""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(MalwareClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def functional_forward(self, x, params):
        """Functional forward pass using provided parameters for MAML."""
        for i, (name, module) in enumerate(self.network.named_children()):
            if isinstance(module, nn.Linear):
                weight_key = f'network.{i}.weight'
                bias_key = f'network.{i}.bias'
                
                x = F.linear(x, params.get(weight_key, module.weight), 
                           params.get(bias_key, module.bias))
            elif isinstance(module, nn.BatchNorm1d):
                # Use running stats for batch norm
                x = F.batch_norm(x, module.running_mean, module.running_var,
                               module.weight, module.bias, training=False)
            elif isinstance(module, nn.ReLU):
                x = F.relu(x)
            elif isinstance(module, nn.Dropout):
                x = F.dropout(x, p=0.3, training=self.training)
        return x


# ============================================================================
# MAML ALGORITHM
# ============================================================================

def maml_step(model, optimizer, x, n_way, k_shot, q_query, loss_fn, 
              inner_train_step, inner_lr, train, return_labels=False):
    """GPU-optimized MAML algorithm implementation."""
    criterion = loss_fn
    task_loss = []
    task_acc = []
    labels = []
    
    for meta_batch in x:
        # Split support and query sets
        support_set = meta_batch[: n_way * k_shot]
        query_set = meta_batch[n_way * k_shot :]

        # Copy the params for inner loop
        fast_weights = OrderedDict(model.named_parameters())

        ### ---------- INNER TRAIN LOOP ---------- ###
        for inner_step in range(inner_train_step):
            train_label = create_label(n_way, k_shot).to(CFG.device, non_blocking=True)
            logits = model.functional_forward(support_set, fast_weights)
            loss = criterion(logits, train_label)
            
            # Calculate gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            # Update fast_weights
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )

        ### ---------- INNER VALID LOOP ---------- ###
        if not return_labels:
            val_label = create_label(n_way, q_query).to(CFG.device, non_blocking=True)
            logits = model.functional_forward(query_set, fast_weights)
            loss = criterion(logits, val_label)
            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, val_label))
        else:
            logits = model.functional_forward(query_set, fast_weights)
            labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    # Update outer loop
    model.train()
    optimizer.zero_grad()

    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        meta_batch_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc


# ============================================================================
# TRAINING
# ============================================================================

def run_epoch(model, optimizer, loader, iterator, loss_fn, train=True):
    """Run one training or validation epoch with memory optimization."""
    losses = []
    preds = []
    labels = []
    
    num_batches = len(loader) // CFG.meta_batch_size if train else CFG.eval_batches
    inner_steps = CFG.inner_steps_train if train else CFG.inner_steps_val

    with torch.cuda.device(CFG.device) if torch.cuda.is_available() else torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Train" if train else "Val"):
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (_ % 10 == 0):
                torch.cuda.empty_cache()
                
            x, iterator = get_meta_batch(
                CFG.meta_batch_size, CFG.k_shot, CFG.q_query, loader, iterator
            )
            
            loss, acc = maml_step(
                model, optimizer, x, CFG.n_way, CFG.k_shot, CFG.q_query, loss_fn,
                inner_train_step=inner_steps, inner_lr=CFG.inner_lr, train=train,
            )
            
            losses.append(loss.item())
            
            if not train:
                pred_labels = maml_step(
                    model, optimizer, x, CFG.n_way, CFG.k_shot, CFG.q_query, loss_fn,
                    inner_train_step=inner_steps, inner_lr=CFG.inner_lr, train=False,
                    return_labels=True,
                )
                
                true_labels = []
                for _ in range(CFG.meta_batch_size):
                    for class_idx in range(CFG.n_way):
                        true_labels.extend([class_idx] * CFG.q_query)
                
                preds.extend(pred_labels)
                labels.extend(true_labels)
    
    # Compute metrics
    if train:
        metrics = {
            "loss": float(np.mean(losses)),
            "acc": acc,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "cm": []
        }
    else:
        if len(preds) > 0 and len(labels) > 0:
            metrics = calculate_metrics(preds, labels, CFG.n_way)
            metrics["loss"] = float(np.mean(losses))
        else:
            metrics = {
                "loss": float(np.mean(losses)),
                "acc": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "cm": []
            }
    
    return metrics, iterator


def test_model(model, test_loader, num_test_tasks=20):
    """Test the trained model with GPU optimization."""
    test_iter = iter(test_loader)
    
    all_predicted_labels = []
    all_true_labels = []
    task_accuracies = []

    print("Starting testing...")

    with torch.no_grad():
        for batch_idx in tqdm(range(num_test_tasks), desc="Testing"):
            x, test_iter = get_meta_batch(1, CFG.k_shot, CFG.q_query, test_loader, test_iter)

            task_true_labels = []
            for class_idx in range(CFG.n_way):
                task_true_labels.extend([class_idx] * CFG.q_query)

            predicted_labels = maml_step(
                model, None, x, CFG.n_way, CFG.k_shot, CFG.q_query, nn.CrossEntropyLoss(),
                inner_train_step=CFG.inner_steps_val, inner_lr=CFG.inner_lr, train=False,
                return_labels=True,
            )

            task_true = np.array(task_true_labels)
            task_pred = np.array(predicted_labels)
            task_acc = (task_true == task_pred).mean()
            task_accuracies.append(task_acc)

            all_predicted_labels.extend(predicted_labels)
            all_true_labels.extend(task_true_labels)

    return all_predicted_labels, all_true_labels, task_accuracies


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, logger: Logger):
    """Load training checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, iter([]), iter([])
    
    checkpoint = torch.load(checkpoint_path, map_location=CFG.device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.best_val = checkpoint["best_val"]
    logger.best_epoch = checkpoint["best_epoch"]
    logger.logs = checkpoint["logs"]
    
    # Restore random states
    torch.set_rng_state(checkpoint["random_state"])
    if torch.cuda.is_available() and checkpoint["cuda_random_state"] is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    random.setstate(checkpoint["python_random_state"])
    
    print(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"], None, None  # Iterator states would need to be reconstructed


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main training loop for MAML meta-learning with GPU optimization."""
    parser = argparse.ArgumentParser(description='MAML Training on GPU')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=CFG.max_epoch, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=CFG.meta_batch_size, help='Meta batch size')
    args = parser.parse_args()
    
    # Override config with args
    CFG.max_epoch = args.epochs
    CFG.meta_batch_size = args.batch_size
    CFG.resume_from_checkpoint = args.resume
    
    print(f"Using device: {CFG.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set random seeds
    set_random_seeds()
    
    # Load environment variables
    load_dotenv()

    # Ensure directories exist
    os.makedirs(CFG.dataset_dir, exist_ok=True)
    os.makedirs(CFG.log_dir, exist_ok=True)
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    # Download dataset if not exists
    if not os.path.exists(CFG.dataset_dir):
        download_dataset(dir=CFG.dataset_dir)

    features_dir = os.path.join(CFG.dataset_dir, 'features')
    split_csv_path = os.path.join(CFG.dataset_dir, 'label_split.csv')

    # Create dataloaders with optimized settings
    dataloaders = create_dataloaders(
        features_dir=features_dir,
        split_csv_path=split_csv_path,
        batch_size=1,
        val_ratio=0.1,
        test_ratio=0.1,
        generalized=False,
        num_workers=4 if CFG.device == "cuda" else 0  # Use more workers for GPU
    )

    print(f"Available dataloaders: {list(dataloaders.keys())}")
    print(f"Train dataset size: {len(dataloaders['train'].dataset)}")

    # Check available classes
    temp_loader = dataloaders['train']
    temp_classes = set()
    for _, label in temp_loader:
        temp_classes.add(label.item())
        if len(temp_classes) >= 10:  # Sample enough to get class info
            break
    
    actual_num_classes = len(temp_classes)
    CFG.n_way = min(CFG.n_way, actual_num_classes)
    print(f"Using {CFG.n_way}-way classification")

    # Create MAML-compatible datasets
    train_maml_dataset = MAMLDatasetWrapper(
        dataloaders['train'], n_way=CFG.n_way, k_shot=CFG.k_shot, 
        q_query=CFG.q_query, num_tasks=1000
    )

    val_maml_dataset = MAMLDatasetWrapper(
        dataloaders['val'], n_way=CFG.n_way, k_shot=CFG.k_shot, 
        q_query=CFG.q_query, num_tasks=200
    )

    test_maml_dataset = MAMLDatasetWrapper(
        dataloaders['test_unseen'], n_way=CFG.n_way, k_shot=CFG.k_shot, 
        q_query=CFG.q_query, num_tasks=300
    )

    # Create DataLoaders
    train_loader = DataLoader(train_maml_dataset, batch_size=1, shuffle=True, 
                            num_workers=4 if CFG.device == "cuda" else 0, 
                            pin_memory=True if CFG.device == "cuda" else False)
    val_loader = DataLoader(val_maml_dataset, batch_size=1, shuffle=False, 
                          num_workers=4 if CFG.device == "cuda" else 0, 
                          pin_memory=True if CFG.device == "cuda" else False)
    test_loader = DataLoader(test_maml_dataset, batch_size=1, shuffle=False, 
                           num_workers=4 if CFG.device == "cuda" else 0, 
                           pin_memory=True if CFG.device == "cuda" else False)

    # Initialize model, optimizer, and logger
    model = MalwareClassifier(CFG.input_dim, output_dim=CFG.n_way).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.meta_lr)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Task configuration: {CFG.n_way}-way {CFG.k_shot}-shot with {CFG.q_query} query samples")

    # Resume from checkpoint if specified
    start_epoch = 1
    if CFG.resume_from_checkpoint:
        start_epoch, _, _ = load_checkpoint(CFG.resume_from_checkpoint, model, optimizer, logger)
        start_epoch += 1

    # Create iterators
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    try:
        for epoch in range(start_epoch, CFG.max_epoch + 1):
            print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
            
            # Train and validate
            train_metrics, train_iter = run_epoch(
                model, optimizer, train_loader, train_iter, loss_fn, train=True
            )
            val_metrics, val_iter = run_epoch(
                model, optimizer, val_loader, val_iter, loss_fn, train=False
            )

            # Print progress
            print(f"Train Loss: {train_metrics['loss']:.3f} | Train Acc: {train_metrics['acc']*100:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.3f} | Val Acc: {val_metrics['acc']*100:.2f}%")
            
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
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg_dict,
                    "epoch": epoch,
                    "val_acc": val_metrics['acc'],
                    "hyperparameters": {
                        'n_way': CFG.n_way,
                        'k_shot': CFG.k_shot,
                        'q_query': CFG.q_query,
                        'input_dim': CFG.input_dim,
                        'inner_lr': CFG.inner_lr,
                        'meta_lr': CFG.meta_lr
                    },
                }, logger.model_path)
                
                print(f"✅ Saved best model (val_acc={val_metrics['acc']*100:.2f}%)")

            # Save checkpoint periodically
            if epoch % CFG.save_every_epochs == 0:
                logger.save_checkpoint(epoch, model, optimizer, 0, 0)
                
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        logger.save_checkpoint(epoch, model, optimizer, 0, 0)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        logger.save_checkpoint(epoch, model, optimizer, 0, 0)
        raise

    print("Training completed!")

    # Testing phase
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    test_predicted_labels, test_true_labels, test_task_accuracies = test_model(
        model, test_loader, num_test_tasks=50
    )
    
    average_test_accuracy = np.mean(test_task_accuracies)
    print(f"Average Test Task Accuracy: {average_test_accuracy*100:.3f}%")

    # Save test results
    import pandas as pd
    results_df = pd.DataFrame({
        'id': range(len(test_predicted_labels)),
        'predicted_class': test_predicted_labels,
        'true_class': test_true_labels
    })

    results_csv_path = os.path.join(CFG.log_dir, 'maml_gpu_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Test results saved as {results_csv_path}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_true_labels, test_predicted_labels, 
                              target_names=[f'Class_{i}' for i in range(CFG.n_way)]))

    print(f"\n✅ Training done. Logs saved at: {logger.path}")
    print(f"✅ Best model saved at: {logger.model_path}")


if __name__ == "__main__":
    main()