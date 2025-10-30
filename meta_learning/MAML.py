# python -m meta_learning.MAML
import os
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
    classification_report
)

from dotenv import load_dotenv

# Add src to path to import project modules
# 根據你從 meta_learning/ 目錄執行，所以路徑應該是 ../src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_path)

try:
    from extraction.data_loader import create_dataloaders
    from extraction.downloader import download_dataset
except ImportError as e:
    print(f"導入錯誤: {e}")
    print(f"當前目錄: {current_dir}")
    print(f"src 路徑: {src_path}")
    print(f"sys.path: {sys.path}")
    
    # 備用方案：嘗試不同的路徑
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
            print(f"成功從路徑導入: {alt_path}")
            imported = True
            break
        except ImportError:
            continue
    
    if not imported:
        print("無法導入所需模組，請檢查專案結構")
        sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

class CFG:
    """Configuration parameters for MAML meta-learning.

    Attributes:
        n_way: Number of classes in each task (default: 3 for 2 malware + 1 benign).
        k_shot: Number of support samples per class (default: 1).
        q_query: Number of query samples per class (default: 5).
        input_dim: Dimensionality of input features (default: 1280).
        inner_lr: Learning rate for inner loop (task-specific adaptation).
        meta_lr: Learning rate for outer loop (meta-learning update).
        inner_steps_train: Number of gradient steps in inner loop during training.
        inner_steps_val: Number of gradient steps in inner loop during validation.
        meta_batch_size: Number of tasks per meta-update.
        max_epoch: Maximum number of training epochs.
        eval_batches: Number of batches for evaluation.
        device: Computation device (cuda/cpu/mps).
        dataset_dir: Path to dataset directory.
        log_dir: Directory for saving logs and model checkpoints.
    """
    n_way = 3
    k_shot = 1
    q_query = 5
    input_dim = 1280

    inner_lr = 0.001
    meta_lr = 0.001
    inner_steps_train = 5
    inner_steps_val = 5

    meta_batch_size = 16
    max_epoch = 30
    eval_batches = 20
    
    # Device selection with MPS support for MacBook
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
            self.best_epoch = len(self.logs["epochs"])
            return True
        return False


# ============================================================================
# DATASET WRAPPER - Using standardized dataloader
# ============================================================================

class MAMLDatasetWrapper(Dataset):
    """
    Wrapper to adapt the standardized dataset for MAML's episodic training.
    Creates N-way K-shot tasks from the standardized malware dataset.
    """
    def __init__(self, dataloader, n_way=3, k_shot=1, q_query=5, num_tasks=1000):
        """
        Args:
            dataloader: One of the standardized dataloaders (train/val/test)
            n_way: Number of classes per task
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
            num_tasks: Number of tasks to generate
        """
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
        
        # For MAML, we just need enough classes to form n_way tasks
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
            class_data[label_item].append(features.squeeze(0))  # Remove batch dimension
        
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
    """
    Get meta batch function adapted for standardized dataloader
    """
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


# ============================================================================
# MODEL
# ============================================================================

class MalwareClassifier(nn.Module):
    """Neural network for malware classification with functional forward pass for MAML."""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        """
        A simple feedforward neural network for malware classification.
        input_dim: 1280 (standardized feature dimension)
        output_dim: 3 (2 malware + 1 benign)
        """
        super(MalwareClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
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
            elif isinstance(module, nn.ReLU):
                x = F.relu(x)
            elif isinstance(module, nn.Dropout):
                x = F.dropout(x, training=self.training)
        return x


# ============================================================================
# MAML ALGORITHM
# ============================================================================

def maml_step(model, optimizer, x, n_way, k_shot, q_query, loss_fn, 
              inner_train_step, inner_lr, train, return_labels=False):
    """
    Main MAML algorithm implementation.
    
    Args:
        model: PyTorch model to meta-learn
        optimizer: Meta-optimizer
        x: Batch of tasks
        n_way: Number of classes per task
        k_shot: Number of support samples per class
        q_query: Number of query samples per class
        loss_fn: Loss function
        inner_train_step: Number of inner loop steps
        inner_lr: Inner loop learning rate
        train: Whether to perform meta-update
        return_labels: Whether to return predictions for evaluation
        
    Returns:
        If return_labels=False: (meta_loss, accuracy)
        If return_labels=True: list of predicted labels
    """
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
            # Training on support set
            train_label = create_label(n_way, k_shot).to(CFG.device)
            logits = model.functional_forward(support_set, fast_weights)
            loss = criterion(logits, train_label)
            
            # Inner gradients update!
            # Calculate gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            # Update fast_weights
            # θ' = θ - α * ∇loss
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )

        ### ---------- INNER VALID LOOP ---------- ###
        if not return_labels:
            """ training / validation """
            val_label = create_label(n_way, q_query).to(CFG.device)

            # Collect gradients for outer loop
            logits = model.functional_forward(query_set, fast_weights)
            loss = criterion(logits, val_label)
            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, val_label))
        else:
            """ testing """
            logits = model.functional_forward(query_set, fast_weights)
            labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    # Update outer loop
    model.train()
    optimizer.zero_grad()

    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        """ Outer Loop Update """
        # φ backpropagation
        meta_batch_loss.backward()
        # Update parameters
        optimizer.step()

    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc


# ============================================================================
# TRAINING
# ============================================================================

def run_epoch(model, optimizer, loader, iterator, loss_fn, train=True):
    """Run one training or validation epoch.

    Args:
        model: Model to train/evaluate.
        optimizer: Meta-optimizer (used only for training).
        loader: DataLoader for the dataset.
        iterator: Iterator over the loader.
        loss_fn: Loss function.
        train: Whether to train the model (True) or evaluate (False).

    Returns:
        Dictionary of metrics and updated iterator.
    """
    losses = []
    preds = []
    labels = []
    
    # Determine number of batches
    if train:
        num_batches = len(loader) // CFG.meta_batch_size
    else:
        num_batches = CFG.eval_batches

    # Choose inner steps based on train/val
    inner_steps = CFG.inner_steps_train if train else CFG.inner_steps_val

    for _ in tqdm(range(num_batches), desc="Train" if train else "Val"):
        # Get meta-batch of tasks
        x, iterator = get_meta_batch(
            CFG.meta_batch_size, CFG.k_shot, CFG.q_query, loader, iterator
        )
        
        # Perform MAML step
        loss, acc = maml_step(
            model,
            optimizer,
            x,
            CFG.n_way,
            CFG.k_shot,
            CFG.q_query,
            loss_fn,
            inner_train_step=inner_steps,
            inner_lr=CFG.inner_lr,
            train=train,
        )
        
        losses.append(loss.item())
        
        # For detailed metrics, we need to collect predictions
        # This is a simplified version - in practice you might want more detailed evaluation
        if not train:  # Only collect detailed metrics during validation
            # Run one more time to get predictions
            pred_labels = maml_step(
                model,
                optimizer,
                x,
                CFG.n_way,
                CFG.k_shot,
                CFG.q_query,
                loss_fn,
                inner_train_step=inner_steps,
                inner_lr=CFG.inner_lr,
                train=False,
                return_labels=True,
            )
            
            # Generate corresponding true labels
            true_labels = []
            for _ in range(CFG.meta_batch_size):
                for class_idx in range(CFG.n_way):
                    true_labels.extend([class_idx] * CFG.q_query)
            
            preds.extend(pred_labels)
            labels.extend(true_labels)
    
    # Compute overall metrics
    if train:
        # For training, we only have loss and approximate accuracy
        metrics = {
            "loss": float(np.mean(losses)),
            "acc": acc,  # Last batch accuracy
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "cm": []
        }
    else:
        # For validation, compute full metrics
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
    """Test the trained model and return detailed results."""
    test_iter = iter(test_loader)
    
    all_predicted_labels = []
    all_true_labels = []
    task_accuracies = []

    print("Starting testing...")

    for batch_idx in tqdm(range(num_test_tasks), desc="Testing"):
        x, test_iter = get_meta_batch(1, CFG.k_shot, CFG.q_query, test_loader, test_iter)

        # Generate true labels for this task
        task_true_labels = []
        for class_idx in range(CFG.n_way):
            task_true_labels.extend([class_idx] * CFG.q_query)

        # Get model predictions
        predicted_labels = maml_step(
            model,
            None,  # No optimizer needed for testing
            x,
            CFG.n_way,
            CFG.k_shot,
            CFG.q_query,
            nn.CrossEntropyLoss(),
            inner_train_step=CFG.inner_steps_val,
            inner_lr=CFG.inner_lr,
            train=False,
            return_labels=True,
        )

        # Calculate current task accuracy
        task_true = np.array(task_true_labels)
        task_pred = np.array(predicted_labels)
        task_acc = (task_true == task_pred).mean()
        task_accuracies.append(task_acc)

        # Collect all labels
        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(task_true_labels)

    return all_predicted_labels, all_true_labels, task_accuracies


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main training loop for MAML meta-learning."""
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
        batch_size=1,  # We'll handle batching in MAML
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
    for _, label in temp_loader:
        temp_classes.add(label.item())
    actual_num_classes = len(temp_classes)
    print(f"Actual number of classes: {actual_num_classes}")

    # Adjust n_way to fit available classes
    CFG.n_way = min(CFG.n_way, actual_num_classes)
    print(f"Using {CFG.n_way}-way classification")

    # Create MAML-compatible datasets from standardized dataloaders
    train_maml_dataset = MAMLDatasetWrapper(
        dataloaders['train'], 
        n_way=CFG.n_way, 
        k_shot=CFG.k_shot, 
        q_query=CFG.q_query, 
        num_tasks=1000
    )

    val_maml_dataset = MAMLDatasetWrapper(
        dataloaders['val'], 
        n_way=CFG.n_way, 
        k_shot=CFG.k_shot, 
        q_query=CFG.q_query, 
        num_tasks=200
    )

    test_maml_dataset = MAMLDatasetWrapper(
        dataloaders['test_unseen'], 
        n_way=CFG.n_way, 
        k_shot=CFG.k_shot, 
        q_query=CFG.q_query, 
        num_tasks=300
    )

    # Create DataLoaders for MAML
    train_loader = DataLoader(train_maml_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_maml_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_maml_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model, optimizer, and logger
    model = MalwareClassifier(CFG.input_dim, output_dim=CFG.n_way).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.meta_lr)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Using standardized malware dataset with {CFG.input_dim} features")
    print(f"Task configuration: {CFG.n_way}-way {CFG.k_shot}-shot with {CFG.q_query} query samples")

    # Create iterators
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Training loop
    print("Starting training with standardized dataloader...")
    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        
        # Train and validate
        train_metrics, train_iter = run_epoch(
            model, optimizer, train_loader, train_iter, loss_fn, train=True
        )
        val_metrics, val_iter = run_epoch(
            model, optimizer, val_loader, val_iter, loss_fn, train=False
        )

        # Print progress
        print(f"Train Loss: {train_metrics['loss']:.3f} | Train Acc: {train_metrics['acc']*100:.3f}%")
        print(f"Val Loss: {val_metrics['loss']:.3f} | Val Acc: {val_metrics['acc']*100:.3f}%")
        
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
            
            print(f"✅ Saved best model: {logger.model_path} "
                  f"(val_acc={val_metrics['acc']*100:.2f}%)")

    print("Training completed!")

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
    print(f"\n✅ Training done. Logs saved at: {logger.path}")
    print(f"✅ Best model saved at: {logger.model_path}")
    print(f"✅ Test results saved at: {results_csv_path}")

    # Verification
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)
    print(f"Dataset directory: {CFG.dataset_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Split CSV path: {split_csv_path}")
    print(f"Available dataloaders: {list(dataloaders.keys())}")
    print(f"Feature dimension: {CFG.input_dim}")
    print(f"Task configuration: {CFG.n_way}-way {CFG.k_shot}-shot classification")
    print(f"Model trained for {CFG.max_epoch} epochs")
    print(f"Final test accuracy: {average_test_accuracy*100:.3f}%")
    print("\nIntegration with standardized dataloader: SUCCESS ✓")


if __name__ == "__main__":
    main()