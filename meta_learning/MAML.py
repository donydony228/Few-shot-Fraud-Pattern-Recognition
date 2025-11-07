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
    classification_report,
    accuracy_score
)

from dotenv import load_dotenv

# Add src to path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_path)

try:
    from extraction.data_loader import create_meta_learning_dataloaders, split_episode_to_support_query
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
            from extraction.data_loader import create_meta_learning_dataloaders, split_episode_to_support_query
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

    inner_lr = 0.005 
    meta_lr = 0.0005
    inner_steps_train = 5  
    inner_steps_val = 5

    meta_batch_size = 16  
    max_epoch = 200  
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

def calculate_metrics(preds: List[int], labels: List[int], num_classes: int) -> Dict[str, Any]:
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

# MODEL
class MalwareClassifier(nn.Module):
    """Neural network for malware classification with functional forward pass for MAML."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
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
                         episode_data: torch.Tensor,
                         train: bool
                         ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """MAML step with predictions."""
    
    # Deal with episode_data format
    if isinstance(episode_data, (list, tuple)) and len(episode_data) >= 1:
        episode_tensor = episode_data[0]
    else:
        episode_tensor = episode_data
    
    episode = episode_tensor.squeeze(0)
    
    # Check episode shape
    if episode.dim() != 3:  # Expect [n_way, k_shot+q_query, feature_dim]
        print(f"Warning: Unexpected episode shape {episode.shape}, attempting to reshape...")
        # Attempt reshape
        total_samples = CFG.n_way * (CFG.k_shot + CFG.q_query)
        if episode.numel() // CFG.input_dim == total_samples:
            episode = episode.view(CFG.n_way, CFG.k_shot + CFG.q_query, CFG.input_dim)
        else:
            raise ValueError(f"Cannot reshape episode {episode.shape} to expected format")
    
    # Split support and query sets
    support_x, query_x, support_y, query_y = split_episode_to_support_query(
        episode, k_shot=CFG.k_shot, q_query=CFG.q_query
    )

    # Move to device
    support_x = support_x.to(CFG.device)
    query_x = query_x.to(CFG.device)
    support_y = support_y.to(CFG.device)
    query_y = query_y.to(CFG.device)
    
    # Check NaN/Inf in support/query data
    if torch.isnan(support_x).any() or torch.isnan(query_x).any():
        raise ValueError("NaN detected in support/query data")
    
    # Initialize fast weights
    fast_weights = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            fast_weights[name] = param.clone()

    inner_steps = CFG.inner_steps_train if train else CFG.inner_steps_val
    
    # Inner loop
    for inner_step in range(inner_steps):
        support_logits = model.functional_forward(support_x, fast_weights)
        support_loss = criterion(support_logits, support_y)
        
        # Check support loss
        if torch.isnan(support_loss) or torch.isinf(support_loss):
            raise ValueError(f"Invalid support loss at inner step {inner_step}")
        
        grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=False)

        # Check gradient validity
        for grad in grads:
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                raise ValueError(f"Invalid gradient at inner step {inner_step}")
        
        # Update fast weights
        fast_weights = OrderedDict(
            (name, param - CFG.inner_lr * grad)
            for ((name, param), grad) in zip(fast_weights.items(), grads)
        )

    # Query forward
    query_logits = model.functional_forward(query_x, fast_weights)
    query_loss = criterion(query_logits, query_y)

    # Check query loss
    if torch.isnan(query_loss) or torch.isinf(query_loss):
        raise ValueError("Invalid query loss")

    # Calculate accuracy
    with torch.no_grad():
        preds = torch.argmax(query_logits, dim=1)
        task_acc = (preds == query_y).sum().item() / len(query_y)

    # FO-MAML outer loop
    if train:
        # Compute meta-gradients w.r.t original model parameters
        meta_grads = torch.autograd.grad(
            query_loss, 
            model.parameters(),
            create_graph=False
        )

        # Check meta gradients
        for grad in meta_grads:
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                raise ValueError("Invalid meta gradient")

        # Accumulate gradients
        for param, meta_grad in zip(model.parameters(), meta_grads):
            if param.grad is None:
                param.grad = meta_grad.detach()
            else:
                param.grad += meta_grad.detach()

    return query_loss, task_acc, preds.detach().cpu(), query_y.detach().cpu()

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
    Run a single epoch of training or validation.
    """
    
    if train:
        meta_optimizer.zero_grad()

    epoch_meta_loss = 0.0
    all_preds = []
    all_labels = []
    sample_episode = next(iter(dataloader))
    
    # Set up progress bar
    epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.max_epoch} [{mode}]", leave=False)
    
    for i, episode_data in enumerate(epoch_iterator):
        
        # Deal with episode_data format
        if isinstance(episode_data, (list, tuple)) and len(episode_data) >= 1:
            episode_tensor = episode_data[0]
        else:
            episode_tensor = episode_data
        
        # Check NaN/Inf in episode data
        if torch.isnan(episode_tensor).any() or torch.isinf(episode_tensor).any():
            print(f"\nWarning: Skipping task {i} due to NaN/Inf in episode data.")
            continue
        
        # MAML step
        try:
            meta_loss, task_acc, preds, labels = maml_step_with_preds(
                model, criterion, episode_data, train=train  
            )
        except Exception as e:
            print(f"\nError in maml_step at task {i}: {e}")
            continue
        
        # Check meta loss validity
        meta_loss_float = meta_loss.item()
        if np.isnan(meta_loss_float) or np.isinf(meta_loss_float):
            print(f"\nWarning: Detected invalid loss in epoch {epoch+1} [{mode}] at task {i}. Stopping epoch early.")
            epoch_meta_loss = np.nan
            break 
        
        epoch_meta_loss += meta_loss_float
        all_preds.extend(preds)
        all_labels.extend(labels)

        # Check gradient update logic
        if train and (i + 1) % CFG.meta_batch_size == 0:
            # Check gradient validity
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            if np.isnan(total_norm) or np.isinf(total_norm):
                print(f"\nWarning: Invalid gradient norm detected, skipping update")
                meta_optimizer.zero_grad()
                continue
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.grad_clip)
            meta_optimizer.step()
            meta_optimizer.zero_grad()
        
        epoch_iterator.set_postfix(
            meta_loss=f"{epoch_meta_loss / (i + 1):.4f}", 
            task_acc=f"{task_acc:.4f}"
        )

    # Calculate average loss
    if np.isnan(epoch_meta_loss):
        avg_epoch_loss = np.nan
    else:
        avg_epoch_loss = epoch_meta_loss / max(len(dataloader), 1)
    
    # Handle case with no valid labels/predictions
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

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    avg_accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    cm = confusion_matrix(all_labels.astype(int), all_preds.astype(int))

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
    
    all_predicted_labels = []
    all_true_labels = []
    task_accuracies = []

    print("Starting testing (collecting predictions for detailed report)...")
    device = CFG.device 

    # Iterate over test tasks
    task_count = 0
    for episode_data in tqdm(test_loader, desc="Testing"):
        if task_count >= num_test_tasks:
            break

        # Retrieve episode tensor
        episode_tensor = episode_data[0] if isinstance(episode_data, (list, tuple)) else episode_data
        episode = episode_tensor.squeeze(0)  # [n_way, k_shot+q_query, feature_dim]
        
        support_x, query_x, support_y, query_y = split_episode_to_support_query(
            episode, CFG.k_shot, CFG.q_query
        )
        
        support_x, query_x = support_x.to(device), query_x.to(device)
        
        # Check NaN/Inf in support/query data
        if torch.isnan(support_x).any() or torch.isnan(query_x).any():
            task_count += 1
            continue

        # Normalize (optional, supported by new dataloader)
        if hasattr(CFG, 'normalize_at_test') and CFG.normalize_at_test:
            mean = torch.mean(support_x, dim=0)
            std = torch.std(support_x, dim=0)
            
            support_x = (support_x - mean) / (std + CFG.eps)
            query_x = (query_x - mean) / (std + CFG.eps)
            
            support_x = torch.nan_to_num(support_x, nan=0.0)
            query_x = torch.nan_to_num(query_x, nan=0.0)
        
        task_tensor = torch.zeros(CFG.n_way * (CFG.k_shot + CFG.q_query), support_x.shape[-1], device=device)
        
        for n in range(CFG.n_way):
            class_start_idx = n * (CFG.k_shot + CFG.q_query)
            support_start = n * CFG.k_shot
            query_start = n * CFG.q_query
            
            task_tensor[class_start_idx:class_start_idx + CFG.k_shot] = support_x[support_start:support_start + CFG.k_shot]
            task_tensor[class_start_idx + CFG.k_shot:class_start_idx + CFG.k_shot + CFG.q_query] = query_x[query_start:query_start + CFG.q_query]
        
        predicted_labels, true_labels = get_task_predictions(
            model, task_tensor, nn.CrossEntropyLoss(), CFG, CFG.inner_steps_val
        )

        task_true = np.array(true_labels)
        task_pred = np.array(predicted_labels)
        task_acc = (task_true == task_pred).mean()
        task_accuracies.append(task_acc)

        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(true_labels)
        
        task_count += 1

    return all_predicted_labels, all_true_labels, task_accuracies

def main():
    """Main training loop for MAML meta-learning."""
    print(f"MAML Meta-Learning for Malware Classification (FIXED VERSION)")
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
    dataloaders = create_meta_learning_dataloaders(
        features_dir=features_dir,
        split_csv_path=split_csv_path,
        n_way=CFG.n_way,
        k_shot=CFG.k_shot,
        q_query=CFG.q_query,
        train_episodes_per_epoch=200,   
        val_episodes_per_epoch=100,     
        test_episodes_per_epoch=200,    
        normalize=True,
        num_workers=0,
        pin_memory=False,
        seed=CFG.random_seed
    )

    # Define dataloaders
    train_loader = dataloaders['train']
    val_loader = dataloaders['val'] 
    test_loader = dataloaders['test_unseen']  # Use unseen test set for final evaluation

    # Check episode format and adjust input_dim if necessary
    sample_episode = next(iter(train_loader))
    if isinstance(sample_episode, (list, tuple)) and len(sample_episode) >= 1:
        episode_tensor = sample_episode[0]  # Get episode tensor
        actual_input_dim = episode_tensor.shape[-1]
    else:
        episode_tensor = sample_episode
        actual_input_dim = episode_tensor.shape[-1]

    # Auto-adjust input_dim
    if actual_input_dim != CFG.input_dim:
        CFG.input_dim = actual_input_dim

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