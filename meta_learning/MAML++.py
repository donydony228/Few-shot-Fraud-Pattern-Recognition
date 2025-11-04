# python -m meta_learning.MAML_PLUS_PLUS
import os
import sys
import random
import json
import glob
import math
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
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

# Add src to path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from extraction.data_loader import create_dataloaders
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure 'src' directory is in the parent folder or PYTHONPATH")
    # Alternative import paths
    alternative_paths = [
        os.path.join(current_dir, '..', 'src'),
        os.path.join(current_dir, 'src'),
    ]
    
    imported = False
    for alt_path in alternative_paths:
        if alt_path not in sys.path:
            sys.path.append(alt_path)
        try:
            from extraction.data_loader import create_dataloaders
            print(f"Successfully imported from path: {alt_path}")
            imported = True
            break
        except ImportError:
            continue
    
    if not imported:
        print("Unable to import create_dataloaders. Please check project structure.")
        sys.exit(1)


# CONFIGURATION
class CFG:
    """Configuration parameters for MAML++ meta-learning."""
    # Task definition
    n_way = 3
    k_shot = 1
    q_query = 5
    input_dim = 1280

    # MAML++ Hyperparameters
    train_inner_train_step = 3
    val_inner_train_step = 3
    inner_lr = 0.01  # Base inner LR (MAML++ learns this)
    meta_lr = 0.001
    meta_batch_size = 8
    max_epoch = 20
    eval_batches = 10

    # MAML++ specific hyperparameters
    use_first_order_epochs = max_epoch // 2  # First 15 epochs use first-order (DA)
    step_weights_initial = [1.0] * (train_inner_train_step + 1)  # For (MSL)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Paths (from MAML.py for Dataloader and Logger)
    dataset_dir = '../dataset'
    features_dir = os.path.join(dataset_dir, 'features')
    split_csv_path = os.path.join(dataset_dir, 'label_split.csv')
    log_dir = "logs"

    # Random seed
    random_seed = 42
    
    # Epsilon
    eps = 1e-6

# UTILITY FUNCTIONS (from MAML.py)
def set_random_seeds(seed: int = CFG.random_seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Logger:
    """Logger for training statistics and model checkpointing. (from MAML.py)"""
    
    def __init__(self):
        """Initialize logger with file paths and empty log structure."""
        os.makedirs(CFG.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.path = os.path.join(CFG.log_dir, f"maml_plus_plus_{timestamp}.json")
        self.model_path = os.path.join(CFG.log_dir, f"maml_plus_plus_best_{timestamp}.pth")
        self.experiment_name = f"maml_plus_plus_experiment_{timestamp}"
        
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

# DATALOADER (from MAML.py)
class MAMLDatasetWrapper(Dataset):
    """Wrapper to adapt the standardized dataset for MAML's episodic training. (from MAML.py)"""
    
    def __init__(self, dataloader, n_way, k_shot, q_query, num_tasks):
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
        
        # Assuming dataloader has batch_size=1 and yields (features, label)
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
        task_tensor = task_tensor.view(-1, CFG.input_dim)
        
        return task_tensor

# MAML++ MODEL DEFINITION 
class PerStepBatchNorm1d(nn.Module):
    """MAML++ Improvement: Per-Step Batch Normalization (BNRS + BNWB)"""
    
    def __init__(self, num_features, num_steps, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_steps = num_steps
        self.momentum = momentum
        self.eps = eps
        
        # BNWB: Per-step weights and biases (including step 0)
        self.weight = nn.Parameter(torch.ones(num_steps + 1, num_features))
        self.bias = nn.Parameter(torch.zeros(num_steps + 1, num_features))
        
        # BNRS: Per-step running statistics (including step 0)
        self.register_buffer('running_mean', torch.zeros(num_steps + 1, num_features))
        self.register_buffer('running_var', torch.ones(num_steps + 1, num_features))
        
    def forward(self, x, step=0):
        step = min(step, self.num_steps)
            
        if self.training:
            # Use batch statistics during training
            batch_mean = x.mean(dim=0, keepdim=False)
            batch_var = x.var(dim=0, unbiased=False, keepdim=False)
            
            # Update running statistics for this step
            with torch.no_grad():
                self.running_mean[step] = (1 - self.momentum) * self.running_mean[step] + self.momentum * batch_mean
                self.running_var[step] = (1 - self.momentum) * self.running_var[step] + self.momentum * batch_var
            
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics during evaluation
            mean = self.running_mean[step]
            var = self.running_var[step]
        
        # Normalize and apply per-step weights and biases
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[step] * x_norm + self.bias[step]

class MalwarePlusPlusClassifier(nn.Module):
    """MAML++ Enhanced Classifier with multiple improvements"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=3, num_inner_steps=5):
        super(MalwarePlusPlusClassifier, self).__init__()
        self.num_inner_steps = num_inner_steps
        
        # Original network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, output_dim)
        
        # MAML++ IMPROVEMENT: Per-step batch normalization (BNRS + BNWB)
        self.bn1 = PerStepBatchNorm1d(hidden_dim, num_inner_steps)
        self.bn2 = PerStepBatchNorm1d(hidden_dim, num_inner_steps)
        self.bn3 = PerStepBatchNorm1d(hidden_dim//2, num_inner_steps)
        
        # MAML++ IMPROVEMENT: Learnable per-layer per-step learning rates (LSLR)
        # Each layer has its own learning rate for each step
        self.layer_lrs = nn.ParameterDict({
            # FC layers use relatively larger learning rates
            'fc1': nn.Parameter(torch.tensor([0.01, 0.008, 0.006, 0.004, 0.002][:num_inner_steps])),
            'fc2': nn.Parameter(torch.tensor([0.005, 0.004, 0.003, 0.002, 0.001][:num_inner_steps])),
            'fc3': nn.Parameter(torch.tensor([0.003, 0.002, 0.002, 0.001, 0.001][:num_inner_steps])),
            'fc4': nn.Parameter(torch.tensor([0.008, 0.006, 0.004, 0.003, 0.002][:num_inner_steps])),  
            # BN layers use smaller learning rates
            'bn1': nn.Parameter(torch.tensor([0.001, 0.0008, 0.0006, 0.0004, 0.0002][:num_inner_steps])),
            'bn2': nn.Parameter(torch.tensor([0.001, 0.0008, 0.0006, 0.0004, 0.0002][:num_inner_steps])),
            'bn3': nn.Parameter(torch.tensor([0.001, 0.0008, 0.0006, 0.0004, 0.0002][:num_inner_steps]))
        })
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, step=0):
        x = self.fc1(x)
        x = self.bn1(x, step)  # Use step-specific BN
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x, step)  # Use step-specific BN
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x, step)  # Use step-specific BN
        x = F.relu(x)
        
        x = self.fc4(x)
        return x
    
    def functional_forward(self, x, params, step=0):
        """Forward pass using custom parameters with step-aware BN"""
        # FC1
        x = F.linear(x, params.get('fc1.weight', self.fc1.weight), 
                    params.get('fc1.bias', self.fc1.bias))
        
        # BN1 - Use step-specific parameters
        step_idx = min(step, self.num_inner_steps)
        if f'bn1.weight' in params:
            bn1_weight = params[f'bn1.weight'][step_idx]
            bn1_bias = params[f'bn1.bias'][step_idx]
        else:
            bn1_weight = self.bn1.weight[step_idx]
            bn1_bias = self.bn1.bias[step_idx]
            
        bn1_running_mean = self.bn1.running_mean[step_idx]
        bn1_running_var = self.bn1.running_var[step_idx]
        
        x = F.batch_norm(x, bn1_running_mean, bn1_running_var, 
                        bn1_weight, bn1_bias, training=self.training, eps=1e-5)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        
        # FC2
        x = F.linear(x, params.get('fc2.weight', self.fc2.weight), 
                    params.get('fc2.bias', self.fc2.bias))
        
        # BN2
        if f'bn2.weight' in params:
            bn2_weight = params[f'bn2.weight'][step_idx]
            bn2_bias = params[f'bn2.bias'][step_idx]
        else:
            bn2_weight = self.bn2.weight[step_idx]
            bn2_bias = self.bn2.bias[step_idx]
            
        bn2_running_mean = self.bn2.running_mean[step_idx]
        bn2_running_var = self.bn2.running_var[step_idx]
        
        x = F.batch_norm(x, bn2_running_mean, bn2_running_var,
                        bn2_weight, bn2_bias, training=self.training, eps=1e-5)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        
        # FC3
        x = F.linear(x, params.get('fc3.weight', self.fc3.weight), 
                    params.get('fc3.bias', self.fc3.bias))
        
        # BN3
        if f'bn3.weight' in params:
            bn3_weight = params[f'bn3.weight'][step_idx]
            bn3_bias = params[f'bn3.bias'][step_idx]
        else:
            bn3_weight = self.bn3.weight[step_idx]
            bn3_bias = self.bn3.bias[step_idx]
            
        bn3_running_mean = self.bn3.running_mean[step_idx]
        bn3_running_var = self.bn3.running_var[step_idx]
        
        x = F.batch_norm(x, bn3_running_mean, bn3_running_var,
                        bn3_weight, bn3_bias, training=self.training, eps=1e-5)
        x = F.relu(x)
        
        # FC4
        x = F.linear(x, params.get('fc4.weight', self.fc4.weight), 
                    params.get('fc4.bias', self.fc4.bias))
        return x


# UTILITY FUNCTIONS
def create_label(n_way, k_shot):
    """Create labels for support set and query set."""
    return torch.arange(n_way).repeat_interleave(k_shot).long()

def calculate_accuracy(logits, labels):
    """utility function for accuracy calculation"""
    acc = np.asarray(
        [(torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy())]
    ).mean()
    return acc

class CosineAnnealingLR:
    """MAML++ Improvement: Cosine Annealing for Meta-Optimizer (CA)"""
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                               (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        return [group['lr'] for group in self.optimizer.param_groups]


# MAML++ CORE SOLVER (MODIFIED)
def MAMLPlusPlusSolver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step=5,
    train=True,
    epoch=0,
    step_weights=None,
    use_first_order_epochs=15,
    return_labels=False
):
    """MAML++ Core Algorithm Solver"""
    
    task_loss = []
    task_acc = []
    
    if return_labels:
        all_predicted_labels = []
        all_true_labels = []
        task_accuracies = []
    
    # MAML++ IMPROVEMENT 4: Derivative-Order Annealing (DA)
    use_second_order = epoch >= use_first_order_epochs
    
    if step_weights is None:
        step_weights = [1.0] * (inner_train_step + 1)
    
    # x is the meta-batch: [meta_batch_size, N*(K+Q), D]
    for meta_batch in x:
        # meta_batch is one task: [N*(K+Q), D]
        
        # Split support and query sets
        # Reshape to [N, K+Q, D] to split, then flatten back
        task_tensor = meta_batch.view(n_way, k_shot + q_query, -1)
        support_set = task_tensor[:, :k_shot, :].contiguous().view(-1, CFG.input_dim)
        query_set = task_tensor[:, k_shot:, :].contiguous().view(-1, CFG.input_dim)
        
        # Copy the params for inner loop
        fast_weights = OrderedDict(model.named_parameters())
        
        # MAML++ IMPROVEMENT 5: Multi-Step Loss Optimization (MSL)
        # Store losses from all steps for multi-step optimization
        step_losses = []
        
        ### ---------- INNER TRAIN LOOP ---------- ###
        for inner_step in range(inner_train_step):
            train_label = create_label(n_way, k_shot).to(CFG.device)
            logits = model.functional_forward(support_set, fast_weights, step=inner_step)
            loss = loss_fn(logits, train_label)
            
            # MSL: Compute query loss for this step
            if not return_labels:
                val_label = create_label(n_way, q_query).to(CFG.device)
                query_logits = model.functional_forward(query_set, fast_weights, step=inner_step)
                query_loss = loss_fn(query_logits, val_label)
                step_losses.append(query_loss * step_weights[inner_step])

            # Compute gradients
            create_graph = use_second_order if train else False

            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                       create_graph=create_graph,
                                       retain_graph=True, 
                                       allow_unused=True)
            
            # Update fast_weights
            new_fast_weights = OrderedDict()
            for (name, param), grad in zip(fast_weights.items(), grads):
                if grad is None:
                    new_fast_weights[name] = param
                    continue
                
                # MAML++ IMPROVEMENT 6: Per-Layer Per-Step Learning Rates (LSLR)
                layer_name = name.split('.')[0]
                if layer_name in model.layer_lrs:
                    lr = model.layer_lrs[layer_name][inner_step]
                else:
                    lr = CFG.inner_lr # Fallback (shouldn't happen)
                    
                new_fast_weights[name] = param - lr * grad
            
            fast_weights = new_fast_weights
        
        ### ---------- FINAL STEP EVALUATION ---------- ###
        # MSL: Compute query loss for this step
        if not return_labels:
            # Evaluate final step
            val_label = create_label(n_way, q_query).to(CFG.device)
            final_logits = model.functional_forward(query_set, fast_weights, step=inner_train_step)
            final_loss = loss_fn(final_logits, val_label)
            step_losses.append(final_loss * step_weights[inner_train_step])
            
            # MSL: Combine all step losses
            total_loss = sum(step_losses) / sum(step_weights)
            task_loss.append(total_loss)
            task_acc.append(calculate_accuracy(final_logits, val_label))
            
        else:
            # Code path for testing (return_labels=True)
            val_label = create_label(n_way, q_query).to(CFG.device)
            final_logits = model.functional_forward(query_set, fast_weights, step=inner_train_step)
            
            predicted_labels = torch.argmax(final_logits, -1).cpu().numpy().tolist()
            task_true_labels = val_label.cpu().numpy().tolist()
            
            # Calculate current task accuracy
            task_true = np.array(task_true_labels)
            task_pred = np.array(predicted_labels)
            task_acc = (task_true == task_pred).mean()
            task_accuracies.append(task_acc)

            # Collect all labels
            all_predicted_labels.extend(predicted_labels)
            all_true_labels.extend(task_true_labels)

    if return_labels:
        return all_predicted_labels, all_true_labels, task_accuracies
    
    # Backward pass
    if train:
        meta_loss = torch.stack(task_loss).mean()
        optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping (from MAML.py, good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        optimizer.step()
        return meta_loss, np.mean(task_acc)
    
    # Evaluation
    return 0.0, np.mean(task_acc)


# MAML++ TEST FUNCTION
def test_maml_plus_plus(model, test_loader, num_test_tasks=20):
    """Test the MAML++ model on unseen tasks"""
    print("Testing MAML++...")
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Use the test loader
    test_epoch_iterator = tqdm(test_loader, desc="Testing MAML++", total=num_test_tasks)
    
    all_predicted_labels = []
    all_true_labels = []
    all_task_accuracies = []
    
    tasks_run = 0
    
    for x in test_epoch_iterator:
        if tasks_run >= num_test_tasks:
            break
        
        x = x.to(CFG.device)
        
        # Get predictions and labels
        predicted_labels, true_labels, task_accuracies = MAMLPlusPlusSolver(
            model,
            None,  # No optimizer needed for testing
            x,
            CFG.n_way,
            CFG.k_shot,
            CFG.q_query,
            loss_fn,
            inner_train_step=CFG.val_inner_train_step, # Use validation steps
            train=False,
            epoch=CFG.max_epoch, # Use 2nd order
            return_labels=True
        )
        
        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(true_labels)
        all_task_accuracies.extend(task_accuracies) # task_accuracies is a list of batch-size
        
        tasks_run += x.size(0) # Add number of tasks in the batch
        
    return all_predicted_labels, all_true_labels, all_task_accuracies

# MAIN TRAINING SCRIPT
def main():
    """Main training and evaluation script."""
    
    # --- 1. Setup (Seed and Logger) ---
    print(f"DEVICE = {CFG.device}")
    set_random_seeds(CFG.random_seed)
    
    logger = Logger()
    print(f"Logging experiment to: {logger.path}")

    # --- 2. Dataloaders (MODIFIED) ---
    print("Creating dataloaders...")
    
    num_workers = max(os.cpu_count() // 4, 1)
    dataloaders_dict = create_dataloaders(
        CFG.features_dir,
        CFG.split_csv_path,
        batch_size=1,  # Base dataloader batch size is 1 for episodic sampling
        num_workers=num_workers
    )
    
    train_dataloader_base = dataloaders_dict['train']
    val_dataloader_base = dataloaders_dict['val']
    test_dataloader_base = dataloaders_dict['test_unseen'] 

    # Wrap for MAML episodic sampling
    train_dataset = MAMLDatasetWrapper(
        train_dataloader_base,
        n_way=CFG.n_way, k_shot=CFG.k_shot, q_query=CFG.q_query,
        num_tasks=CFG.meta_batch_size * 200 # ~200 steps per epoch
    )
    val_dataset = MAMLDatasetWrapper(
        val_dataloader_base,
        n_way=CFG.n_way, k_shot=CFG.k_shot, q_query=CFG.q_query,
        num_tasks=CFG.meta_batch_size * 50
    )
    test_dataset = MAMLDatasetWrapper(
        test_dataloader_base,
        n_way=CFG.n_way, k_shot=CFG.k_shot, q_query=CFG.q_query,
        num_tasks=CFG.meta_batch_size * 50
    )

    # Create final DataLoaders
    pin_mem = (CFG.device == 'cuda')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.meta_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.meta_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.meta_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    print("Dataloaders created.")

    # --- 3. Model, Optimizer, Loss (Unchanged from MAML++) ---
    meta_model = MalwarePlusPlusClassifier(
        input_dim=CFG.input_dim, 
        output_dim=CFG.n_way,
        num_inner_steps=CFG.train_inner_train_step
    ).to(CFG.device)

    # MAML++ IMPROVEMENT: Meta-optimizer with cosine annealing (CA)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=CFG.meta_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.max_epoch, eta_min=CFG.meta_lr * 0.01)

    loss_fn = nn.CrossEntropyLoss()

    print(f"MAML++ Model parameters: {sum(p.numel() for p in meta_model.parameters())}")
    print(f"Using first-order gradients for first {CFG.use_first_order_epochs} epochs")
    
    # --- 4. Training Loop (MODIFIED) ---
    step_weights = CFG.step_weights_initial
    
    for epoch in range(CFG.max_epoch):
        current_lr = optimizer.param_groups[0]['lr'] 
        order = "1st" if epoch < CFG.use_first_order_epochs else "2nd"
        print("--------------------------------------------------")
        print(f"Epoch {epoch+1}/{CFG.max_epoch}")
        print(f"Meta-LR: {current_lr:.6f}, Using {order}-order gradients")
        
        meta_model.train()
        train_meta_loss = []
        train_acc = []
  
        train_epoch_iterator = tqdm(train_loader, desc=f"Training")
        
        for x in train_epoch_iterator:
            # x is a meta-batch: [meta_batch_size, N*(K+Q), D]
            x = x.to(CFG.device)

            # Check for NaN/Inf (good practice from MAML.py)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: Skipping meta-batch due to NaN/Inf.")
                continue
                
            # MAMLPlusPlusSolver loops over the meta-batch internally
            meta_loss, acc = MAMLPlusPlusSolver(
                meta_model,
                optimizer,
                x,
                CFG.n_way,
                CFG.k_shot,
                CFG.q_query,
                loss_fn,
                inner_train_step=CFG.train_inner_train_step,
                train=True,
                epoch=epoch,
                step_weights=step_weights,
                use_first_order_epochs=CFG.use_first_order_epochs
            )
            
            if meta_loss is not None:
                train_meta_loss.append(meta_loss.item())
                train_acc.append(acc)
                train_epoch_iterator.set_postfix({
                    "loss": f"{meta_loss.item():.3f}", 
                    "acc": f"{acc*100:.2f}%"
                })

        avg_train_loss = np.mean(train_meta_loss)
        avg_train_acc = np.mean(train_acc)
        print(f"Train Loss: {avg_train_loss:.3f}\tAccuracy: {avg_train_acc*100:.3f}%")
        
        # Validation
        meta_model.eval()
        val_acc = []
        
        val_epoch_iterator = tqdm(val_loader, desc="Validation")
        
        for x in val_epoch_iterator:
            # x is a meta-batch: [meta_batch_size, N*(K+Q), D]
            x = x.to(CFG.device)
            
            _, acc = MAMLPlusPlusSolver(
                meta_model,
                optimizer,
                x,
                CFG.n_way,
                CFG.k_shot,
                CFG.q_query,
                loss_fn,
                inner_train_step=CFG.val_inner_train_step,
                train=False, # Set train=False
                epoch=epoch,
                step_weights=step_weights,
                use_first_order_epochs=CFG.use_first_order_epochs
            )
            val_acc.append(acc)
        
        avg_val_acc = np.mean(val_acc)
        print(f"Validation accuracy: {avg_val_acc*100:.3f}%")
        
        train_metrics = {
            "loss": avg_train_loss,
            "accuracy": avg_train_acc
        }
        val_metrics = {
            "accuracy": avg_val_acc
        }
        logger.add(epoch, train_metrics, val_metrics)
        
        if logger.should_save_best(avg_val_acc):
            print(f"New best model found! Saving to {logger.model_path}")
            torch.save(meta_model.state_dict(), logger.model_path)
            
        # Update scheduler
        scheduler.step()

    print("\nMAML++ Training completed.")
    print(f"Best validation accuracy: {logger.best_val*100:.3f}% at epoch {logger.best_epoch}")
    print(f"Best model saved to: {logger.model_path}")

    # --- 5. Testing (MODIFIED) ---
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    # Load best model for testing
    if os.path.exists(logger.model_path):
        print(f"Loading best model from {logger.model_path}")
        meta_model.load_state_dict(torch.load(logger.model_path, map_location=CFG.device))
    else:
        print("Warning: Best model not found. Testing with the final model.")
        
    test_predicted_labels, test_true_labels, test_task_accuracies = test_maml_plus_plus(
        meta_model, test_loader, num_test_tasks=CFG.eval_batches * CFG.meta_batch_size
    )
    
    average_test_accuracy = np.mean(test_task_accuracies)
    std_test_accuracy = np.std(test_task_accuracies)
    
    print(f"MAML++ Final Test Results:")
    print(f"Average Test Task Accuracy: {average_test_accuracy*100:.3f}% ± {std_test_accuracy*100:.3f}%")
    print(f"Best Task Accuracy: {np.max(test_task_accuracies)*100:.3f}%")
    print(f"Worst Task Accuracy: {np.min(test_task_accuracies)*100:.3f}%")
    
    # Save MAML++ test results
    results_df = pd.DataFrame({
        'id': range(len(test_predicted_labels)),
        'predicted_class': test_predicted_labels,
        'true_class': test_true_labels
    })

    results_csv_path = os.path.join(CFG.log_dir, 'maml_plus_plus_test_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"MMAML++ test results saved as {results_csv_path}")
    
    # Calculate detailed metrics
    print("\nDetailed Classification Report:")
    print(classification_report(test_true_labels, test_predicted_labels, 
                              target_names=[f'Class {i}' for i in range(CFG.n_way)]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_true_labels, test_predicted_labels))


if __name__ == "__main__":
    main()