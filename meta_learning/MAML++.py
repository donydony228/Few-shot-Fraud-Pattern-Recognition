# python -m meta_learning.MAML++
import os
import sys
sys.path.append('.') 
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
from torch.optim.lr_scheduler import CosineAnnealingLR
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
if src_path not in sys.path:
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
    """Configuration parameters for MAML++ meta-learning."""
    # Task definition - 固定1-shot設置
    n_way = 3
    k_shot = 5  # 固定1-shot
    q_query = 15  # 減少query數量以平衡訓練速度
    input_dim = 1280

    # MAML++ Hyperparameters - 優化for 200 epochs
    train_inner_train_step = 3  # 增加內部步驟，1-shot需要更多適應
    val_inner_train_step = 3    # 測試時也用更多步驟
    inner_lr = 0.0005             # 保持合理的內部學習率
    meta_lr = 0.0001            # 降低meta學習率for更長訓練
    meta_batch_size = 32        # 減小batch size提高穩定性
    max_epoch = 200             # 增加到200 epochs
    eval_batches = 20

    # MAML++ specific hyperparameters - 調整for長期訓練
    use_first_order_epochs = 100  # 前100epochs用一階，後100epochs用二階
    # step_weights_initial = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]  # 多步驟權重
    step_weights_initial = [0.2, 0.4, 0.6, 0.8, 1.0]

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

# MAML++ Per-Step BatchNorm
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

# MODEL (MAML++ with all improvements)
class MalwarePlusPlusClassifier(nn.Module):
    """MAML++ Enhanced Classifier with multiple improvements"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=3, num_inner_steps=3):
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
            'fc2': nn.Parameter(torch.tensor([0.008, 0.006, 0.005, 0.003, 0.002][:num_inner_steps])),
            'fc3': nn.Parameter(torch.tensor([0.006, 0.004, 0.003, 0.002, 0.001][:num_inner_steps])),
            'fc4': nn.Parameter(torch.tensor([0.01, 0.008, 0.006, 0.004, 0.002][:num_inner_steps])),  
            # BN layers use smaller learning rates
            'bn1': nn.Parameter(torch.tensor([0.002, 0.0015, 0.001, 0.0008, 0.0005][:num_inner_steps])),
            'bn2': nn.Parameter(torch.tensor([0.002, 0.0015, 0.001, 0.0008, 0.0005][:num_inner_steps])),
            'bn3': nn.Parameter(torch.tensor([0.002, 0.0015, 0.001, 0.0008, 0.0005][:num_inner_steps]))
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

# MAML++ SOLVER (MAIN FUNCTION)
def MAMLPlusPlusSolver(
    model,
    optimizer,
    episode_data,  # 🔥 改變參數名
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step=1,
    train=True,
    epoch=0,
    step_weights=None,
    use_first_order_epochs=15,
    return_labels=False
):
    """
    MAML++ solver adapted for new meta-learning dataloader.
    
    Key MAML++ improvements implemented:
    1. Per-parameter learning rates (PU) - now layer-wise (LSLR)
    2. Multi-step loss optimization (MSL)  
    3. Derivative-order annealing (DA)
    4. Per-step batch normalization (BNRS + BNWB)
    """
    
    # 🔥 處理新版dataloader格式
    if isinstance(episode_data, (list, tuple)) and len(episode_data) >= 1:
        # episode_data is tuple: (task_tensor, selected_labels, task_labels)
        episode_tensor = episode_data[0]  # [1, n_way, k_shot+q_query, feature_dim]
    else:
        episode_tensor = episode_data
    
    # 移除batch維度: [n_way, k_shot+q_query, feature_dim]
    episode = episode_tensor.squeeze(0)
    
    # 🔥 使用新版split函數替代手動處理
    support_x, query_x, support_y, query_y = split_episode_to_support_query(
        episode, k_shot=k_shot, q_query=q_query
    )
    
    # 移動到device
    support_x = support_x.to(CFG.device)
    query_x = query_x.to(CFG.device)
    support_y = support_y.to(CFG.device)
    query_y = query_y.to(CFG.device)
    
    # 🔥 檢查數據有效性
    if torch.isnan(support_x).any() or torch.isnan(query_x).any():
        raise ValueError("NaN detected in support/query data")
    
    # MAML++ IMPROVEMENT: Derivative-order annealing (DA)
    use_second_order = epoch >= use_first_order_epochs and train
    
    # Get initial parameters - 確保參數需要梯度
    fast_weights = OrderedDict()
    for name, param in model.named_parameters():
        # 🔥 修復：驗證時也需要梯度來進行內部適應，只是不更新meta參數
        if k_shot > 0:  # 只要有support set就需要梯度來做適應
            fast_weights[name] = param.clone().requires_grad_(True)
        else:
            fast_weights[name] = param.clone()
    
    # MAML++ IMPROVEMENT: Multi-step loss optimization (MSL)
    if step_weights is None:
        step_weights = [1.0] * (inner_train_step + 1)
    
    task_losses = []
    
    # Inner loop adaptation
    for step in range(inner_train_step):
        if k_shot > 0:  # 如果有support set
            # Forward pass on support set with step-aware model
            support_pred = model.functional_forward(support_x, fast_weights, step=step)
            support_loss = loss_fn(support_pred, support_y)
            
            # 🔥 修復：驗證時也需要做參數適應，只是不更新meta參數
            # 計算梯度（驗證時用一階梯度更快）
            create_graph = use_second_order and train  # 只有訓練時才用二階
            grads = torch.autograd.grad(
                support_loss, 
                fast_weights.values(),
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True
            )
            
            # MAML++ IMPROVEMENT: Layer-wise Step-wise Learning Rates (LSLR)
            new_fast_weights = OrderedDict()
            for (name, param), grad in zip(fast_weights.items(), grads):
                if grad is None:
                    new_fast_weights[name] = param
                    continue
                
                # Get layer name and use layer-specific learning rate
                layer_name = name.split('.')[0]  # e.g., 'fc1', 'bn1'
                if layer_name in model.layer_lrs and step < len(model.layer_lrs[layer_name]):
                    lr = model.layer_lrs[layer_name][step]
                else:
                    lr = CFG.inner_lr  # Fallback
                
                new_fast_weights[name] = param - lr * grad
            
            fast_weights = new_fast_weights
        
        # Evaluate on query set for this step (always compute this)
        query_pred = model.functional_forward(query_x, fast_weights, step=step)
        query_loss = loss_fn(query_pred, query_y)
        
        # MAML++ IMPROVEMENT: Multi-step loss optimization (MSL) 
        weighted_loss = step_weights[step] * query_loss
        task_losses.append(weighted_loss)
    
    # Final query evaluation (step = inner_train_step)
    final_query_pred = model.functional_forward(query_x, fast_weights, step=inner_train_step)
    final_query_loss = loss_fn(final_query_pred, query_y)
    weighted_final_loss = step_weights[-1] * final_query_loss
    task_losses.append(weighted_final_loss)
    
    # Calculate meta loss (sum of weighted losses)
    meta_loss = torch.stack(task_losses).sum() / sum(step_weights)
    
    # Training mode: update meta parameters
    if train and optimizer is not None:
        optimizer.zero_grad()
        meta_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    
    # 計算準確率用於監控
    with torch.no_grad():
        predicted_labels = torch.argmax(final_query_pred, dim=1).cpu().numpy()
        true_labels = query_y.cpu().numpy()
        
    if return_labels:
        return predicted_labels, true_labels, meta_loss, task_losses
    else:
        # 返回meta_loss和準確率用於訓練監控
        accuracy = (predicted_labels == true_labels).mean()
        return [meta_loss, accuracy, predicted_labels.tolist(), true_labels.tolist()]


def test_maml_plus_plus(meta_model, test_loader, num_test_tasks=100):
    """Test MAML++ on unseen classes"""
    meta_model.eval()
    
    all_predicted_labels = []
    all_true_labels = []
    task_accuracies = []
    
    # 🔥 修復：MAML測試時也需要梯度來做內部適應
    for i, episode_data in enumerate(test_loader):
        if i >= num_test_tasks:
            break
            
        # 使用MAMLPlusPlusSolver進行測試
        predicted_labels, true_labels, _, _ = MAMLPlusPlusSolver(
            meta_model,
            None,  # No optimizer for testing
            episode_data,
            CFG.n_way,
            CFG.k_shot, 
            CFG.q_query,
            nn.CrossEntropyLoss(),
            inner_train_step=CFG.val_inner_train_step,
            train=False,
            return_labels=True
        )
        
        # Calculate task accuracy
        task_acc = (np.array(predicted_labels) == np.array(true_labels)).mean()
        task_accuracies.append(task_acc)
        
        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(true_labels)
    
    return all_predicted_labels, all_true_labels, task_accuracies


def main():
    """Main training function for MAML++."""
    print("=== MAML++ for Malware Classification ===")
    print(f"Device: {CFG.device}")
    
    # --- 1. Seed and Logger (MAML.py) ---
    set_random_seeds()
    
    # Load environment variables
    load_dotenv()
    
    logger = Logger()
    print(f"Logging experiment to: {logger.path}")

    # Download dataset if not exists
    if not os.path.exists(CFG.dataset_dir):
        download_dataset(dir=CFG.dataset_dir)

    # --- 2. Dataloaders (MODIFIED) ---
    print("Creating dataloaders...")
    
    dataloaders = create_meta_learning_dataloaders(
        features_dir=CFG.features_dir,
        split_csv_path=CFG.split_csv_path,
        n_way=CFG.n_way,
        k_shot=CFG.k_shot,
        q_query=CFG.q_query,
        train_episodes_per_epoch=200,  # 每個epoch的訓練episodes
        val_episodes_per_epoch=50,     # 每個epoch的驗證episodes  
        test_episodes_per_epoch=100,   # 測試episodes
        normalize=True,
        num_workers=0,
        pin_memory=(CFG.device == 'cuda'),
        seed=CFG.random_seed
    )
    
    # 🔥 直接獲取DataLoaders，無需wrapper
    train_loader = dataloaders['train']        # seen classes episodes
    val_loader = dataloaders['val']            # seen classes episodes
    test_loader = dataloaders['test_unseen']   # unseen classes episodes

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
        print(f"Auto-adjusting input_dim from {CFG.input_dim} to {actual_input_dim}")
        CFG.input_dim = actual_input_dim

    # --- 3. Model, Optimizer, Loss (Unchanged from MAML++) ---
    meta_model = MalwarePlusPlusClassifier(
        input_dim=CFG.input_dim, 
        hidden_dim=256,  # 使用更大的隱藏維度
        output_dim=CFG.n_way,
        num_inner_steps=CFG.train_inner_train_step
    ).to(CFG.device)

    # MAML++ IMPROVEMENT: Meta-optimizer with cosine annealing (CA)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=CFG.meta_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.max_epoch, eta_min=CFG.meta_lr * 0.01)

    loss_fn = nn.CrossEntropyLoss()

    print(f"MAML++ Model parameters: {sum(p.numel() for p in meta_model.parameters())}")
    print(f"Using first-order gradients for first {CFG.use_first_order_epochs} epochs")
    
    # Early stopping setup (調整for 200 epochs)
    early_stopping = EarlyStopping(patience=25, min_delta=0.005)  # 增加patience
    best_val_acc = 0
    no_improve_epochs = 0
    
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
        all_train_preds_epoch = []
        all_train_trues_epoch = []
  
        train_epoch_iterator = tqdm(train_loader, desc=f"Training")
        
        for episode_data in train_epoch_iterator:
            result = MAMLPlusPlusSolver(
                meta_model,
                optimizer,
                episode_data,  # 🔥 直接傳遞
                CFG.n_way,
                CFG.k_shot,
                CFG.q_query,
                loss_fn,
                inner_train_step=CFG.train_inner_train_step,
                train=True,
                epoch=epoch,
                step_weights=step_weights,
                use_first_order_epochs=CFG.use_first_order_epochs,
                return_labels=False
            )
            
            # result = [meta_loss, accuracy, predicted_labels, true_labels]
            meta_loss, accuracy, predicted_labels, true_labels = result
            
            train_meta_loss.append(meta_loss.item())
            all_train_preds_epoch.extend(predicted_labels)
            all_train_trues_epoch.extend(true_labels)
            
            train_epoch_iterator.set_postfix({
                "loss": f"{meta_loss.item():.3f}", 
                "acc": f"{accuracy*100:.2f}%"
            })

        avg_train_loss = np.mean(train_meta_loss)
        avg_train_acc = accuracy_score(all_train_trues_epoch, all_train_preds_epoch)
        avg_train_precision = precision_score(all_train_trues_epoch, all_train_preds_epoch, average='macro', zero_division=0)
        avg_train_recall = recall_score(all_train_trues_epoch, all_train_preds_epoch, average='macro', zero_division=0)
        avg_train_f1 = f1_score(all_train_trues_epoch, all_train_preds_epoch, average='macro', zero_division=0)
        print(f"Train Loss: {avg_train_loss:.3f}\tAccuracy: {avg_train_acc*100:.3f}%")
        print(f"Train F1 (Macro): {avg_train_f1:.3f} | Precision: {avg_train_precision:.3f} | Recall: {avg_train_recall:.3f}")
        
        # Validation
        meta_model.eval()
        
        #  lists to store all labels from all validation tasks
        all_val_preds = []
        all_val_trues = []
        all_val_loss = []
        
        val_epoch_iterator = tqdm(val_loader, desc="Validation")
        
        # 🔥 修復：MAML驗證時需要梯度來做內部適應，不能用no_grad
        for episode_data in val_epoch_iterator:  # 🔥 直接迭代
    
            # 🔥 直接傳遞episode_data
            predicted_labels, true_labels, meta_loss, batch_task_losses = MAMLPlusPlusSolver(
                meta_model,
                None,
                episode_data,  # 🔥 直接傳遞
                CFG.n_way,
                CFG.k_shot,
                CFG.q_query,
                loss_fn,
                inner_train_step=CFG.val_inner_train_step,
                train=False,
                epoch=epoch,
                step_weights=step_weights,
                use_first_order_epochs=CFG.use_first_order_epochs,
                return_labels=True 
            )
            
            all_val_preds.extend(predicted_labels)
            all_val_trues.extend(true_labels)
            all_val_loss.append(meta_loss.item())
        
        # Calculate metrics *after* the loop, on all collected labels
        avg_val_loss = np.mean(all_val_loss)
        avg_val_acc = accuracy_score(all_val_trues, all_val_preds)
        avg_val_precision = precision_score(all_val_trues, all_val_preds, average='macro', zero_division=0)
        avg_val_recall = recall_score(all_val_trues, all_val_preds, average='macro', zero_division=0)
        avg_val_f1 = f1_score(all_val_trues, all_val_preds, average='macro', zero_division=0)

        # Print progress with more detailed info (like MAML.py)
        print(f"Val Loss: {avg_val_loss:.3f} | Val Acc: {avg_val_acc*100:.2f}%")
        print(f"Train F1: {avg_train_f1:.3f} | Val F1: {avg_val_f1:.3f}")
        
        # Check for overfitting (like MAML.py)
        acc_gap = avg_train_acc - avg_val_acc
        if acc_gap > 0.3:
            print(f"Warning: Possible overfitting! Gap: {acc_gap*100:.1f}%")
        
        train_metrics = {
            "loss": avg_train_loss,
            "accuracy": avg_train_acc,
            "precision": avg_train_precision,
            "recall": avg_train_recall,
            "f1_macro": avg_train_f1
        }
        # Add all metrics to the log
        val_metrics = {
            "loss": avg_val_loss,
            "accuracy": avg_val_acc,
            "precision": avg_val_precision,
            "recall": avg_val_recall,
            "f1_macro": avg_val_f1
        }
        logger.add(epoch, train_metrics, val_metrics)
        
        # Save best model (enhanced like MAML.py)
        if logger.should_save_best(avg_val_acc):
            print(f"New best model found! Saving to {logger.model_path}")
            
            cfg_dict = {
                k: v for k, v in vars(CFG).items()
                if not k.startswith("__") and isinstance(
                    v, (int, float, str, bool, list, type(None))
                )
            }
            
            torch.save({
                "model_state_dict": meta_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg_dict,
                "epoch": epoch,
                "val_accuracy": avg_val_acc,
                "hyperparameters": {
                    'n_way': CFG.n_way,
                    'k_shot': CFG.k_shot,
                    'q_query': CFG.q_query,
                    'input_dim': CFG.input_dim,
                    'inner_lr': CFG.inner_lr,
                    'meta_lr': CFG.meta_lr,
                    'train_inner_train_step': CFG.train_inner_train_step,
                    'use_first_order_epochs': CFG.use_first_order_epochs
                },
            }, logger.model_path)
            
            print(f"Saved best model: {logger.model_path} "
                f"(val_accuracy={avg_val_acc*100:.2f}%)")
            best_val_acc = avg_val_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        # Update scheduler
        scheduler.step()
        
        # Early stopping check (like MAML.py)
        if early_stopping(avg_val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation accuracy: {early_stopping.best_val_acc*100:.2f}%")
            break
        
        # Hard stop after 40 epochs of no improvement (for 200 epoch training)
        if no_improve_epochs >= 40:
            print(f"Stopping: No improvement for {no_improve_epochs} epochs")
            break


    # --- 5. Testing (MODIFIED) ---
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    # Load best model for testing
    if os.path.exists(logger.model_path):
        print(f"Loading best model from {logger.model_path}")
        checkpoint = torch.load(logger.model_path, map_location=CFG.device)
        if 'model_state_dict' in checkpoint:
            meta_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            meta_model.load_state_dict(checkpoint)
    else:
        print("Warning: Best model not found. Testing with the final model.")
        
    test_predicted_labels, test_true_labels, test_task_accuracies = test_maml_plus_plus(
        meta_model, test_loader, num_test_tasks=20  # Reduced for testing
    )
    
    average_test_accuracy = np.mean(test_task_accuracies)
    std_test_accuracy = np.std(test_task_accuracies)
    
    print(f"MAML++ Final Test Results:")
    print(f"Average Test Task Accuracy: {average_test_accuracy*100:.3f}% ± {std_test_accuracy*100:.3f}%")
    print(f"Best Task Accuracy: {np.max(test_task_accuracies)*100:.3f}%")
    print(f"Worst Task Accuracy: {np.min(test_task_accuracies)*100:.3f}%")
    print(f"Total test samples: {len(test_predicted_labels)}")
    
    # Save MAML++ test results
    results_df = pd.DataFrame({
        'id': range(len(test_predicted_labels)),
        'predicted_class': test_predicted_labels,
        'true_class': test_true_labels
    })

    results_csv_path = os.path.join(CFG.log_dir, 'maml_plus_plus_test_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"MAML++ test results saved as {results_csv_path}")
    
    # Calculate detailed metrics (like MAML.py)
    print("\nClassification Report:")
    print(classification_report(test_true_labels, test_predicted_labels, 
                              target_names=[f'Class_{i}' for i in range(CFG.n_way)]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_true_labels, test_predicted_labels))
    
    # Training summary (like MAML.py)
    print("\nMAML++ Training completed!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Logs saved at: {logger.path}")
    print(f"Best model saved at: {logger.model_path}")
    print(f"Test results saved at: {results_csv_path}")


if __name__ == "__main__":
    main()