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


# ============================================================================
# CONFIGURATION
# ============================================================================

class CFG:
    """Configuration parameters for MAML meta-learning."""
    n_way = 3
    k_shot = 5  # Reduced for better stability
    q_query = 15
    input_dim = 1280

    # 🔥 修復：調整學習率參數
    inner_lr = 0.01  # 提高 inner learning rate
    meta_lr = 0.001
    inner_steps_train = 3  # 增加 inner steps
    inner_steps_val = 3

    meta_batch_size = 64  # Reduced for stability
    max_epoch = 100  # Increased for better convergence
    eval_batches = 20
    grad_clip = 10.0

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

if not hasattr(CFG, 'eps'):
    CFG.eps = 1e-6


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


# ============================================================================
# DATASET WRAPPER - Using standardized dataloader
# ============================================================================

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

# ============================================================================
# MODEL
# ============================================================================

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
    
    # MAML.py (修改後)
    def functional_forward(self, x, params):
        """🔥 修復：正確的函數式前向傳播實現 - 使用提供的參數"""
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
            # --- 結束 ---
            
            elif isinstance(module, nn.BatchNorm1d):
                # (這個分支現在不會被觸發了，但保留也無妨)
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


# ============================================================================
# 🔥 修復：MAML ALGORITHM - 核心修復
# ============================================================================
# MAML.py (替換整個函數)

def maml_step_with_preds(model: nn.Module,
                         criterion: nn.Module,
                         task_batch_normalized: torch.Tensor,
                         train: bool
                         ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    執行單個 MAML 任務（內循環和外循環損失計算）。
    
    🔥 此版本已修改為「正確」的 FO-MAML (First-Order MAML)。
    """
    
    # 1. 將 task_batch 分割為 support/query
    # task_batch_normalized Shape: [1, N*(K+Q), D]
    task_tensor = task_batch_normalized.squeeze(0) # Shape: [N*(K+Q), D]

    # (N*K) support samples
    # support_indices = np.zeros(task_tensor.size(0), dtype=bool)
    # support_indices[np.arange(CFG.n_way) * (CFG.k_shot + CFG.q_query) < CFG.k_shot] = True
    
    # 這是錯誤的！上面的索引是錯的。必須使用 CFG 中的原始邏輯
    support_indices = []
    query_indices = []
    for n in range(CFG.n_way):
        # 每個類別的起始索引
        class_start_idx = n * (CFG.k_shot + CFG.q_query)
        # K-shot support
        support_indices.extend(range(class_start_idx, class_start_idx + CFG.k_shot))
        # Q-query query
        query_indices.extend(range(class_start_idx + CFG.k_shot, class_start_idx + CFG.k_shot + CFG.q_query))
    
    support_x = task_tensor[support_indices] # [N*K, D]
    query_x = task_tensor[query_indices]     # [N*Q, D]

    # 2. 創建標籤
    support_y = torch.arange(CFG.n_way).repeat_interleave(CFG.k_shot).to(CFG.device) # [N*K]
    query_y = torch.arange(CFG.n_way).repeat_interleave(CFG.q_query).to(CFG.device)   # [N*Q]
    
    # ---
    # 3. 內循環 (Task-specific Adaptation)
    # ---
    
    # 3.1. 複製模型權重 (θ -> θ')
    fast_weights = OrderedDict(model.named_parameters())
    
    # 根據 train/val 設置內循環步驟
    inner_steps = CFG.inner_steps_train if train else CFG.inner_steps_val
    
    for inner_step in range(inner_steps):
        
        # 3.2.1. 在 Support set (K-shot) 上計算損失
        support_logits = model.functional_forward(support_x, fast_weights)
        support_loss = criterion(support_logits, support_y)
        
        # 3.2.2. 計算內循環梯度 (∇_θ)
        # 🔥 FO-MAML 修正 #1：
        #    - 設置 create_graph=False。
        #    - 這會切斷計算圖，使其成為一階近似。
        grads = torch.autograd.grad(support_loss, 
                                    fast_weights.values(), 
                                    create_graph=False) # <--- FO-MAML
        
        # 3.2.3. 手動更新快速權重 (θ')
        fast_weights = OrderedDict(
            (name, param - CFG.inner_lr * grad)
            for ((name, param), grad) in zip(fast_weights.items(), grads)
        )
        
    # --- (內循環結束) ---

    # 3.3. 計算外循環 (Query) 損失 (使用最終的 fast_weights, θ')
    query_logits = model.functional_forward(query_x, fast_weights)
    query_loss = criterion(query_logits, query_y)
    
    # meta_loss 是 L_Q(θ')
    meta_loss = query_loss 
    
    # (計算準確率和預測值)
    with torch.no_grad():
        preds = torch.argmax(query_logits, dim=1)
        labels = query_y
        task_acc = (preds == labels).sum().item() / len(labels)


    # 3.4. 🔥 FO-MAML Meta-Update (外循環)
    if train:
        # 🔥 FO-MAML 修正 #2：
        #    - 我們不能使用 meta_loss.backward()，因為計算圖已在內循環中被切斷。
        #    - 我們必須手動計算 ∇_θ' L_Q(θ')
        #    - 並將該梯度分配給「原始模型」的 .grad 屬性。

        # (A) 計算 Query Loss 對「快速權重 (θ')」的梯度
        meta_grads = torch.autograd.grad(meta_loss, 
                                         fast_weights.values())
        
        # (B) 獲取「原始模型 (θ)」的參數
        params = model.parameters()

        # (C) 手動將 meta_grads 分配給原始參數的 .grad 字段
        #    run_epoch 中的 meta_optimizer.step() 將會使用這些梯度。
        for param, meta_grad in zip(params, meta_grads):
            if param.grad is None:
                param.grad = meta_grad.detach()
            else:
                # 處理 meta_batch_size > 1 時的梯度累積
                param.grad += meta_grad.detach()

    # 4. 返回
    return meta_loss, task_acc, preds.detach().cpu(), labels.detach().cpu()

# MAML.py (新增: 獨立的測試預測函數)
def get_task_predictions(model, task_tensor, loss_fn, cfg, inner_steps):
    """
    對單個任務執行 MAML 內循環適應，並返回 Query Set 上的預測標籤和真實標籤。
    """
    
    # 數據切割修復 (與 maml_step 相同)
    task_tensor = task_tensor.view(
        cfg.n_way, cfg.k_shot + cfg.q_query, -1
    ).squeeze(0) # [N_way, K+Q, D] -> [N*(K+Q), D]

    # 拆分 Features (x)
    support_x = task_tensor[:, :cfg.k_shot, :].contiguous().view(-1, cfg.input_dim)
    query_x = task_tensor[:, cfg.k_shot:, :].contiguous().view(-1, cfg.input_dim)
    
    # 生成 Labels (y)
    support_y = create_label(cfg.n_way, cfg.k_shot).to(cfg.device)
    query_y = create_label(cfg.n_way, cfg.q_query).to(cfg.device)

    # 1. 內循環初始化：獲取初始參數 \theta
    fast_weights = OrderedDict()
    for name, param in model.named_parameters():
            if param.requires_grad:  # <--- 修正：包含所有可訓練參數
                fast_weights[name] = param.clone()

    # --- 內循環 (Inner Loop) 適應 ---
    current_fast_weights = fast_weights
    
    for k in range(inner_steps):
        support_logits = model.functional_forward(support_x, current_fast_weights)
        support_loss = loss_fn(support_logits, support_y.long())
        
        # 這裡不要求二次梯度 (create_graph=False)
        grad = torch.autograd.grad(
            support_loss, 
            current_fast_weights.values(), 
            create_graph=False, 
            # allow_unused=True
        )

        # 參數更新
        current_fast_weights = OrderedDict(
            (name, param - cfg.inner_lr * g if g is not None else param)
            for ((name, param), g) in zip(current_fast_weights.items(), grad)
        )

    # --- 外循環 (Outer Loop) 預測 ---
    model.eval()
    with torch.no_grad():
        query_logits = model.functional_forward(query_x, current_fast_weights)
        query_pred = torch.argmax(query_logits, dim=1).cpu().numpy()
        
    # 返回 Query Set 上的預測和真實標籤
    return query_pred.tolist(), query_y.cpu().numpy().tolist()

# MAML.py (修正後的 run_epoch)
# 請用這個版本替換你原有的 run_epoch 函數
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
    🔥 [數據洩漏修復版本]
    執行一個完整的 epoch。
    此版本在迴圈內部對每個 batch 執行「正確」的 Z-Score 歸一化
    (僅使用 Support Set 統計數據)
    """
    
    # 1. 初始化
    if train:
        meta_optimizer.zero_grad()

    epoch_meta_loss = 0.0
    all_preds = []
    all_labels = []
    
    # ==================
    # 🚨 數據診斷代碼 (V3 - StandardScaler 檢查)
    # ==================
    # (我們保留它，用來查看 "原始" 數據)
    # print("\n" + "="*30)
    # print(f"Running diagnostics (Mode: {mode}) - Checking RAW data...")
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
                print("❌ 錯誤: Dataloader's batch is an empty list/tuple.")
        else:
            # print("  Batch is not a list/tuple. Assuming it's the tensor itself.")
            support_x = diag_batch.to(CFG.device)

        # (註釋掉詳細的數據日誌，保持簡潔)
        # if support_x is not None:
        #     print(f"Data diagnostics (RAW support_x):")
        #     ...
            
    except Exception as e:
        print(f"❌ 數據診斷失敗: {e}")
    print("="*30 + "\n")
    
    # 2. 迭代任務
    epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.max_epoch} [{mode}]", leave=False)
    
    for i, task_batch in enumerate(epoch_iterator):
        
        # 🔥 ====================
        # 🔥 數據洩漏修復 (Data Leakage Fix)
        # 🔥 (x - mean_support) / (std_support + eps)
        # 🔥 ====================
        
        # 步驟 A: 將 task_batch 移到 device
        task_batch = task_batch.to(CFG.device) # Shape [1, N*(K+Q), D]
        
        # 步驟 B: 檢查 NaN 或 Inf
        if torch.isnan(task_batch).any() or torch.isinf(task_batch).any():
            if hasattr(CFG, 'verbose') and CFG.verbose:
                print(f"\nWarning: Skipping task {i} due to NaN/Inf in raw data.")
            continue

        # Squeeze to [N*(K+Q), D] for normalization
        task_tensor_raw = task_batch.squeeze(0)

        # 步驟 C: 🔥 (關鍵) 獲取 Support/Query 索引
        # (此邏輯必須與 maml_step_with_preds 內部一致)
        support_indices = []
        query_indices = []
        for n in range(CFG.n_way):
            class_start_idx = n * (CFG.k_shot + CFG.q_query)
            support_indices.extend(range(class_start_idx, class_start_idx + CFG.k_shot))
            query_indices.extend(range(class_start_idx + CFG.k_shot, class_start_idx + CFG.k_shot + CFG.q_query))

        support_x_raw = task_tensor_raw[support_indices] # [N*K, D]
        query_x_raw = task_tensor_raw[query_indices]     # [N*Q, D]
        
        # 步驟 D: 🔥 (關鍵) *僅* 從 Support Set 計算統計數據
        mean = torch.mean(support_x_raw, dim=0) # Shape: [D]
        std = torch.std(support_x_raw, dim=0)   # Shape: [D]
        
        # 步驟 E: 🔥 (關鍵) 應用 Z-Score 到 Support 和 Query
        support_x_norm = (support_x_raw - mean) / (std + CFG.eps)
        query_x_norm = (query_x_raw - mean) / (std + CFG.eps)
        
        # 替換 NaN (如果某些特徵的 std 為 0)
        support_x_norm = torch.nan_to_num(support_x_norm, nan=0.0)
        query_x_norm = torch.nan_to_num(query_x_norm, nan=0.0)

        # 步驟 F: 🔥 (關鍵) 重建 Task Tensor
        # 創建一個與 task_tensor_raw 相同形狀的張量
        task_tensor_normalized = torch.zeros_like(task_tensor_raw)
        
        # 將歸一化後的數據放回
        task_tensor_normalized[support_indices] = support_x_norm
        task_tensor_normalized[query_indices] = query_x_norm
        
        # 重新添加 batch 維度 [1, N*(K+Q), D] 以便傳遞
        task_batch_normalized = task_tensor_normalized.unsqueeze(0)
        
        # --- (修復結束) ---

        # 3. 執行單一任務
        #    傳遞「已正確歸一化」的 batch
        meta_loss, task_acc, preds, labels = maml_step_with_preds(
            model, criterion, task_batch_normalized, train=train
        )
        
        # 4. 累加統計數據
        meta_loss_float = meta_loss.item()

        if np.isnan(meta_loss_float):
            print(f"\nWarning: Detected 'nan' loss in epoch {epoch+1} [{mode}] at task {i}. Stopping epoch early.")
            epoch_meta_loss = np.nan
            break 
        
        epoch_meta_loss += meta_loss_float
        all_preds.extend(preds)
        all_labels.extend(labels)

        # 5. [MAML 核心更新邏輯]
        if train and (i + 1) % CFG.meta_batch_size == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.grad_clip)
            meta_optimizer.step()
            meta_optimizer.zero_grad()
        
        # 6. 更新 tqdm 進度條
        epoch_iterator.set_postfix(
            meta_loss=f"{epoch_meta_loss / (i + 1):.4f}", 
            task_acc=f"{task_acc:.4f}"
        )

    # 7. 處理 epoch 結束時剩餘的梯度
    if train and (i + 1) % CFG.meta_batch_size != 0 and not np.isnan(epoch_meta_loss):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.grad_clip)
        meta_optimizer.step()
        meta_optimizer.zero_grad()

    # --- 以下日誌記錄邏輯 ---

    # 8. 計算整個 epoch 的平均指標
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

    # 計算整體指標
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

# MAML.py (替換您的 'test_model' 函數 - 第 821 行)
def test_model(model, test_loader, num_test_tasks=20):
    """Test the trained model and return detailed results."""
    test_iter = iter(test_loader)
    
    all_predicted_labels = []
    all_true_labels = []
    task_accuracies = []

    print("Starting testing (collecting predictions for detailed report)...")
    device = CFG.device # 確保 device 已定義

    for batch_idx in tqdm(range(num_test_tasks), desc="Testing"):
        # 1. 獲取單個任務的 meta-batch (Batch size = 1)
        x, test_iter = get_meta_batch(1, CFG.k_shot, CFG.q_query, test_loader, test_iter)
        
        # x shape: [1, N*(K+Q), D]
        task_tensor_raw = x[0] # [N*(K+Q), D]

        # 🔥 ====================
        # 🔥 數據洩漏修復 (Data Leakage Fix)
        # 🔥 必須在測試時应用與訓練/驗證時相同的歸一化！
        # 🔥 ====================

        # (1) 將張量移至 Device
        task_tensor_raw = task_tensor_raw.to(device)
        
        # (2) 檢查 NaN/Inf (安全起見)
        if torch.isnan(task_tensor_raw).any() or torch.isinf(task_tensor_raw).any():
            if hasattr(CFG, 'verbose') and CFG.verbose:
                print(f"\nWarning: Skipping test task {batch_idx} due to NaN/Inf in raw data.")
            continue
            
        # (3) 🔥 (關鍵) 獲取 Support/Query 索引
        support_indices = []
        query_indices = []
        for n in range(CFG.n_way):
            class_start_idx = n * (CFG.k_shot + CFG.q_query)
            support_indices.extend(range(class_start_idx, class_start_idx + CFG.k_shot))
            query_indices.extend(range(class_start_idx + CFG.k_shot, class_start_idx + CFG.k_shot + CFG.q_query))
        
        support_x_raw = task_tensor_raw[support_indices]
        query_x_raw = task_tensor_raw[query_indices]

        # (4) 🔥 (關鍵) *僅* 從 Support Set 計算統計數據
        mean = torch.mean(support_x_raw, dim=0)
        std = torch.std(support_x_raw, dim=0)

        # (5) 🔥 (關鍵) 應用 Z-Score 到 Support 和 Query
        support_x_norm = (support_x_raw - mean) / (std + CFG.eps)
        query_x_norm = (query_x_raw - mean) / (std + CFG.eps)
        
        # 替換 NaN (如果 std 為 0)
        support_x_norm = torch.nan_to_num(support_x_norm, nan=0.0)
        query_x_norm = torch.nan_to_num(query_x_norm, nan=0.0)

        # (6) 🔥 (關鍵) 重建 Task Tensor
        task_tensor_normalized = torch.zeros_like(task_tensor_raw)
        task_tensor_normalized[support_indices] = support_x_norm
        task_tensor_normalized[query_indices] = query_x_norm
        
        # --- (修復結束) ---

        # 3. 🔥 呼叫預測函數（傳入「已正確歸一化」的張量）
        #    注意：get_task_predictions 函數現在接收的是一個
        #    扁平化的 [N*(K+Q), D] 張量，它內部的邏輯
        #    (MAML.py 第 512 行) 需要正確處理這個。
        #    我們傳遞的 task_tensor_normalized 已經
        #    是 [N*(K+Q), D] 形狀，這是正確的。
        predicted_labels, true_labels = get_task_predictions(
            model,
            task_tensor_normalized, # <--- 已修復
            nn.CrossEntropyLoss(),
            CFG,
            CFG.inner_steps_val
        )

        # 4. 計算當前任務準確度
        task_true = np.array(true_labels)
        task_pred = np.array(predicted_labels)
        task_acc = (task_true == task_pred).mean()
        task_accuracies.append(task_acc)

        # 5. 收集所有標籤
        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(true_labels)

    return all_predicted_labels, all_true_labels, task_accuracies
# =========================================================================
# MAIN ENTRY POINT - 保持原有邏輯
# ============================================================================

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

    # 在創建模型之前檢查數據維度
    sample_task = next(iter(train_loader))
    actual_input_dim = sample_task.shape[-1]
    
    print(f"檢測到實際輸入維度: {actual_input_dim}")
    
    if actual_input_dim != CFG.input_dim:
        print(f"自動調整input_dim從 {CFG.input_dim} 到 {actual_input_dim}")
        CFG.input_dim = actual_input_dim

    # Initialize model, optimizer, and logger
    model = MalwareClassifier(CFG.input_dim, output_dim=CFG.n_way).to(CFG.device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=CFG.meta_lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss().to(CFG.device)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    print("🔍 Model parameter names:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    print()

    print(f"🔥 FIXED MAML Configuration:")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Using standardized malware dataset with {CFG.input_dim} features")
    print(f"Task configuration: {CFG.n_way}-way {CFG.k_shot}-shot with {CFG.q_query} query samples")
    print(f"Inner LR: {CFG.inner_lr}, Meta LR: {CFG.meta_lr}, Inner steps: {CFG.inner_steps_train}")

    # Create iterators
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # 🔥 添加早停和學習率調度
    early_stopping = EarlyStopping(patience=15, min_delta=0.01)
    initial_inner_lr = CFG.inner_lr
    best_val_acc = 0
    no_improve_epochs = 0

    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        meta_optimizer, mode='max', factor=0.8, patience=10
    )

    # Training loop
    print("\n" + "="*60)
    print("🔥 STARTING FIXED MAML TRAINING")
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
            mode="Train",  # <--- 新增 "mode" 參數
            train=True
        )

        # Log training metrics
        train_loss = train_metrics["loss"]
        train_acc = train_metrics["accuracy"]
        train_f1 = train_metrics["f1_macro"]
        val_metrics = run_epoch(
            model=model,
            meta_optimizer=meta_optimizer, # 雖然 val 不用, 但函數簽名需要
            criterion=criterion,
            dataloader=val_loader,
            epoch=epoch,
            mode="Validation", # <--- 新增 "mode" 參數
            train=False
        )
        
        # Log validation metrics
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_macro"]

        # Print progress with more detailed info (已更新)
        print(f"Train Loss: {train_metrics['loss']:.3f} | Train Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.3f} | Val Acc: {val_metrics['accuracy']*100:.2f}%")
        print(f"Train F1: {train_metrics.get('f1_macro', 0):.3f} | Val F1: {val_metrics.get('f1_macro', 0):.3f}")
        
        # 🔥 動態學習率調整
        if epoch % 20 == 0 and epoch > 0:
            CFG.inner_lr = max(CFG.inner_lr * 0.8, 0.001)
            # print(f"📉 Reduced inner learning rate to: {CFG.inner_lr:.4f}")
        
        # 🔥 檢查過擬合
        acc_gap = train_metrics['accuracy'] - val_metrics['accuracy']
        if acc_gap > 0.3:
            print(f"⚠️  Warning: Possible overfitting! Gap: {acc_gap*100:.1f}%")
        
        # 🔥 更新學習率調度器
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
            
            print(f"✅ Saved best model: {logger.model_path} "
                f"(val_accuracy={val_metrics['accuracy']*100:.2f}%)")
        else:
            no_improve_epochs += 1
        
        # 🔥 早停檢查
        if early_stopping(val_metrics['accuracy']):
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            print(f"📊 Best validation accuracy: {early_stopping.best_val_acc*100:.2f}%")
            break
        
        # 🔥 額外的停止條件
        if no_improve_epochs >= 25:
            print(f"🛑 Stopping: No improvement for {no_improve_epochs} epochs")
            break
        
        # 🔥 進度提示
        if val_metrics['accuracy'] > 0.5:
            print("🎉 Validation accuracy > 50% - Excellent progress!")
        elif val_metrics['accuracy'] > 0.4:
            print("🎉 Validation accuracy > 40% - Good progress!")
        elif val_metrics['accuracy'] > 0.35:
            print("📈 Validation accuracy > 35% - Making progress!")
        elif val_metrics['accuracy'] > 0.33:
            print("📊 Validation accuracy improving from random baseline!")

    # Testing phase
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    test_predicted_labels, test_true_labels, test_task_accuracies = test_model(
        model, test_loader, num_test_tasks=20
    )
    
    average_test_accuracy = np.mean(test_task_accuracies)
    print(f"📊 Average Test Task Accuracy: {average_test_accuracy*100:.3f}%")
    print(f"📊 Total test samples: {len(test_predicted_labels)}")

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
    print(f"\n✅ FIXED MAML Training completed!")
    print(f"✅ Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"✅ Logs saved at: {logger.path}")
    print(f"✅ Best model saved at: {logger.model_path}")
    print(f"✅ Test results saved at: {results_csv_path}")


if __name__ == "__main__":
    main()