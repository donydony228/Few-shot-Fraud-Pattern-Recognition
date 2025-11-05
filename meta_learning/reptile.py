"""
Reptile Baseline (Nichol et al., 2018)
Faithful implementation for N-way K-shot classification
- No cosine head / normalization / pretrain
- Evaluates best model on seen, unseen, and generalized splits
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Path to src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extraction.data_loader import (
    create_meta_learning_dataloaders,
    split_episode_to_support_query
)

# =========================================================
# CONFIG
# =========================================================
class CFG:
    # Task setup
    n_way = 3
    k_shot = 5
    q_query = 5
    input_dim = 1280

    # Inner/outer loop
    inner_lr = 0.01
    inner_steps = 5
    meta_lr = 0.001
    meta_batch_size = 8

    # Training control
    train_episodes_per_epoch = 200
    val_episodes_per_epoch = 60
    test_episodes_per_epoch = 100
    max_epoch = 200
    eval_batches = 10

    # Early stop & LR decay
    early_stopping_patience = 15
    lr_decay_patience = 10
    lr_decay_factor = 0.5
    min_lr = 1e-6

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # IO
    features_dir = "../MalVis_dataset_small/features"
    split_csv_path = "../MalVis_dataset_small/label_split.csv"
    log_dir = "logs"

    # System
    seed = 42
    num_workers = 0
    pin_memory = torch.cuda.is_available()

# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_label(n_way: int, num_per_class: int) -> torch.Tensor:
    return torch.arange(n_way).repeat_interleave(num_per_class).long()

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy()).mean()

def calculate_metrics(preds: List[int], labels: List[int], num_classes: int) -> Dict[str, Any]:
    preds = np.array(preds)
    labels = np.array(labels)
    return {
        "acc": (preds == labels).mean(),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "cm": confusion_matrix(labels, preds).tolist(),
    }

class Logger:
    def __init__(self, name: str = "reptile_baseline"):
        os.makedirs(CFG.log_dir, exist_ok=True)
        ts = timestamp()
        self.path = os.path.join(CFG.log_dir, f"{name}_{ts}.json")
        self.model_path = os.path.join(CFG.log_dir, f"{name}_best_{ts}.pth")
        self.logs = {"epochs": []}
        self.best_val = -1.0

    def add(self, epoch: int, train: Dict[str, Any], val: Dict[str, Any]):
        self.logs["epochs"].append({"epoch": epoch, "train": train, "val": val})
        with open(self.path, "w") as f:
            json.dump(self.logs, f, indent=2)

# =========================================================
# Model (simple MLP)
# =========================================================
class MalwareNet(nn.Module):
    def __init__(self, input_dim=1280, hidden=512, n_way=3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_way)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.norm(x)
        return self.net(x)

# =========================================================
# Reptile Step (faithful to original)
# =========================================================
def reptile_step(model: nn.Module,
                 x: torch.Tensor,
                 loss_fn: nn.Module,
                 train: bool = True) -> Tuple[float, float, List[int], List[int]]:
    n_way, k, q = CFG.n_way, CFG.k_shot, CFG.q_query

    theta0 = {n: p.data.clone() for n, p in model.named_parameters()}
    meta_delta = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    all_preds, all_labels = [], []
    task_losses, task_accs = [], []

    for task in x:
        # Reset weights for this task
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(theta0[n])

        episode = task.squeeze(0) if task.dim() == 4 else task
        support, query, y_s, y_q = split_episode_to_support_query(episode, k_shot=k, q_query=q)
        support, query, y_s, y_q = support.to(CFG.device), query.to(CFG.device), y_s.to(CFG.device), y_q.to(CFG.device)

        # Inner loop (adaptation)
        for _ in range(CFG.inner_steps):
            model.train()
            out = model(support)
            loss = loss_fn(out, y_s)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= CFG.inner_lr * p.grad

        # After adaptation: θ'
        adapted = {n: p.data.clone() for n, p in model.named_parameters()}

        # Evaluate on query
        model.eval()
        with torch.no_grad():
            q_out = model(query)
            q_loss = loss_fn(q_out, y_q)
            acc = calculate_accuracy(q_out, y_q)
            preds = torch.argmax(q_out, -1).cpu().numpy()
        task_losses.append(q_loss.item())
        task_accs.append(acc)
        all_preds.extend(preds)
        all_labels.extend(y_q.cpu().numpy())

        # accumulate meta update
        with torch.no_grad():
            for n, p in model.named_parameters():
                meta_delta[n] += adapted[n] - theta0[n]

    if train:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data = theta0[n] + CFG.meta_lr * meta_delta[n] / len(x)

    return np.mean(task_losses), np.mean(task_accs), all_preds, all_labels

# =========================================================
# Epoch loop
# =========================================================
def run_epoch(model: nn.Module,
              loader: DataLoader,
              loss_fn: nn.Module,
              train: bool,
              desc: str) -> Dict[str, Any]:
    losses, preds, labels = [], [], []
    num_batches = CFG.train_episodes_per_epoch // CFG.meta_batch_size if train else CFG.eval_batches
    it = iter(loader)
    for _ in tqdm(range(num_batches), desc=desc):
        meta_batch = []
        for _ in range(CFG.meta_batch_size):
            try:
                ep = next(it)
            except StopIteration:
                it = iter(loader)
                ep = next(it)
            meta_batch.append(ep.squeeze(0).to(CFG.device))
        x = torch.stack(meta_batch)
        loss, acc, p, y = reptile_step(model, x, loss_fn, train=train)
        losses.append(loss)
        preds.extend(p)
        labels.extend(y)
    m = calculate_metrics(preds, labels, CFG.n_way)
    m["loss"] = float(np.mean(losses))
    return m

# =========================================================
# Loader helper
# =========================================================
def pick_loader(dls: Dict[str, DataLoader], keys: List[str], name: str):
    for k in keys:
        if k in dls:
            return dls[k]
    raise KeyError(f"Missing {name} loader. Available keys: {list(dls.keys())}")

# =========================================================
# Main
# =========================================================
def main():
    set_seed(CFG.seed)
    print(f"[Reptile_Baseline] device={CFG.device}  n_way={CFG.n_way}  k_shot={CFG.k_shot}")

    model = MalwareNet(CFG.input_dim, hidden=512, n_way=CFG.n_way).to(CFG.device)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    # Create dataloaders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.abspath(os.path.join(script_dir, CFG.features_dir))
    split_csv_path = os.path.abspath(os.path.join(script_dir, CFG.split_csv_path))
    dataloaders = create_meta_learning_dataloaders(
        features_dir=features_dir,
        split_csv_path=split_csv_path,
        n_way=CFG.n_way,
        k_shot=CFG.k_shot,
        q_query=CFG.q_query,
        train_episodes_per_epoch=CFG.train_episodes_per_epoch,
        val_episodes_per_epoch=CFG.val_episodes_per_epoch,
        test_episodes_per_epoch=CFG.test_episodes_per_epoch,
        normalize=True,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        seed=CFG.seed
    )

    train_loader = pick_loader(dataloaders, ["train"], "train")
    val_loader = pick_loader(dataloaders, ["val"], "val")
    test_seen = pick_loader(dataloaders, ["test_seen"], "test_seen")
    test_unseen = pick_loader(dataloaders, ["test_unseen"], "test_unseen")
    test_general = pick_loader(dataloaders, ["test_generalized"], "test_generalized")

    print(f"[DataLoader] Ready: train/val/test splits loaded.")

    best_val, patience, lr_wait = -1.0, 0, 0
    save_path = logger.model_path
    current_meta_lr = CFG.meta_lr

    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        print(f"Current meta_lr: {current_meta_lr:.6f}")

        train_m = run_epoch(model, train_loader, loss_fn, train=True, desc="Train")
        val_m = run_epoch(model, val_loader, loss_fn, train=False, desc="Val")

        print(f"Train: acc={train_m['acc']*100:.2f}% loss={train_m['loss']:.4f}")
        print(f"Val  : acc={val_m['acc']*100:.2f}% loss={val_m['loss']:.4f}")
        logger.add(epoch, train_m, val_m)

        if val_m["acc"] > best_val:
            best_val = val_m["acc"]
            patience, lr_wait = 0, 0
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"✓ Saved best model (val_acc={best_val*100:.2f}%)")
        else:
            patience += 1
            lr_wait += 1
            print(f"No improvement ({patience}/{CFG.early_stopping_patience})")

        if lr_wait >= CFG.lr_decay_patience:
            current_meta_lr = max(current_meta_lr * CFG.lr_decay_factor, CFG.min_lr)
            CFG.meta_lr = current_meta_lr
            lr_wait = 0
            print(f"⚠ meta_lr decayed to {current_meta_lr:.6f}")

        if patience >= CFG.early_stopping_patience:
            print(f"⚠ Early stopping. Best val_acc={best_val*100:.2f}%")
            break

    print(f"\n✅ Training done. Best val_acc={best_val*100:.2f}%")
    print(f"Model saved at: {save_path}")

    # ======================
    # Testing (best model)
    # ======================
    if os.path.exists(save_path):
        state = torch.load(save_path, map_location=CFG.device)
        model.load_state_dict(state["model_state_dict"])
        print("\n[Eval] Loaded best model for testing.")

    def evaluate(name: str, loader: DataLoader):
        m = run_epoch(model, loader, loss_fn, train=False, desc=name)
        print(f"{name}: acc={m['acc']*100:.2f}% loss={m['loss']:.4f}")
        return m

    print("\n=== Final Evaluation ===")
    seen_m = evaluate("test_seen", test_seen)
    unseen_m = evaluate("test_unseen", test_unseen)
    general_m = evaluate("test_generalized", test_general)

    print("\n====== Test Summary ======")
    print(f"Seen:        {seen_m['acc']*100:.2f}%")
    print(f"Unseen:      {unseen_m['acc']*100:.2f}%")
    print(f"Generalized: {general_m['acc']*100:.2f}%")
    print("==========================")

if __name__ == "__main__":
    main()
