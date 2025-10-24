import os, random, json
from collections import OrderedDict
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# =========================================================
# CONFIG
# =========================================================
class CFG:
    n_way = 3
    k_shot = 1
    q_query = 5
    input_dim = 1280

    inner_lr = 0.05
    meta_lr = 0.1
    inner_steps = 5

    meta_batch_size = 8
    max_epoch = 20
    eval_batches = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_json = "../malware_data_structure.json"
    log_dir = "logs"


# =========================================================
# UTILITIES
# =========================================================
def create_label(n_way, num_per_class):
    return torch.arange(n_way).repeat_interleave(num_per_class).long()

def calculate_accuracy(logits, labels):
    return (torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy()).mean()

def calculate_metrics(preds, labels, num_classes):
    preds, labels = np.array(preds), np.array(labels)
    metrics = {
        "acc": (preds == labels).mean(),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "cm": confusion_matrix(labels, preds).tolist(),
    }
    return metrics

class Logger:
    def __init__(self):
        os.makedirs(CFG.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(CFG.log_dir, f"reptile_{ts}.json")
        self.logs = {"config": vars(CFG), "epochs": []}

    def add(self, epoch, train, val):
        self.logs["epochs"].append({"epoch": epoch, "train": train, "val": val})
        with open(self.path, "w") as f:
            json.dump(self.logs, f, indent=2)


# =========================================================
# DATASET
# =========================================================
class MalwareDataset(Dataset):
    def __init__(self, json_path, split, k_shot, q_query):
        with open(json_path, "r") as f:
            self.data = json.load(f)[split]
        self.classes = list(self.data.keys())
        self.k_shot, self.q_query = k_shot, q_query
        self.normal = "benign"

    def __getitem__(self, idx):
        np.random.seed(42 + idx)
        frauds = [c for c in self.classes if c != self.normal]
        selected = np.random.choice(frauds, 2, replace=False).tolist() + [self.normal]

        task = []
        for cls in selected:
            files = self.data[cls]
            need = self.k_shot + self.q_query
            chosen = np.random.choice(files, need, replace=(len(files) < need))
            cls_features = []
            for f in chosen:
                f = self._fix(f)
                try:
                    arr = np.load(f)
                    arr = arr.flatten() if arr.ndim > 1 else arr
                    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
                except:
                    arr = np.zeros(CFG.input_dim)
                cls_features.append(arr)
            task.append(torch.tensor(np.stack(cls_features), dtype=torch.float32))
        return torch.stack(task)  # [n_way, k+q, feat]

    def _fix(self, path):
        if os.path.exists(path):
            return path
        for p in ["../", "../../", "./"]:
            if os.path.exists(os.path.join(p, path)):
                return os.path.abspath(os.path.join(p, path))
        return path

    def __len__(self):
        return 200


def get_meta_batch(loader, iterator):
    batch = []
    for _ in range(CFG.meta_batch_size):
        try:
            task = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            task = next(iterator)
        batch.append(task.squeeze(0).to(CFG.device))
    return torch.stack(batch), iterator


# =========================================================
# MODEL
# =========================================================
class MalwareNet(nn.Module):
    def __init__(self, input_dim=1280, hidden=512, n_way=3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_way),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.norm(x)
        return self.net(x)


# =========================================================
# REPTILE SOLVER
# =========================================================
def reptile_step(model, x, loss_fn, train=True):
    n_way, k, q = CFG.n_way, CFG.k_shot, CFG.q_query
    theta0 = {n: p.data.clone() for n, p in model.named_parameters()}
    meta_delta = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    task_losses, task_accs, all_preds, all_labels = [], [], [], []

    for task in x:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(theta0[n])

        support = task[:, :k, :].reshape(n_way * k, -1)
        query = task[:, k:, :].reshape(n_way * q, -1)
        y_s, y_q = create_label(n_way, k).to(CFG.device), create_label(n_way, q).to(CFG.device)

        model.train()
        for _ in range(CFG.inner_steps):
            out = model(support)
            loss = loss_fn(out, y_s)
            model.zero_grad(); loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= CFG.inner_lr * p.grad

        adapted = {n: p.data.clone() for n, p in model.named_parameters()}

        model.eval()
        with torch.no_grad():
            q_out = model(query)
            q_loss = loss_fn(q_out, y_q)
            acc = calculate_accuracy(q_out, y_q)
            preds = torch.argmax(q_out, -1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(y_q.cpu().numpy())

        task_losses.append(q_loss.item()); task_accs.append(acc)
        with torch.no_grad():
            for n, p in model.named_parameters():
                meta_delta[n] += adapted[n] - theta0[n]

    if train:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data = theta0[n] + CFG.meta_lr * meta_delta[n] / len(x)

    return np.mean(task_losses), np.mean(task_accs), all_preds, all_labels


# =========================================================
# TRAINER
# =========================================================
def run_epoch(model, loader, iterator, loss_fn, train):
    losses, preds, labels = [], [], []
    num_batches = CFG.eval_batches if not train else len(loader) // CFG.meta_batch_size

    for _ in tqdm(range(num_batches), desc="Train" if train else "Val"):
        x, iterator = get_meta_batch(loader, iterator)
        l, a, p, y = reptile_step(model, x, loss_fn, train=train)
        losses.append(l); preds.extend(p); labels.extend(y)
    metrics = calculate_metrics(preds, labels, CFG.n_way)
    metrics["loss"] = float(np.mean(losses))
    return metrics


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print(f"Using device: {CFG.device}")
    model = MalwareNet(CFG.input_dim, 512, CFG.n_way).to(CFG.device)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger()

    train_ds = MalwareDataset(CFG.data_json, "train", CFG.k_shot, CFG.q_query)
    val_ds = MalwareDataset(CFG.data_json, "val", CFG.k_shot, CFG.q_query)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    train_iter, val_iter = iter(train_loader), iter(val_loader)

    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        train_metrics = run_epoch(model, train_loader, train_iter, loss_fn, train=True)
        val_metrics = run_epoch(model, val_loader, val_iter, loss_fn, train=False)

        print(f"Train Acc: {train_metrics['acc']*100:.2f}% | Val Acc: {val_metrics['acc']*100:.2f}%")
        logger.add(epoch, train_metrics, val_metrics)

    print("\nTraining done. Logs saved at:", logger.path)
