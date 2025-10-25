# MI-MAML.py
# Embedding-level MI-MAML (FOMAML variant) for MalVis 1280-D features
# Jeremy-friendly: single file, no imports from your other scripts

import os, json, math, random
from datetime import datetime
from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# =========================================================
# CONFIG
# =========================================================
class CFG:
    # Task setup
    n_way = 3
    k_shot = 1
    q_query = 5
    input_dim = 1280

    # Inner / outer
    inner_lr = 0.01        # base inner-loop LR (will be cycled)
    inner_steps = 5        # K steps per task in inner loop
    meta_lr = 1e-3         # Adam LR for outer loop
    meta_batch_size = 8    # number of tasks per meta step (grad accumulate)

    # Training
    max_epoch = 200
    train_episodes_per_epoch = 200    # how many tasks per epoch (train)
    val_episodes_per_epoch = 60       # how many tasks per epoch (val)

    # IO
    data_json = "../malware_data_structure.json"
    log_dir = "logs_mimaml"

    # System
    seed = 42
    # Support for MacBook MPS
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    num_workers = 0
    pin_memory = torch.cuda.is_available()


# =========================================================
# SEED
# =========================================================
def set_seed(sd=42):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)

set_seed(CFG.seed)


# =========================================================
# UTILITIES
# =========================================================
def create_label(n_way, num_per_class):
    return torch.arange(n_way).repeat_interleave(num_per_class).long()

def calculate_metrics(preds, labels, num_classes):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return {
        "acc": (preds == labels).mean().item() if preds.size else 0.0,
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)) if preds.size else 0.0,
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)) if preds.size else 0.0,
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)) if preds.size else 0.0,
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)) if preds.size else 0.0,
        "cm": confusion_matrix(labels, preds).tolist() if preds.size else [[0]],
    }

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def cyclic_inner_lr(base_lr, step_idx, total_steps):
    # 1-cycle cosine schedule in inner loop
    return float(base_lr * 0.5 * (1.0 + math.cos(2.0 * math.pi * (step_idx / max(1, total_steps)))))


# =========================================================
# DATASET (MalVis .npy 1280-D features)
# Task: 2 malware + 1 benign (3-way by default)
# =========================================================
class MalwareDataset(Dataset):
    def __init__(self, json_path, split, k_shot, q_query, input_dim=1280):
        with open(json_path, "r") as f:
            self.data = json.load(f)[split]
        self.classes = list(self.data.keys())
        self.k_shot = k_shot
        self.q_query = q_query
        self.input_dim = input_dim
        self.normal = "benign"

    def __getitem__(self, idx):
        np.random.seed(CFG.seed + idx)
        # pick 2 malware families + 1 benign
        frauds = [c for c in self.classes if c != self.normal]
        if len(frauds) < 2:
            raise ValueError("Not enough malware families in JSON for a 3-way task.")

        selected = np.random.choice(frauds, CFG.n_way - 1, replace=False).tolist() + [self.normal]

        task = []
        for cls in selected:
            files = self.data[cls]
            need = self.k_shot + self.q_query
            chosen = np.random.choice(files, need, replace=(len(files) < need))
            cls_feats = []
            for fp in chosen:
                p = self._fix(fp)
                try:
                    arr = np.load(p)
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    # per-sample z-score (stabilize inner loop)
                    mu, std = arr.mean(), arr.std()
                    arr = (arr - mu) / (std + 1e-6)
                except Exception as e:
                    # fallback
                    arr = np.zeros(self.input_dim, dtype=np.float32)
                cls_feats.append(arr.astype(np.float32))
            task.append(torch.tensor(np.stack(cls_feats), dtype=torch.float32))
        return torch.stack(task)  # [n_way, k+q, feat]

    def _fix(self, path):
        if os.path.exists(path): return path
        for prefix in ["../", "../../", "./"]:
            cand = os.path.join(prefix, path)
            if os.path.exists(cand):
                return os.path.abspath(cand)
        return path

    def __len__(self):
        # define a "virtually infinite" pool; outer loop controls episodes per epoch
        return 100000


def make_loader(json_path, split):
    ds = MalwareDataset(json_path, split, CFG.k_shot, CFG.q_query, CFG.input_dim)
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)


# =========================================================
# MODEL: Lightweight MLP Head (best fit for 1280-D embeddings)
# + Ordered param access for functional forward
# =========================================================
class MalwareHead(nn.Module):
    def __init__(self, input_dim=1280, hidden=512, n_way=3, p_drop=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_way)
        self.p_drop = p_drop

    def forward(self, x):
        # standard forward (not used for meta-update, but useful for sanity checks)
        if x.dim() == 1: x = x.unsqueeze(0)
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.p_drop, training=self.training)
        x = torch.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.p_drop, training=self.training)
        return self.fc3(x)

    # return parameters in a stable order to be used by "functional forward"
    def ordered_params(self):
        return [
            self.norm.weight, self.norm.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias,
        ]


def functional_forward(model: MalwareHead, x, fast_params):
    """Functional forward using a flat list of tensors in the order of model.ordered_params()."""
    if x.dim() == 1: x = x.unsqueeze(0)
    i = 0
    # LayerNorm
    w_ln, b_ln = fast_params[i], fast_params[i+1]; i += 2
    x = nn.functional.layer_norm(x, (w_ln.shape[0],), w_ln, b_ln, eps=1e-5)
    # FC1
    w, b = fast_params[i], fast_params[i+1]; i += 2
    x = nn.functional.linear(x, w, b)
    x = nn.functional.relu(x)
    x = nn.functional.dropout(x, p=model.p_drop, training=True)
    # FC2
    w, b = fast_params[i], fast_params[i+1]; i += 2
    x = nn.functional.linear(x, w, b)
    x = nn.functional.relu(x)
    x = nn.functional.dropout(x, p=model.p_drop, training=True)
    # FC3 (logits)
    w, b = fast_params[i], fast_params[i+1]; i += 2
    x = nn.functional.linear(x, w, b)
    return x


# =========================================================
# MI-MAML (FOMAML) STEP
# =========================================================
def maml_inner_adapt(model, task_batch, n_way, k_shot, q_query,
                     inner_steps, inner_base_lr, device, loss_fn=None):
    """
    One FOMAML inner adaptation for a single task.
    Returns:
        query_loss (tensor), query_acc (float), grads_wrt_fast (list of tensors)
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    task = task_batch.squeeze(0).to(device)  # [n_way, k+q, feat]
    support = task[:, :k_shot, :].reshape(n_way * k_shot, -1)
    query   = task[:, k_shot:, :].reshape(n_way * q_query, -1)

    y_s = create_label(n_way, k_shot).to(device)
    y_q = create_label(n_way, q_query).to(device)

    # θ0 (base) → list
    base_params = model.ordered_params()
    # fast params need grad
    fast = [p.clone().detach().requires_grad_(True) for p in base_params]

    # Inner loop
    for t in range(inner_steps):
        lr_t = cyclic_inner_lr(inner_base_lr, t, inner_steps)
        logits_s = functional_forward(model, support, fast)
        loss_s = loss_fn(logits_s, y_s)
        grads = torch.autograd.grad(loss_s, fast, create_graph=False)
        # θ' ← θ' - lr_t * grad
        fast = [w - lr_t * g if g is not None else w for w, g in zip(fast, grads)]
        # 再次 requires_grad 以便下一步可求梯度
        fast = [w.detach().requires_grad_(True) for w in fast]

    # Query 上計算 meta loss
    logits_q = functional_forward(model, query, fast)
    loss_q = loss_fn(logits_q, y_q)
    acc_q = (logits_q.argmax(-1) == y_q).float().mean().item()

    # 對 fast 參數求導，作為近似對 base 參數的梯度
    grads_q = torch.autograd.grad(loss_q, fast, create_graph=False)

    return loss_q.detach(), acc_q, grads_q


def accumulate_grads(model, grads_accum):
    """Write accumulated grads back into model.parameters() in the same order."""
    model_params = model.ordered_params()
    with torch.no_grad():
        for p, g in zip(model_params, grads_accum):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad += g


# =========================================================
# TRAIN / EVAL
# =========================================================
def run_epoch(train, model, loader, cfg: CFG):
    device = cfg.device
    loss_fn = nn.CrossEntropyLoss()
    model.train() if train else model.eval()

    meta_opt = getattr(run_epoch, "_opt", None)
    if meta_opt is None:
        meta_opt = torch.optim.Adam(model.parameters(), lr=cfg.meta_lr)
        run_epoch._opt = meta_opt

    episodes = cfg.train_episodes_per_epoch if train else cfg.val_episodes_per_epoch
    iterator = iter(loader)

    total_loss, total_acc = 0.0, 0.0
    meta_batch_grads = None
    preds_all, labels_all = [], []

    for ep in tqdm(range(episodes), desc="Train" if train else "Val"):
        try:
            task = next(iterator)  # [1, n_way, k+q, feat]
        except StopIteration:
            iterator = iter(loader)
            task = next(iterator)

        # inner adaptation for one task
        q_loss, q_acc, grads_q = maml_inner_adapt(
            model, task, cfg.n_way, cfg.k_shot, cfg.q_query,
            cfg.inner_steps, cfg.inner_lr, device, loss_fn
        )

        total_loss += q_loss.item()
        total_acc += q_acc

        # 用 query logits 產生 preds/labels（為了統計）
        with torch.no_grad():
            task_t = task.squeeze(0).to(device)
            query = task_t[:, cfg.k_shot:, :].reshape(cfg.n_way * cfg.q_query, -1)
            y_q = create_label(cfg.n_way, cfg.q_query).to(device)
            # 需要用 fast params才是準確的 preds，但這裡用 acc 即可；詳細 preds 可另外返回
            # 為了簡潔，這裡略過逐步保存 preds，僅用 acc/f1 等 summary 指標

        # 累加 grads（做 meta-batch）
        if meta_batch_grads is None:
            meta_batch_grads = [g.clone() for g in grads_q]
        else:
            for i in range(len(meta_batch_grads)):
                meta_batch_grads[i] += grads_q[i]

        # 每 meta_batch_size 個 task 做一次 outer step
        if (ep + 1) % cfg.meta_batch_size == 0:
            # 將平均梯度寫回模型參數 .grad
            avg_grads = [g / cfg.meta_batch_size for g in meta_batch_grads]
            # 梯度歸零
            meta_opt.zero_grad(set_to_none=True)
            accumulate_grads(model, avg_grads)
            # meta step
            meta_opt.step()
            meta_batch_grads = None

    # 還有殘餘的梯度要推一下
    if meta_batch_grads is not None and train:
        avg_grads = [g / ((episodes % cfg.meta_batch_size) or cfg.meta_batch_size) for g in meta_batch_grads]
        meta_opt.zero_grad(set_to_none=True)
        accumulate_grads(model, avg_grads)
        meta_opt.step()

    # 統計
    avg_loss = total_loss / max(1, episodes)
    avg_acc = total_acc / max(1, episodes)
    return {"loss": float(avg_loss), "acc": float(avg_acc)}


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(CFG.log_dir)
    print(f"[MI-MAML] device={CFG.device}  n_way={CFG.n_way}  k_shot={CFG.k_shot}  q_query={CFG.q_query}")

    # Dataloaders
    train_loader = make_loader(CFG.data_json, "train")
    val_loader   = make_loader(CFG.data_json, "val")

    # Model
    model = MalwareHead(CFG.input_dim, hidden=512, n_way=CFG.n_way).to(CFG.device)

    # Train loop
    best_val = -1.0
    save_path = os.path.join(CFG.log_dir, f"mi_maml_{timestamp()}.pth")

    for epoch in range(1, CFG.max_epoch + 1):
        print(f"\n===== Epoch {epoch}/{CFG.max_epoch} =====")
        train_stats = run_epoch(train=True,  model=model, loader=train_loader, cfg=CFG)
        val_stats   = run_epoch(train=False, model=model, loader=val_loader,   cfg=CFG)

        print(f"Train: acc={train_stats['acc']*100:.2f}%  loss={train_stats['loss']:.4f}")
        print(f"Val  : acc={val_stats['acc']*100:.2f}%  loss={val_stats['loss']:.4f}")

        # save best
        if val_stats["acc"] > best_val:
            best_val = val_stats["acc"]
            cfg_dict = {
                k: v for k, v in CFG.__dict__.items()
                if not k.startswith("__") and isinstance(v, (int, float, str, bool, list))
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg_dict,
                "epoch": epoch,
                "val_acc": best_val,
            }, save_path)
            print(f"✅ Saved best to: {save_path} (val_acc={best_val*100:.2f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
