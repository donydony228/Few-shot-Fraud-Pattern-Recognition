# episode_loader.py
# -*- coding: utf-8 -*-
"""
Episode-based loader wrapper for MI-MAML and Reptile.
Takes a standard DataLoader (from create_dataloaders) and samples N-way K-shot tasks.
"""

import random
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass(frozen=True)
class Episode:
    """Container for a single few-shot episode."""
    support_x: torch.Tensor  # shape: (n_way * k_shot, feat_dim)
    support_y: torch.Tensor
    query_x: torch.Tensor    # shape: (n_way * q_query, feat_dim)
    query_y: torch.Tensor
    class_names: Tuple[int, ...]


def _build_class_cache(loader: DataLoader, device: Optional[torch.device] = None, normalize: bool = True) -> Dict[int, List[torch.Tensor]]:
    """Group all samples by label and normalize."""
    class_cache: Dict[int, List[torch.Tensor]] = {}
    print("Building class cache...")
    for batch_idx, (x, y) in enumerate(loader):
        for xi, yi in zip(x, y):
            cid = int(yi.item())
            
            # Z-score normalization per sample (关键修复!)
            if normalize:
                xi_mean = xi.mean()
                xi_std = xi.std()
                if xi_std > 1e-6:
                    xi = (xi - xi_mean) / xi_std
                else:
                    xi = xi - xi_mean
            
            # 保持在 CPU，使用时再移动到 device
            class_cache.setdefault(cid, []).append(xi)
            
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1} batches...")
    
    print(f"Class cache built: {len(class_cache)} classes")
    for cid, samples in class_cache.items():
        print(f"  Class {cid}: {len(samples)} samples")
    return class_cache


class EpisodeIterator:
    """Yield few-shot tasks from class cache."""

    def __init__(
        self,
        loader: DataLoader,
        n_way: int,
        k_shot: int,
        q_query: int,
        episodes_per_epoch: int,
        seed: int = 42,
        device: Optional[torch.device] = None,
        normalize: bool = True,
        normal_class_id: Optional[int] = None,
        normal_class_loader: Optional[DataLoader] = None,
    ):
        self.device = device
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.rng = random.Random(seed)
        # 构建缓存时不移动数据到 device（保持 CPU），归一化在这里进行
        self.class_cache = _build_class_cache(loader, device=None, normalize=normalize)
        self.classes = list(self.class_cache.keys())
        
        self.normal_class_id = normal_class_id
        self.has_normal_in_cache = self.normal_class_id in self.classes
        
        # 如果当前数据集中没有 normal 类别，尝试从另一个 loader 借用
        if not self.has_normal_in_cache and normal_class_loader is not None:
            print(f"  Normal class {self.normal_class_id} not in current dataset, borrowing from provided loader...")
            normal_cache = _build_class_cache(normal_class_loader, device=None, normalize=normalize)
            if self.normal_class_id in normal_cache:
                # 将 normal 类别添加到当前缓存中
                self.class_cache[self.normal_class_id] = normal_cache[self.normal_class_id]
                self.classes.append(self.normal_class_id)
                self.has_normal_in_cache = True
                print(f"  ✓ Borrowed {len(self.class_cache[self.normal_class_id])} normal samples")
            else:
                print(f"  ⚠️ Normal class {self.normal_class_id} also not found in provided loader")
        
        # 如果还是没有，尝试自动检测或使用默认值
        if not self.has_normal_in_cache:
            print(f"⚠️ Warning: Normal class ID {self.normal_class_id} not in available classes")
            print(f"   Available classes: {sorted(self.classes)}")
            # 对于 unseen 测试集，如果没有 benign，我们允许只用恶意软件类别
            # 但这会改变 n_way，所以先尝试找到可能的替代
            if len(self.classes) >= self.n_way:
                # 如果类别数足够，我们可以不用 benign，只用恶意软件
                print(f"   Will use only fraud classes for {self.n_way}-way task")
                self.normal_class_id = None  # 不使用 normal
            else:
                # 尝试自动检测
                if len(self.classes) >= 3:
                    self.normal_class_id = sorted(self.classes)[2] if 2 in self.classes else self.classes[0]
                    print(f"   Auto-detected normal class ID: {self.normal_class_id}")
        
        # 分离恶意软件类别和正常类别
        if self.normal_class_id is not None and self.normal_class_id in self.classes:
            self.fraud_classes = [c for c in self.classes if c != self.normal_class_id]
        else:
            # 如果没有 normal，所有类别都视为 fraud
            self.fraud_classes = list(self.classes)
            self.normal_class_id = None
        
        print(f"EpisodeIterator initialized:")
        print(f"  Total classes: {len(self.classes)}")
        print(f"  Normal class ID: {self.normal_class_id if self.normal_class_id is not None else 'None (using fraud only)'}")
        print(f"  Fraud classes: {len(self.fraud_classes)} classes (IDs: {sorted(self.fraud_classes)})")
        
        if len(self.fraud_classes) < (n_way - 1) and self.normal_class_id is None:
            raise ValueError(
                f"Not enough classes ({len(self.fraud_classes)}) for {n_way}-way task "
                f"and no normal class available"
            )
        elif len(self.fraud_classes) < (n_way - 1):
            raise ValueError(
                f"Not enough fraud classes ({len(self.fraud_classes)}) for {n_way}-way task "
                f"(need {n_way - 1} fraud classes + 1 normal)"
            )

        for cid, lst in self.class_cache.items():
            if len(lst) < (k_shot + q_query):
                print(f"⚠️ Class {cid} has only {len(lst)} samples (need {k_shot+q_query})")

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            # 原始实现逻辑：选择 (n_way - 1) 个恶意软件类别 + 1 个 benign
            # 对于 n_way=3，选择 2 个恶意软件 + 1 个 benign
            if self.normal_class_id is not None:
                # 有 normal 类别：选择 (n_way - 1) 个恶意软件 + 1 个 normal
                if len(self.fraud_classes) < (self.n_way - 1):
                    raise ValueError(
                        f"Not enough fraud classes ({len(self.fraud_classes)}) for {self.n_way}-way task"
                    )
                selected_frauds = self.rng.sample(self.fraud_classes, self.n_way - 1)
                chosen = selected_frauds + [self.normal_class_id]
            else:
                # 没有 normal 类别：只用恶意软件类别（用于 unseen 测试）
                if len(self.fraud_classes) < self.n_way:
                    raise ValueError(
                        f"Not enough classes ({len(self.fraud_classes)}) for {self.n_way}-way task"
                    )
                chosen = self.rng.sample(self.fraud_classes, self.n_way)
            
            support_x, query_x, support_y, query_y = [], [], [], []

            for epi_label, cls in enumerate(chosen):
                available = len(self.class_cache[cls])
                need = self.k_shot + self.q_query
                
                # 如果样本不够，使用 replacement=True
                if available < need:
                    samples = self.rng.choices(self.class_cache[cls], k=need)
                else:
                    samples = self.rng.sample(self.class_cache[cls], need)
                
                s, q = samples[:self.k_shot], samples[self.k_shot:]
                support_x.extend(s)
                query_x.extend(q)
                support_y.extend([epi_label] * self.k_shot)
                query_y.extend([epi_label] * self.q_query)

            # 移动到 device（如果指定）
            support_x_t = torch.stack(support_x)
            query_x_t = torch.stack(query_x)
            support_y_t = torch.tensor(support_y, dtype=torch.long)
            query_y_t = torch.tensor(query_y, dtype=torch.long)
            
            if self.device:
                support_x_t = support_x_t.to(self.device)
                query_x_t = query_x_t.to(self.device)
                support_y_t = support_y_t.to(self.device)
                query_y_t = query_y_t.to(self.device)
            
            yield Episode(
                support_x=support_x_t,
                support_y=support_y_t,
                query_x=query_x_t,
                query_y=query_y_t,
                class_names=tuple(chosen),
            )
