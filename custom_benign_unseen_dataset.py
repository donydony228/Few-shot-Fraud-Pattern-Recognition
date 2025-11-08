"""
Custom Episode Dataset for Benign + Unseen Malware Classes
專門用於生成 1個benign（seen）+ 2個惡意軟體（unseen四選二）的episodes
"""

import os
import numpy as np
import torch
import random
from typing import Dict, List, Optional
from torch.utils.data import Dataset

class BenignUnseenEpisodeDataset(Dataset):
    """
    Custom dataset for generating episodes with:
    - 1 benign class (from seen labels)
    - 2 malware classes (randomly selected from 4 unseen labels)
    
    This simulates realistic malware detection scenarios where benign samples
    are available but malware types are new/unseen.
    """
    
    def __init__(
        self,
        seen_label_to_files: Dict[str, List[str]],
        unseen_label_to_files: Dict[str, List[str]],
        benign_label: str = "benign",  # 根據 label_split.csv 確認是 "benign"
        k_shot: int = 1,
        q_query: int = 5,
        episodes_per_epoch: int = 100,
        normalize: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            seen_label_to_files: Dict mapping seen labels to file paths
            unseen_label_to_files: Dict mapping unseen labels to file paths  
            benign_label: Name of benign class (must be in seen_label_to_files)
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
            episodes_per_epoch: Number of episodes per epoch
            normalize: Whether to apply Z-score normalization
            seed: Random seed for reproducibility
        """
        self.seen_label_to_files = seen_label_to_files
        self.unseen_label_to_files = unseen_label_to_files
        self.benign_label = benign_label
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.normalize = normalize
        self.seed = seed
        
        # 驗證 benign 類別存在於 seen labels 中
        if benign_label not in seen_label_to_files:
            raise ValueError(f"Benign label '{benign_label}' not found in seen labels. "
                           f"Available seen labels: {list(seen_label_to_files.keys())}")
        
        # 獲取可用的 unseen 惡意軟體類別（排除 benign）
        self.available_unseen_labels = [
            label for label in unseen_label_to_files.keys() 
            if label != benign_label and len(unseen_label_to_files[label]) >= (k_shot + q_query)
        ]
        
        if len(self.available_unseen_labels) < 2:
            raise ValueError(f"Need at least 2 unseen malware classes with sufficient samples, "
                           f"but only {len(self.available_unseen_labels)} available: {self.available_unseen_labels}")
        
        # 檢查 benign 是否有足夠樣本
        min_samples_needed = k_shot + q_query
        if len(seen_label_to_files[benign_label]) < min_samples_needed:
            raise ValueError(f"Benign class needs at least {min_samples_needed} samples, "
                           f"but only {len(seen_label_to_files[benign_label])} available")
        
        # 檢測特徵維度
        sample_file = seen_label_to_files[benign_label][0]
        sample_feat = np.load(sample_file)
        if sample_feat.ndim > 1:
            sample_feat = sample_feat.flatten()
        self.feature_dim = len(sample_feat)
        
        print(f"BenignUnseenEpisodeDataset initialized:")
        print(f"  - Benign class: {benign_label} ({len(seen_label_to_files[benign_label])} samples)")
        print(f"  - Available unseen malware: {len(self.available_unseen_labels)} classes")
        print(f"  - Unseen classes: {self.available_unseen_labels}")
        print(f"  - Feature dimension: {self.feature_dim}")
    
    def __len__(self):
        return self.episodes_per_epoch
    
    def __getitem__(self, idx):
        """
        Generate one episode with 3-way classification:
        - Class 0: benign (from seen)
        - Class 1: malware type 1 (from unseen)
        - Class 2: malware type 2 (from unseen)
        
        Returns:
            task: Tensor of shape [3, k_shot + q_query, feature_dim]
        """
        # Set random seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed + idx)
            np.random.seed(self.seed + idx)
        
        # 1. 選擇 benign 樣本
        benign_files = self.seen_label_to_files[self.benign_label]
        need_samples = self.k_shot + self.q_query
        
        if len(benign_files) >= need_samples:
            chosen_benign = random.sample(benign_files, need_samples)
        else:
            chosen_benign = random.choices(benign_files, k=need_samples)
        
        # 2. 從4個 unseen 類別中隨機選2個惡意軟體類別
        selected_malware_labels = random.sample(self.available_unseen_labels, 2)
        
        # 3. 生成 episode 數據
        task = []
        
        # Class 0: Benign
        benign_features = self._load_class_features(chosen_benign)
        task.append(benign_features)
        
        # Class 1 & 2: Two selected malware types
        for malware_label in selected_malware_labels:
            malware_files = self.unseen_label_to_files[malware_label]
            if len(malware_files) >= need_samples:
                chosen_malware = random.sample(malware_files, need_samples)
            else:
                chosen_malware = random.choices(malware_files, k=need_samples)
            
            malware_features = self._load_class_features(chosen_malware)
            task.append(malware_features)
        
        # Stack all classes: [3, k_shot + q_query, feature_dim]
        return torch.stack(task)
    
    def _load_class_features(self, file_paths: List[str]) -> torch.Tensor:
        """
        Load features for one class from file paths
        
        Args:
            file_paths: List of .npy file paths
            
        Returns:
            features: Tensor of shape [k_shot + q_query, feature_dim]
        """
        features = []
        
        for filepath in file_paths:
            try:
                feat = np.load(filepath)
                # Ensure feature is 1D
                if feat.ndim > 1:
                    feat = feat.flatten()
                feat = feat.astype(np.float32)
                
                # Z-score normalization (per sample)
                if self.normalize:
                    mu, std = feat.mean(), feat.std()
                    feat = (feat - mu) / (std + 1e-6)
                
                features.append(feat)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                # Use zero vector if loading fails
                feat = np.zeros(self.feature_dim, dtype=np.float32)
                features.append(feat)
        
        return torch.tensor(np.stack(features), dtype=torch.float32)
    
    def get_episode_info(self, idx):
        """
        Get information about the episode (for debugging)
        
        Returns:
            Dict with episode information
        """
        if self.seed is not None:
            random.seed(self.seed + idx)
        
        selected_malware = random.sample(self.available_unseen_labels, 2)
        
        return {
            'episode_idx': idx,
            'benign_label': self.benign_label,
            'malware_labels': selected_malware,
            'class_mapping': {
                0: self.benign_label,
                1: selected_malware[0], 
                2: selected_malware[1]
            }
        }


# 為了整合方便，創建一個 wrapper 函數
def add_custom_test_loader_to_existing(
    existing_dataloaders: dict,
    features_dir: str,
    split_csv_path: str,
    benign_label: str = "benign",
    k_shot: int = 1,
    q_query: int = 5,
    episodes_per_epoch: int = 100,
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None
) -> dict:
    """
    Add custom benign+unseen test loader to existing dataloaders dict
    
    Args:
        existing_dataloaders: 現有的 dataloaders dictionary
        其他參數同上
    
    Returns:
        Updated dataloaders dict with 'test_benign_unseen' added
    """
    from torch.utils.data import DataLoader
    import pandas as pd
    
    # Helper functions (copy from data_loader.py if needed)
    def get_label_splits_local(split_csv_path):
        df = pd.read_csv(split_csv_path)
        seen_labels = df[df['is_train'] == 1]['label'].tolist()
        unseen_labels = df[df['is_train'] == 0]['label'].tolist()
        return seen_labels, unseen_labels
    
    def build_label_to_files_local(features_dir, labels):
        label_to_files = {}
        for label in labels:
            label_dir = os.path.join(features_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Label directory not found: {label_dir}")
                continue
            
            files = []
            for fname in os.listdir(label_dir):
                if fname.endswith('.npy'):
                    files.append(os.path.join(label_dir, fname))
            
            if len(files) > 0:
                label_to_files[label] = files
        
        return label_to_files
    
    # Read label splits
    seen_labels, unseen_labels = get_label_splits_local(split_csv_path)
    
    # Build label to files mapping  
    seen_label_to_files = build_label_to_files_local(features_dir, seen_labels)
    unseen_label_to_files = build_label_to_files_local(features_dir, unseen_labels)
    
    # Create custom dataset
    custom_dataset = BenignUnseenEpisodeDataset(
        seen_label_to_files=seen_label_to_files,
        unseen_label_to_files=unseen_label_to_files,
        benign_label=benign_label,
        k_shot=k_shot,
        q_query=q_query,
        episodes_per_epoch=episodes_per_epoch,
        normalize=normalize,
        seed=seed
    )
    
    # Create DataLoader
    custom_loader = DataLoader(
        custom_dataset,
        batch_size=1,
        shuffle=False,  # Set to False for consistent testing
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Add to existing dataloaders
    existing_dataloaders['test_benign_unseen'] = custom_loader
    
    return existing_dataloaders
