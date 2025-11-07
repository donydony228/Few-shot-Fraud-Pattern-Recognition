import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict, Optional

import torch
import random

random_seed = 15 # Group number

class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted .npy features.
    """
    def __init__(self, file_paths: List[str], label_to_idx: Dict[str, int], class_names: List[str]):
        self.file_paths = file_paths
        self.label_to_idx = label_to_idx
        self.class_names = class_names
        self.labels = [val for key, val in label_to_idx.items() if key in class_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = os.path.basename(os.path.dirname(path))
        feature = np.load(path)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(self.label_to_idx[label])


def get_label_splits(split_csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Reads label_split.csv and returns seen/unseen labels.
    """
    df = pd.read_csv(split_csv_path)
    seen_labels = df[df['is_train'] == 1]['label'].tolist()
    unseen_labels = df[df['is_train'] == 0]['label'].tolist()
    return seen_labels, unseen_labels


def collect_file_paths(base_dir: str, labels: List[str]) -> List[str]:
    """
    Collects all .npy feature file paths for given labels.
    """
    paths = []
    for label in labels:
        label_dir = os.path.join(base_dir, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f'Directory not found for label: {label}')
        for fname in os.listdir(label_dir):
            if fname.endswith('.npy'):
                paths.append(os.path.join(label_dir, fname))
    return paths


def load_data(
    features_dir: str,
    split_csv_path: str,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    generalized: bool = False,
    num_workers: int = 4,
    one_shot = False
) -> Tuple[Dict[str, FeatureDataset], Dict[str, DataLoader]]:
    """
    Creates PyTorch dataloaders for Zero-Shot Learning setup.

    Args:
        features_dir: Path to directory containing features/{label}/{id}.npy
        split_csv_path: Path to label_split.csv
        batch_size: Batch size for DataLoaders
        val_ratio: Validation split ratio (from seen data)
        test_ratio: Test split ratio (from seen data)
        generalized: If True, test set includes both seen + unseen labels
        num_workers: Number of DataLoader workers

    Returns:
        A dict of DataLoaders: {'train', 'val', 'test_seen', 'test_unseen'}
    """
    seen_labels, unseen_labels = get_label_splits(split_csv_path)

    seen_files = collect_file_paths(features_dir, seen_labels)
    unseen_files = collect_file_paths(features_dir, unseen_labels)

    # Encode labels to integers
    all_labels = sorted(list(set(seen_labels + unseen_labels)))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    np.random.seed(random_seed)

    # Extract class labels corresponding to each file path
    seen_file_labels = [os.path.basename(os.path.dirname(p)) for p in seen_files]

    # Stratified split so every seen class appears in both splits
    train_files, test_seen_files, train_labels, test_seen_labels = train_test_split(
        seen_files,
        seen_file_labels,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=seen_file_labels
    )

    # One-shot unseen samples (1 sample per unseen class)
    one_shot_files = []
    if one_shot:
        for label in unseen_labels:
            label_dir = os.path.join(features_dir, label)
            npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
            if npy_files:
                one_shot_files.append(os.path.join(label_dir, np.random.choice(npy_files)))

        # Add one-shot unseen samples to both train and test_seen
        train_files += one_shot_files
        test_seen_files += one_shot_files

    # Dataset for seen classes
    train_set = FeatureDataset(train_files, label_to_idx, all_labels)
    test_seen_set = FeatureDataset(test_seen_files, label_to_idx, all_labels)

    # Dataset for unseen classes
    if generalized:
        combined_files = seen_files + unseen_files
        test_unseen_set = FeatureDataset(combined_files, label_to_idx, all_labels)
    else:
        test_unseen_set = FeatureDataset(unseen_files, label_to_idx, unseen_labels)

    datasets = {
        'train': train_set,
        'test_seen': test_seen_set,
        'test_unseen': test_unseen_set
    }

    # Build DataLoaders
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'test_seen': DataLoader(test_seen_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test_unseen': DataLoader(test_unseen_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    return dataloaders


# ============================================================================
# META LEARNING DATALOADERS
# ============================================================================

class EpisodeDataset(Dataset):
    """
    Dataset for meta learning that generates N-way K-shot Q-query episodes.
    
    Each episode contains:
    - N classes (n_way)
    - K support samples per class (k_shot)
    - Q query samples per class (q_query)
    - Returns shape: [n_way, k_shot + q_query, feature_dim]
    """
    
    def __init__(
        self,
        label_to_files: Dict[str, List[str]],
        n_way: int,
        k_shot: int,
        q_query: int,
        episodes_per_epoch: int = 1000,
        normalize: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            label_to_files: Dictionary mapping label names to lists of file paths
            n_way: Number of classes in each episode
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
            episodes_per_epoch: Number of episodes generated per epoch (virtual length)
            normalize: Whether to apply Z-score normalization to features
            seed: Random seed
        """
        self.label_to_files = label_to_files
        # Support 0-shot: if k_shot=0, only need q_query samples
        min_samples_per_class = max(1, k_shot + q_query) if k_shot == 0 else (k_shot + q_query)
        self.available_labels = [label for label, files in label_to_files.items() 
                                 if len(files) >= min_samples_per_class]
        
        if len(self.available_labels) < n_way:
            raise ValueError(
                f"Need at least {n_way} labels with sufficient samples, but only {len(self.available_labels)} available. "
                f"Each label needs at least {min_samples_per_class} samples."
            )
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.normalize = normalize
        self.seed = seed
        
        # Check feature dimension (load one sample)
        if len(self.available_labels) > 0 and len(self.label_to_files[self.available_labels[0]]) > 0:
            sample_file = self.label_to_files[self.available_labels[0]][0]
            sample_feat = np.load(sample_file)
            if sample_feat.ndim > 1:
                sample_feat = sample_feat.flatten()
            self.feature_dim = len(sample_feat)
        else:
            self.feature_dim = None
    
    def __len__(self):
        """Return virtual length (actual number of episodes controlled by training loop)"""
        return self.episodes_per_epoch
    
    def __getitem__(self, idx):
        """
        Generate one episode (N-way K-shot Q-query task)
        
        Returns:
            task: Tensor of shape [n_way, k_shot + q_query, feature_dim]
        """
        # Set random seed for reproducibility (based on idx)
        if self.seed is not None:
            random.seed(self.seed + idx)
            np.random.seed(self.seed + idx)
        
        # Randomly select N classes
        selected_labels = random.sample(self.available_labels, self.n_way)
        
        task = []
        for label in selected_labels:
            files = self.label_to_files[label]
            need = self.k_shot + self.q_query
            
            # Randomly select files (allow replacement if not enough files)
            if len(files) >= need:
                chosen_files = random.sample(files, need)
            else:
                # If not enough files, allow replacement sampling
                chosen_files = random.choices(files, k=need)
            
            cls_features = []
            for filepath in chosen_files:
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
                    
                    cls_features.append(feat)
                except Exception:
                    # Use zero vector if loading fails
                    feat = np.zeros(self.feature_dim, dtype=np.float32)
                    cls_features.append(feat)
            
            # Stack all samples for this class
            # If k_shot=0, only contains query samples
            if len(cls_features) > 0:
                cls_features = np.stack(cls_features)
                task.append(torch.tensor(cls_features, dtype=torch.float32))
            else:
                # If k_shot=0 and q_query=0 (should not happen), create empty tensor
                task.append(torch.zeros((0, self.feature_dim), dtype=torch.float32))
        
        # Stack all classes: [n_way, k_shot + q_query, feature_dim]
        # If k_shot=0, then [n_way, q_query, feature_dim]
        return torch.stack(task)
    
    def get_episode_with_labels(self, idx):
        """
        Generate episode and return label information (for debugging or special purposes)
        
        Returns:
            task: Tensor of shape [n_way, k_shot + q_query, feature_dim]
            selected_labels: List[str] Selected label names
            task_labels: Tensor of shape [n_way * (k_shot + q_query)] In-task label indices
        """
        if self.seed is not None:
            random.seed(self.seed + idx)
        
        selected_labels = random.sample(self.available_labels, self.n_way)
        
        task = []
        for label_idx, label in enumerate(selected_labels):
            files = self.label_to_files[label]
            need = self.k_shot + self.q_query
            
            if len(files) >= need:
                chosen_files = random.sample(files, need)
            else:
                chosen_files = random.choices(files, k=need)
            
            cls_features = []
            for filepath in chosen_files:
                try:
                    feat = np.load(filepath)
                    if feat.ndim > 1:
                        feat = feat.flatten()
                    feat = feat.astype(np.float32)
                    
                    if self.normalize:
                        mu, std = feat.mean(), feat.std()
                        feat = (feat - mu) / (std + 1e-6)
                    
                    cls_features.append(feat)
                except Exception:
                    feat = np.zeros(self.feature_dim, dtype=np.float32)
                    cls_features.append(feat)
            
            cls_features = np.stack(cls_features)
            task.append(torch.tensor(cls_features, dtype=torch.float32))
        
        task_tensor = torch.stack(task)
        
        # Create in-task labels (0 to n_way-1)
        # support: [0,0,...,0, 1,1,...,1, ...] (each repeated k_shot times)
        # query: [0,0,...,0, 1,1,...,1, ...] (each repeated q_query times)
        support_labels = torch.arange(self.n_way).repeat_interleave(self.k_shot)
        query_labels = torch.arange(self.n_way).repeat_interleave(self.q_query)
        task_labels = torch.cat([support_labels, query_labels])
        
        return task_tensor, selected_labels, task_labels


def build_label_to_files(features_dir: str, labels: List[str]) -> Dict[str, List[str]]:
    """
    Build mapping from labels to file path lists.
    
    Args:
        features_dir: Path to features directory (contains label subdirectories)
        labels: List of labels to include
    
    Returns:
        Dict[str, List[str]]: Mapping from label names to file path lists
    """
    label_to_files = {}
    for label in labels:
        label_dir = os.path.join(features_dir, label)
        if not os.path.exists(label_dir):
            print(f"Warning: Label directory not found: {label_dir}, skipping this label")
            continue
        
        files = []
        for fname in os.listdir(label_dir):
            if fname.endswith('.npy'):
                files.append(os.path.join(label_dir, fname))
        
        if len(files) > 0:
            label_to_files[label] = files
        else:
            print(f"Warning: Label {label} has no .npy files")
    
    return label_to_files


def create_meta_learning_dataloaders(
    features_dir: str,
    split_csv_path: str,
    n_way: int,
    k_shot: int,
    q_query: int,
    train_episodes_per_epoch: int = 200,
    val_episodes_per_epoch: int = 60,
    test_episodes_per_epoch: int = 100,
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Create episode-based dataloaders for meta learning.
    
    Args:
        features_dir: Path to features directory (contains features/{label}/{id}.npy)
        split_csv_path: Path to label_split.csv
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        q_query: Number of query samples per class
        train_episodes_per_epoch: Number of episodes per training epoch
        val_episodes_per_epoch: Number of episodes per validation epoch
        test_episodes_per_epoch: Number of episodes per test epoch
        normalize: Whether to normalize features
        num_workers: Number of DataLoader workers
        pin_memory: Whether to use pin memory (for GPU acceleration)
        seed: Random seed
    
    Returns:
        Dict containing the following DataLoaders:
        - 'train': Training episodes using seen labels
        - 'val': Validation episodes using seen labels
        - 'test_seen': Test episodes using seen labels
        - 'test_unseen': Test episodes using unseen labels
        - 'test_generalized': Test episodes using seen + unseen labels (generalized)
    """
    # Read seen/unseen label splits
    seen_labels, unseen_labels = get_label_splits(split_csv_path)
    
    # Build label to files mapping
    seen_label_to_files = build_label_to_files(features_dir, seen_labels)
    unseen_label_to_files = build_label_to_files(features_dir, unseen_labels)
    all_label_to_files = {**seen_label_to_files, **unseen_label_to_files}
    
    # Create datasets
    train_dataset = EpisodeDataset(
        seen_label_to_files, n_way, k_shot, q_query,
        train_episodes_per_epoch, normalize, seed
    )
    
    val_dataset = EpisodeDataset(
        seen_label_to_files, n_way, k_shot, q_query,
        val_episodes_per_epoch, normalize, seed
    )
    
    test_seen_dataset = EpisodeDataset(
        seen_label_to_files, n_way, k_shot, q_query,
        test_episodes_per_epoch, normalize, seed
    )
    
    test_unseen_dataset = EpisodeDataset(
        unseen_label_to_files, n_way, k_shot, q_query,
        test_episodes_per_epoch, normalize, seed
    )
    
    test_generalized_dataset = EpisodeDataset(
        all_label_to_files, n_way, k_shot, q_query,
        test_episodes_per_epoch, normalize, seed
    )
    
    # Create DataLoaders (batch_size=1, since each item is already a complete episode)
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test_seen': DataLoader(
            test_seen_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test_unseen': DataLoader(
            test_unseen_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test_generalized': DataLoader(
            test_generalized_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
    }
    
    return dataloaders


def split_episode_to_support_query(
    episode: torch.Tensor,
    k_shot: int,
    q_query: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split episode into support and query sets, and create corresponding labels.
    Supports 0-shot (k_shot=0).
    
    Args:
        episode: Tensor of shape [n_way, k_shot + q_query, feature_dim] 
                 or [n_way, q_query, feature_dim] if k_shot=0
        k_shot: Number of support samples (can be 0)
        q_query: Number of query samples
    
    Returns:
        support: Tensor of shape [n_way * k_shot, feature_dim] (empty tensor if k_shot=0)
        query: Tensor of shape [n_way * q_query, feature_dim]
        support_labels: Tensor of shape [n_way * k_shot] (empty tensor if k_shot=0)
        query_labels: Tensor of shape [n_way * q_query]
    """
    n_way = episode.shape[0]
    
    # Split support and query
    if k_shot > 0:
        support = episode[:, :k_shot, :].reshape(n_way * k_shot, -1)
        query = episode[:, k_shot:, :].reshape(n_way * q_query, -1)
        support_labels = torch.arange(n_way).repeat_interleave(k_shot).long()
    else:
        # 0-shot case: all samples are query
        support = torch.empty((0, episode.shape[2]), dtype=episode.dtype)
        query = episode.reshape(n_way * q_query, -1)
        support_labels = torch.empty((0,), dtype=torch.long)
    
    query_labels = torch.arange(n_way).repeat_interleave(q_query).long()
    
    return support, query, support_labels, query_labels