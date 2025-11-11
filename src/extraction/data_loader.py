import os
import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

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
        feature = torch.tensor(feature, dtype=torch.float32)
        feature = torch.nn.functional.normalize(feature, p=2, dim=0)
        return feature, torch.tensor(self.label_to_idx[label])


def get_label_splits(split_csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Reads label_split.csv and returns seen/unseen labels.
    """
    df = pd.read_csv(split_csv_path)
    seen_labels = df[df['is_train'] == 1]['label'].tolist()
    unseen_labels = df[df['is_train'] == 0]['label'].tolist()
    return seen_labels, unseen_labels


def collect_file_paths(base_dir: str, labels: List[str]) -> Dict[str, List[str]]:
    """
    Collects all .npy feature file paths for each label.
    Returns a dict: {label: [file_paths]}
    """
    paths = {}
    for label in labels:
        label_dir = os.path.join(base_dir, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Directory not found for label: {label}")
        npy_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')]
        if len(npy_files) == 0:
            raise ValueError(f"No .npy files found for label: {label}")
        paths[label] = npy_files
    return paths


def get_n_samples(unseen_class_files, num):
    n_shot_files = []
    for label, files in unseen_class_files.items():
            n_select = min(num, len(files))
            n_shot_files.extend(random.sample(files, n_select))
    return n_shot_files


def load_data(
    features_dir: str,
    split_csv_path: str,
    batch_size: int = 64,
    test_ratio: float = 0.2,
    generalized: bool = False,
    num_workers: int = 4,
    n_shot = 0
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

    # Collect all class files
    seen_class_files = collect_file_paths(features_dir, seen_labels)
    unseen_class_files = collect_file_paths(features_dir, unseen_labels)

    # Determine minimum count across ALL classes (seen + unseen)
    min_samples = min(
        min(len(v) for v in seen_class_files.values()),
        min(len(v) for v in unseen_class_files.values())
    )

    # Sample the same number of examples per class
    balanced_seen_files, balanced_unseen_files = [], []
    for label, files in seen_class_files.items():
        balanced_seen_files.extend(random.sample(files, min_samples))
    for label, files in unseen_class_files.items():
        balanced_unseen_files.extend(random.sample(files, min_samples))

    # Extract class labels for stratification
    seen_file_labels = [os.path.basename(os.path.dirname(p)) for p in balanced_seen_files]

    # Stratified split for seen classes
    train_files, test_seen_files, _, _ = train_test_split(
        balanced_seen_files,
        seen_file_labels,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=seen_file_labels
    )

    # Few-shot unseen addition (n_shot per unseen class)
    if n_shot:
        # Called twice to get different samples selected randomly
        train_files += get_n_samples(unseen_class_files, n_shot)
        test_seen_files += get_n_samples(unseen_class_files, n_shot)


    # Encode labels to integers
    all_labels = sorted(list(set(seen_labels + unseen_labels)))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    # Label encoding
    all_labels = sorted(list(set(seen_labels + unseen_labels)))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    # Build datasets
    train_set = FeatureDataset(train_files, label_to_idx, all_labels if n_shot > 0 else seen_labels)
    test_seen_set = FeatureDataset(test_seen_files, label_to_idx, all_labels if n_shot > 0 else seen_labels)

    if generalized:
        combined_files = balanced_seen_files + balanced_unseen_files
        test_unseen_set = FeatureDataset(combined_files, label_to_idx, all_labels)
    else:
        test_unseen_set = FeatureDataset(balanced_unseen_files, label_to_idx, unseen_labels)

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

    return datasets, dataloaders
