import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict
import torch

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

    if one_shot:
        # Add 1 sample per unseen class to seen_files
        one_shot_files = []
        for label in unseen_labels:
            label_dir = os.path.join(features_dir, label)
            npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
            if npy_files:
                np.random.seed(random_seed)
                one_shot_files.append(os.path.join(label_dir, np.random.choice(npy_files)))
        # Merge seen_files with 1-shot unseen samples
        seen_files = seen_files + one_shot_files
        # -------------------------

    # Dataset for seen classes
    seen_dataset = FeatureDataset(seen_files, label_to_idx, all_labels)

    # Train/Val/Test split for seen classes
    n_total = len(seen_dataset)
    n_val = int(val_ratio * n_total)
    n_test = int(test_ratio * n_total)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_seen_set = random_split(
        seen_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # Dataset for unseen classes
    if generalized:
        combined_files = seen_files + unseen_files
        test_unseen_set = FeatureDataset(combined_files, label_to_idx, all_labels)
    else:
        test_unseen_set = FeatureDataset(unseen_files, label_to_idx, unseen_labels)

    datasets = {
        'seen': seen_dataset,
        'unseen': test_unseen_set
    }

    # Build DataLoaders
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test_seen': DataLoader(test_seen_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test_unseen': DataLoader(test_unseen_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return datasets, dataloaders
