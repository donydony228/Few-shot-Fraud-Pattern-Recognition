# Import modules
import glob, random
from collections import OrderedDict
import os
import pandas as pd

import numpy as np
from tqdm.auto import tqdm
import json

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display

# Check device - Support for MacBook MPS
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"DEVICE = {device}")

# Fix random seeds
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
# Hyperparameters
n_way = 3
k_shot = 1
q_query = 5
input_dim = 1280  # Adjust according to feature dimension
train_inner_train_step = 5
val_inner_train_step = 5
inner_lr = 0.001
meta_lr = 0.001
meta_batch_size = 16
max_epoch = 30
eval_batches = 20

# Utility functions for labels and accuracy
def create_malware_label(k_shot, q_query):
    """
    Create labels for query set only (for accuracy calculation).
    3 classes: 2 abnormal + 1 normal
    """
    n_way = 3  # 2 abnormal + 1 normal
    labels = []
    for class_idx in range(n_way):
        class_labels = [class_idx] * (k_shot + q_query)
        labels.extend(class_labels)
    
    return torch.tensor(labels, dtype=torch.long)

def create_label(n_way, k_shot):
    """
    Create labels for support set and query set.
    """
    return torch.arange(n_way).repeat_interleave(k_shot).long()

def calculate_accuracy(logits, labels):
    """utility function for accuracy calculation"""
    acc = np.asarray(
        [(torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy())]
    ).mean()
    return acc

class MalwareClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        """
        A simple feedforward neural network for malware classification.
        input_dim: 1280
        output_dim: 3 (2 abnormal + 1 normal)
        """
        super(MalwareClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def functional_forward(self, x, params):
        for i, (name, module) in enumerate(self.network.named_children()):
            if isinstance(module, nn.Linear):
                weight_key = f'network.{i}.weight'
                bias_key = f'network.{i}.bias'
                
                x = F.linear(x, params.get(weight_key, module.weight), 
                           params.get(bias_key, module.bias))
            elif isinstance(module, nn.ReLU):
                x = F.relu(x)
            elif isinstance(module, nn.Dropout):
                x = F.dropout(x, training=self.training)
        return x
# Setup Dataset
class MalwareDetection(Dataset):
    def __init__(self, data_structure_file, split='train', k_shot=1, q_query=5):
        """
        Load dataset structure from JSON file.
        """
        with open(data_structure_file, 'r') as f:
            self.data_structure = json.load(f)
        
        self.split = split
        self.classes = list(self.data_structure[split].keys())
        self.k_shot = k_shot
        self.q_query = q_query
        self.normal_class = 'benign'
        
        self._validate_data()
    
    def _validate_data(self):
        min_samples = self.k_shot + self.q_query
        for cls, files in self.data_structure[self.split].items():
            if len(files) < min_samples:
                print(f"Warning: only {len(files)} samples in class '{cls}' for split '{self.split}'. Required: {min_samples}. Will sample with replacement.")

    def __getitem__(self, idx):
        np.random.seed(42 + idx)  # Ensure reproducibility
        
        # Get available fraud classes
        fraud_classes = [cls for cls in self.classes if cls != self.normal_class]
        
        if len(fraud_classes) >= 2:
            selected_frauds = np.random.choice(fraud_classes, 2, replace=False)
            task_classes = list(selected_frauds) + [self.normal_class]
        elif len(fraud_classes) == 1:
            if self.split == 'test':
                task_classes = fraud_classes + [self.normal_class]
            else:
                task_classes = fraud_classes + fraud_classes + [self.normal_class]
        else:
            raise ValueError(f"No fraud classes available in {self.split} split")
        
        task_data = []
        for cls in task_classes:
            class_files = self.data_structure[self.split][cls]
            
            if len(class_files) >= self.k_shot + self.q_query:
                selected_files = np.random.choice(class_files, 
                                                self.k_shot + self.q_query, 
                                                replace=False)
            else:
                selected_files = np.random.choice(class_files, 
                                                self.k_shot + self.q_query, 
                                                replace=True)
                        
            class_features = []
            for file_path in selected_files:
                corrected_path = self._fix_file_path(file_path)
                
                try:
                    features = np.load(corrected_path)
                    if features.ndim > 1:
                        features = features.flatten()
                    class_features.append(features)
                except Exception as e:
                    if idx == 0:
                        print(f"Error loading {corrected_path}: {e}")
                    class_features.append(np.zeros(1280))
            
            task_data.append(torch.tensor(np.array(class_features), dtype=torch.float32))
        
        return torch.stack(task_data)
    
    def _fix_file_path(self, original_path):
        if os.path.exists(original_path):
            return original_path
        
        possible_prefixes = ['../', '../../', './']
        for prefix in possible_prefixes:
            new_path = os.path.join(prefix, original_path)
            if os.path.exists(new_path):
                return os.path.abspath(new_path)
        
        return original_path
    
    def __len__(self):
        fraud_classes = [cls for cls in self.classes if cls != self.normal_class]
        if len(fraud_classes) >= 2:
            from math import comb
            return comb(len(fraud_classes), 2) * 100
        else:
            return 100
def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    """
    Get meta batch function
    """
    data = []
    for _ in range(meta_batch_size):
        try:
            task_data = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            task_data = next(iterator)
        
        # task_data shape: [1, 3, k_shot+q_query, feature_dim]
        # Transform to [3*(k_shot+q_query), feature_dim]
        task_data = task_data.squeeze(0)  # [3, k_shot+q_query, feature_dim]
        task_data = task_data.view(-1, task_data.size(-1))  # [3*(k_shot+q_query), feature_dim]
        data.append(task_data)
    
    return torch.stack(data).to(device), iterator

# Main MAML Algorithm
def Solver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step,
    inner_lr,
    train,
    return_labels=False,
):
    """
    Main MAML algorithm
    """
    criterion = loss_fn
    task_loss = []
    task_acc = []
    labels = []
    
    for meta_batch in x:
        # Split support and query sets
        support_set = meta_batch[: n_way * k_shot]
        query_set = meta_batch[n_way * k_shot :]

        # Copy the params for inner loop
        fast_weights = OrderedDict(model.named_parameters())

        ### ---------- INNER TRAIN LOOP ---------- ###
        for inner_step in range(inner_train_step):
            # Simply training
            train_label = create_label(n_way, k_shot).to(device)
            logits = model.functional_forward(support_set, fast_weights)
            loss = criterion(logits, train_label)
            # Inner gradients update!
            # Calculate gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            # Update fast_weights
            # θ' = θ - α * ∇loss
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )

        ### ---------- INNER VALID LOOP ---------- ###
        if not return_labels:
            """ training / validation """
            val_label = create_label(n_way, q_query).to(device)

            # Collect gradients for outer loop
            logits = model.functional_forward(query_set, fast_weights)
            loss = criterion(logits, val_label)
            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, val_label))
        else:
            """ testing """
            logits = model.functional_forward(query_set, fast_weights)
            labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    # Update outer loop
    model.train()
    optimizer.zero_grad()

    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        """ Outer Loop Update """
        # φ backpropagation
        meta_batch_loss.backward()
        # Update parameters
        optimizer.step()

    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc

# Prepare datasets and dataloaders
train_dataset = MalwareDetection('../malware_data_structure.json', 'train', k_shot, q_query)
val_dataset = MalwareDetection('../malware_data_structure.json', 'val', k_shot, q_query)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Create model, optimizer, and loss function
meta_model = MalwareClassifier(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
loss_fn = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in meta_model.parameters())}")

# Training loop
train_iter = iter(train_loader)
val_iter = iter(val_loader)

print("Starting training...")
for epoch in range(max_epoch):
    print(f"Epoch {epoch+1}/{max_epoch}")
    
    # Training
    train_meta_loss = []
    train_acc = []
    
    for train_step in tqdm(range(len(train_loader) // meta_batch_size), desc="Training"):
        x, train_iter = get_meta_batch(
            meta_batch_size, k_shot, q_query, train_loader, train_iter
        )
        
        meta_loss, acc = Solver(
            meta_model,
            optimizer,
            x,
            n_way,
            k_shot,
            q_query,
            loss_fn,
            inner_train_step=train_inner_train_step,
            inner_lr=inner_lr,
            train=True,
        )
        
        train_meta_loss.append(meta_loss.item())
        train_acc.append(acc)
    
    print(f"Loss: {np.mean(train_meta_loss):.3f}\tAccuracy: {np.mean(train_acc)*100:.3f}%")
    
    # Validation
    val_acc = []
    for eval_step in tqdm(range(min(eval_batches, len(val_loader) // meta_batch_size)), desc="Validation"):
        x, val_iter = get_meta_batch(
            meta_batch_size, k_shot, q_query, val_loader, val_iter
        )
        
        _, acc = Solver(
            meta_model,
            optimizer,
            x,
            n_way,
            k_shot,
            q_query,
            loss_fn,
            inner_train_step=val_inner_train_step,
            inner_lr=inner_lr,
            train=False,
        )
        val_acc.append(acc)
    
    print(f"Validation accuracy: {np.mean(val_acc)*100:.3f}%")
    print("-" * 50)

print("Done！")

# Save the trained model
torch.save({
    'model_state_dict': meta_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {
        'n_way': n_way,
        'k_shot': k_shot,
        'q_query': q_query,
        'input_dim': input_dim,
        'inner_lr': inner_lr,
        'meta_lr': meta_lr
    }
}, 'malware_maml_model.pth')

print("Model saved as malware_maml_model.pth")

def test_model(model, test_data_path_or_dataset, inner_train_step=500):
    """
    Test function that returns predicted and true labels for accuracy calculation
    3-way tasks: 2 abnormal + 1 normal
    """
    test_dataset = MalwareDetection(test_data_path_or_dataset, 'test', k_shot, q_query)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_iter = iter(test_loader)
    
    test_batches = min(20, len(test_loader))
    all_predicted_labels = []
    all_true_labels = []
    task_accuracies = []

    print("Starting testing and accuracy calculation...")

    # Fix random seed for consistent label generation
    np.random.seed(42)
    
    for batch_idx in tqdm(range(test_batches), desc="Testing with Accuracy"):
        x, test_iter = get_meta_batch(1, k_shot, q_query, test_loader, test_iter)

        # Check the actual task dimensions
        batch_size, total_samples, feature_dim = x.shape
        actual_n_way = total_samples // (k_shot + q_query)

        # 3-way task query set labels
        task_true_labels = []
        for class_idx in range(3):
            task_true_labels.extend([class_idx] * q_query)

        # Get model predictions
        predicted_labels = Solver(
            model,
            optimizer,
            x,
            3,  
            k_shot,
            q_query,
            loss_fn,
            inner_train_step=inner_train_step,
            inner_lr=inner_lr,
            train=False,
            return_labels=True,
        )

        # Calculate current task accuracy
        task_true = np.array(task_true_labels)
        task_pred = np.array(predicted_labels)
        task_acc = (task_true == task_pred).mean()
        task_accuracies.append(task_acc)

        # Collect all labels
        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(task_true_labels)

        if batch_idx % 5 == 0:  # Print every 5 batches
            print(f"Batch {batch_idx+1}/{test_batches} - Task Accuracy: {task_acc:.4f}")
    
    return all_predicted_labels, all_true_labels, task_accuracies

# Execute improved testing with accuracy calculation
test_predicted_labels, test_true_labels, test_task_accuracies = test_model(meta_model, '../malware_data_structure.json')
average_test_accuracy = np.mean(test_task_accuracies)
print(f"Average Test Task Accuracy: {average_test_accuracy*100:.3f}%")

# Save test results
results_df = pd.DataFrame({
    'id': range(len(test_predicted_labels)),
    'predicted_class': test_predicted_labels,
    'true_class': test_true_labels
})

results_df.to_csv('malware_maml_predictions.csv', index=False)
print("Test results saved as malware_maml_predictions.csv")