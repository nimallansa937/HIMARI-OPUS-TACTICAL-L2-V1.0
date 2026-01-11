"""
HIMARI Layer 2 - Data Loading Module
Handles loading preprocessed features and labels for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import json

logger = logging.getLogger(__name__)


class HIMARIDataset(Dataset):
    """
    PyTorch Dataset for HIMARI Layer 2 training data.
    
    Loads preprocessed 60D feature vectors and classification labels.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Args:
            features: Array of shape (N, 60) - feature vectors
            labels: Array of shape (N,) - labels (0=SELL, 1=HOLD, 2=BUY)
            transform: Optional transform to apply to features
        """
        assert len(features) == len(labels), "Features and labels must have same length"
        
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


class CurriculumDataset(Dataset):
    """
    Dataset that supports curriculum learning by filtering samples by difficulty.
    
    Difficulty is estimated based on:
    - Volatility of the sample period
    - Proximity to regime changes
    - Label confidence (how clear the signal was)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        difficulties: Optional[np.ndarray] = None,
        difficulty_threshold: float = 1.0
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
        # If no difficulties provided, compute simple proxy
        if difficulties is None:
            # Use feature variance as difficulty proxy
            difficulties = np.var(features, axis=1)
            difficulties = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min() + 1e-8)
        
        self.difficulties = torch.FloatTensor(difficulties)
        self.difficulty_threshold = difficulty_threshold
        
        # Filter samples by difficulty
        self.mask = self.difficulties <= difficulty_threshold
        self.indices = torch.where(self.mask)[0]
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        return self.features[real_idx], self.labels[real_idx]
    
    def set_difficulty_threshold(self, threshold: float):
        """Update difficulty threshold for curriculum progression."""
        self.difficulty_threshold = threshold
        self.mask = self.difficulties <= threshold
        self.indices = torch.where(self.mask)[0]
        logger.info(f"Curriculum: threshold={threshold:.2f}, samples={len(self.indices)}")


def load_training_data(
    data_dir: str = "./data",
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load preprocessed training data and create DataLoaders.
    
    Args:
        data_dir: Directory containing preprocessed_features.npy and labels.npy
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        batch_size: Batch size for DataLoaders
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader, metadata_dict
    """
    data_path = Path(data_dir)
    
    # Load features and labels
    features_path = data_path / "preprocessed_features.npy"
    labels_path = data_path / "labels.npy"
    metadata_path = data_path / "metadata.json"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    logger.info(f"Loading features from {features_path}")
    features = np.load(features_path)
    
    logger.info(f"Loading labels from {labels_path}")
    labels = np.load(labels_path)
    
    # Load metadata if available
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
    logger.info(f"Label distribution: SELL={np.sum(labels==0)}, HOLD={np.sum(labels==1)}, BUY={np.sum(labels==2)}")
    
    # Create dataset
    dataset = HIMARIDataset(features, labels)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Avoid batch norm issues with small last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Add dataset info to metadata
    metadata.update({
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'feature_dim': features.shape[1],
        'num_classes': 3,
        'class_names': ['SELL', 'HOLD', 'BUY']
    })
    
    return train_loader, val_loader, test_loader, metadata


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.
    
    Args:
        labels: Array of labels
        
    Returns:
        Tensor of class weights for CrossEntropyLoss
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)  # Normalize
    return torch.FloatTensor(weights)


# Convenience function to check data availability
def verify_data_files(data_dir: str = "./data") -> bool:
    """Check if all required data files exist."""
    data_path = Path(data_dir)
    required = ["preprocessed_features.npy", "labels.npy"]
    
    for f in required:
        if not (data_path / f).exists():
            logger.error(f"Missing required file: {f}")
            return False
    
    return True
