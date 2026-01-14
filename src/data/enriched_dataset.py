"""
PyTorch Dataset Wrapper for Enriched Trading Data

Provides efficient data loading for PPO/A2C training with:
    - Context windows (sequence of past observations)
    - Regime conditioning (regime_id for each timestep)
    - Proper train/val/test splits

Usage:
    from src.data import EnrichedTradingDataset, load_enriched_dataset

    # Load dataset
    train_samples, val_samples, metadata = load_enriched_dataset("data/enriched.pkl")

    # Create PyTorch datasets
    train_dataset = EnrichedTradingDataset(train_samples, context_len=100)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    for batch in train_loader:
        features = batch['features']        # (B, context_len, feature_dim)
        regime_ids = batch['regime_ids']    # (B, context_len)
        prices = batch['prices']            # (B, context_len)
        returns = batch['returns']          # (B, context_len)

Author: HIMARI Development Team
Date: January 2026
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_enriched_dataset(path: str) -> Tuple[List, List, List, Dict]:
    """
    Load enriched dataset from pickle file.

    Args:
        path: Path to .pkl file created by DatasetGenerator

    Returns:
        (train_samples, val_samples, test_samples, metadata)
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return (
        data['train'],
        data['val'],
        data['test'],
        data['metadata']
    )


class EnrichedTradingDataset(Dataset):
    """
    PyTorch Dataset for enriched trading samples with context windows.

    Each __getitem__ returns a context window of `context_len` consecutive
    samples, enabling the policy to learn temporal patterns.

    Parameters:
        samples: List of EnrichedSample objects
        context_len: Number of timesteps in context window (default 100)
        use_denoised: Whether to use denoised features (default True)
        normalize: Whether to normalize features (default True)
        device: Target device for tensors (default 'cpu')

    Returns dict per sample:
        - features: (context_len, feature_dim) float tensor
        - regime_ids: (context_len,) long tensor
        - regime_confidences: (context_len,) float tensor
        - prices: (context_len,) float tensor
        - returns: (context_len,) float tensor
        - current_idx: int, index in original sequence
    """

    def __init__(
        self,
        samples: List[Any],
        context_len: int = 100,
        use_denoised: bool = True,
        normalize: bool = True,
        device: str = 'cpu'
    ):
        self.samples = samples
        self.context_len = context_len
        self.use_denoised = use_denoised
        self.normalize = normalize
        self.device = device

        # Validate
        if len(samples) <= context_len:
            raise ValueError(
                f"Not enough samples ({len(samples)}) for context_len ({context_len})"
            )

        # Pre-extract arrays for faster access
        self._preprocess()

    def _preprocess(self):
        """Pre-extract numpy arrays for efficient batching."""
        n = len(self.samples)

        # Determine feature dim from first sample
        sample0 = self.samples[0]
        if self.use_denoised:
            self.feature_dim = len(sample0.features_denoised)
        else:
            self.feature_dim = len(sample0.features_raw)

        # Pre-allocate arrays
        self.features_arr = np.zeros((n, self.feature_dim), dtype=np.float32)
        self.regime_ids_arr = np.zeros(n, dtype=np.int64)
        self.regime_confs_arr = np.zeros(n, dtype=np.float32)
        self.prices_arr = np.zeros(n, dtype=np.float32)
        self.returns_arr = np.zeros(n, dtype=np.float32)

        # Extract
        for i, sample in enumerate(self.samples):
            if self.use_denoised:
                self.features_arr[i] = sample.features_denoised
            else:
                self.features_arr[i] = sample.features_raw

            self.regime_ids_arr[i] = sample.regime_id
            self.regime_confs_arr[i] = sample.regime_confidence
            self.prices_arr[i] = sample.price
            self.returns_arr[i] = sample.returns

        # Normalize features if requested
        if self.normalize:
            self.feature_mean = self.features_arr.mean(axis=0)
            self.feature_std = self.features_arr.std(axis=0) + 1e-8
            self.features_arr = (self.features_arr - self.feature_mean) / self.feature_std

            # Store normalization params for inference
            self.normalization_params = {
                'mean': self.feature_mean,
                'std': self.feature_std
            }

    def __len__(self) -> int:
        """Number of available context windows."""
        return len(self.samples) - self.context_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a context window starting at idx.

        Args:
            idx: Start index of context window

        Returns:
            Dict with tensors for features, regimes, prices, returns
        """
        # Extract context window
        end_idx = idx + self.context_len

        features = torch.from_numpy(
            self.features_arr[idx:end_idx].copy()
        ).to(self.device)

        regime_ids = torch.from_numpy(
            self.regime_ids_arr[idx:end_idx].copy()
        ).to(self.device)

        regime_confs = torch.from_numpy(
            self.regime_confs_arr[idx:end_idx].copy()
        ).to(self.device)

        prices = torch.from_numpy(
            self.prices_arr[idx:end_idx].copy()
        ).to(self.device)

        returns = torch.from_numpy(
            self.returns_arr[idx:end_idx].copy()
        ).to(self.device)

        return {
            'features': features,              # (context_len, feature_dim)
            'regime_ids': regime_ids,          # (context_len,)
            'regime_confidences': regime_confs,# (context_len,)
            'prices': prices,                  # (context_len,)
            'returns': returns,                # (context_len,)
            'current_idx': end_idx - 1         # Index of "current" timestep
        }

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim

    def get_normalization_params(self) -> Optional[Dict]:
        """Get normalization parameters for inference."""
        if self.normalize:
            return self.normalization_params
        return None


class SequentialTradingDataset(Dataset):
    """
    Sequential dataset for backtesting/evaluation.

    Unlike EnrichedTradingDataset which returns random context windows,
    this returns samples in strict sequential order for proper backtesting.

    Parameters:
        samples: List of EnrichedSample objects
        use_denoised: Whether to use denoised features
    """

    def __init__(
        self,
        samples: List[Any],
        use_denoised: bool = True
    ):
        self.samples = samples
        self.use_denoised = use_denoised

        # Pre-extract
        n = len(samples)
        sample0 = samples[0]

        if use_denoised:
            self.feature_dim = len(sample0.features_denoised)
            self.features = np.array([s.features_denoised for s in samples])
        else:
            self.feature_dim = len(sample0.features_raw)
            self.features = np.array([s.features_raw for s in samples])

        self.regime_ids = np.array([s.regime_id for s in samples])
        self.prices = np.array([s.price for s in samples])
        self.returns = np.array([s.returns for s in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.from_numpy(self.features[idx]),
            'regime_id': torch.tensor(self.regime_ids[idx]),
            'price': torch.tensor(self.prices[idx]),
            'returns': torch.tensor(self.returns[idx])
        }


def create_dataloaders(
    dataset_path: str,
    context_len: int = 100,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test DataLoaders from enriched dataset.

    Args:
        dataset_path: Path to enriched .pkl file
        context_len: Context window length
        batch_size: Batch size for training
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU

    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    # Load data
    train_samples, val_samples, test_samples, metadata = load_enriched_dataset(dataset_path)

    # Create datasets
    train_dataset = EnrichedTradingDataset(train_samples, context_len=context_len)
    val_dataset = EnrichedTradingDataset(val_samples, context_len=context_len)
    test_dataset = EnrichedTradingDataset(test_samples, context_len=context_len)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, metadata


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EnrichedTradingDataset Test")
    print("=" * 60)

    # Create mock samples for testing
    from dataclasses import dataclass
    import pandas as pd

    @dataclass
    class MockSample:
        timestamp: pd.Timestamp
        features_raw: np.ndarray
        features_denoised: np.ndarray
        regime_id: int
        regime_confidence: float
        price: float
        returns: float

    # Generate mock data
    np.random.seed(42)
    n_samples = 500
    feature_dim = 44

    mock_samples = []
    price = 100.0
    for i in range(n_samples):
        ret = np.random.randn() * 0.01
        price *= (1 + ret)

        sample = MockSample(
            timestamp=pd.Timestamp.now(),
            features_raw=np.random.randn(feature_dim).astype(np.float32),
            features_denoised=np.random.randn(feature_dim).astype(np.float32),
            regime_id=np.random.randint(0, 4),
            regime_confidence=np.random.uniform(0.5, 1.0),
            price=price,
            returns=ret
        )
        mock_samples.append(sample)

    # Create dataset
    dataset = EnrichedTradingDataset(mock_samples, context_len=100)

    print(f"\nDataset length: {len(dataset)}")
    print(f"Feature dim: {dataset.get_feature_dim()}")

    # Test __getitem__
    sample = dataset[0]
    print(f"\nSample shapes:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s}: {val.shape}")
        else:
            print(f"  {key:20s}: {val}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(loader))

    print(f"\nBatch shapes:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s}: {val.shape}")

    print("\n✅ EnrichedTradingDataset test passed!")
