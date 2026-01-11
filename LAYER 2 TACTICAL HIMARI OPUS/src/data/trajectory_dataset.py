"""
HIMARI Layer 2 - Trajectory Dataset
Preprocessing for Decision Transformer and FLAG-TRADER models.
Converts static data to trajectory format with return-to-go.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    Dataset for transformer-based models (CGDT, FLAG-TRADER).

    Formats data as trajectories: sequences of (state, action, reward, return-to-go).
    """

    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray,
                 prices: Optional[np.ndarray] = None,
                 context_length: int = 64,
                 reward_scale: float = 1.0):
        """
        Initialize trajectory dataset.

        Args:
            features: Feature array (num_samples, feature_dim)
            labels: Label array (num_samples,) - 0=SELL, 1=HOLD, 2=BUY
            prices: Optional price array for reward calculation (num_samples,)
            context_length: Length of trajectory sequences
            reward_scale: Scaling factor for rewards
        """
        super().__init__()

        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.context_length = context_length
        self.reward_scale = reward_scale

        # Calculate rewards from price changes or use label-based rewards
        if prices is not None:
            self.rewards = self._calculate_returns(prices, labels)
        else:
            # Synthetic rewards based on labels
            self.rewards = self._synthetic_rewards(labels)

        # Calculate return-to-go (cumulative future returns)
        self.returns_to_go = self._calculate_returns_to_go(self.rewards)

        # Create trajectories
        self.trajectories = self._create_trajectories()

        logger.info(f"TrajectoryDataset initialized: {len(self.trajectories)} trajectories, "
                   f"context_length={context_length}")

    def _calculate_returns(self, prices: np.ndarray, labels: np.ndarray) -> torch.Tensor:
        """
        Calculate actual returns based on price changes.

        Args:
            prices: Price array
            labels: Action labels

        Returns:
            Rewards tensor
        """
        rewards = np.zeros(len(prices))

        # Calculate returns based on action taken
        for i in range(len(prices) - 1):
            price_change = (prices[i + 1] - prices[i]) / prices[i]
            action = labels[i]

            if action == 2:  # BUY
                reward = price_change
            elif action == 0:  # SELL
                reward = -price_change
            else:  # HOLD
                reward = 0.0

            rewards[i] = reward * self.reward_scale

        return torch.FloatTensor(rewards)

    def _synthetic_rewards(self, labels: np.ndarray) -> torch.Tensor:
        """
        Create synthetic rewards when prices are not available.

        Args:
            labels: Action labels

        Returns:
            Synthetic rewards
        """
        # Simple reward: small positive for any action, encourage trading
        rewards = np.ones(len(labels)) * 0.01

        # Penalty for excessive holding
        for i in range(len(labels)):
            if labels[i] == 1:  # HOLD
                rewards[i] = -0.001

        return torch.FloatTensor(rewards)

    def _calculate_returns_to_go(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Calculate return-to-go (cumulative future returns).

        Args:
            rewards: Reward tensor

        Returns:
            Return-to-go tensor
        """
        returns_to_go = torch.zeros_like(rewards)
        cumulative = 0.0

        for i in reversed(range(len(rewards))):
            cumulative = rewards[i] + 0.99 * cumulative  # gamma=0.99
            returns_to_go[i] = cumulative

        return returns_to_go

    def _create_trajectories(self) -> List[dict]:
        """
        Create trajectory sequences from data.

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        num_samples = len(self.features)

        # Create overlapping windows
        for i in range(num_samples - self.context_length + 1):
            end_idx = i + self.context_length

            trajectory = {
                'states': self.features[i:end_idx],           # (context_length, feature_dim)
                'actions': self.labels[i:end_idx],            # (context_length,)
                'rewards': self.rewards[i:end_idx],           # (context_length,)
                'returns_to_go': self.returns_to_go[i:end_idx],  # (context_length,)
                'timesteps': torch.arange(self.context_length)    # (context_length,)
            }

            trajectories.append(trajectory)

        return trajectories

    def __len__(self) -> int:
        """Number of trajectories."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict:
        """Get trajectory at index."""
        return self.trajectories[idx]


class SequenceDataset(Dataset):
    """
    Simple sequence dataset for FLAG-TRADER (without return-to-go).

    Just state sequences and action labels.
    """

    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray,
                 context_length: int = 256):
        """
        Initialize sequence dataset.

        Args:
            features: Feature array (num_samples, feature_dim)
            labels: Label array (num_samples,)
            context_length: Sequence length
        """
        super().__init__()

        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.context_length = context_length

        # Create sequences
        self.sequences = self._create_sequences()

        logger.info(f"SequenceDataset initialized: {len(self.sequences)} sequences, "
                   f"context_length={context_length}")

    def _create_sequences(self) -> List[dict]:
        """Create state-action sequences."""
        sequences = []
        num_samples = len(self.features)

        for i in range(num_samples - self.context_length + 1):
            end_idx = i + self.context_length

            sequence = {
                'states': self.features[i:end_idx],    # (context_length, feature_dim)
                'actions': self.labels[i:end_idx],     # (context_length,)
            }

            sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        return self.sequences[idx]


def create_trajectory_dataloader(features: np.ndarray,
                                 labels: np.ndarray,
                                 prices: Optional[np.ndarray] = None,
                                 context_length: int = 64,
                                 batch_size: int = 256,
                                 shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for trajectory-based training (CGDT).

    Args:
        features: Feature array
        labels: Label array
        prices: Optional prices for reward calculation
        context_length: Trajectory length
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    dataset = TrajectoryDataset(
        features=features,
        labels=labels,
        prices=prices,
        context_length=context_length
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )

    return dataloader


def create_sequence_dataloader(features: np.ndarray,
                               labels: np.ndarray,
                               context_length: int = 256,
                               batch_size: int = 64,
                               shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for sequence-based training (FLAG-TRADER).

    Args:
        features: Feature array
        labels: Label array
        context_length: Sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    dataset = SequenceDataset(
        features=features,
        labels=labels,
        context_length=context_length
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )

    return dataloader


def collate_trajectories(batch: List[dict]) -> dict:
    """
    Collate function for trajectory batches.

    Args:
        batch: List of trajectory dictionaries

    Returns:
        Batched trajectory dictionary
    """
    states = torch.stack([item['states'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    rewards = torch.stack([item['rewards'] for item in batch])
    returns_to_go = torch.stack([item['returns_to_go'] for item in batch])
    timesteps = torch.stack([item['timesteps'] for item in batch])

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'returns_to_go': returns_to_go,
        'timesteps': timesteps
    }


def collate_sequences(batch: List[dict]) -> dict:
    """
    Collate function for sequence batches.

    Args:
        batch: List of sequence dictionaries

    Returns:
        Batched sequence dictionary
    """
    states = torch.stack([item['states'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])

    return {
        'states': states,
        'actions': actions
    }
