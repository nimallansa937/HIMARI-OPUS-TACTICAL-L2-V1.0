"""
HIMARI Layer 2 - Baseline MLP Model
Simple feedforward network for BUY/HOLD/SELL classification.
This serves as a proof-of-concept before implementing full RL agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BaselineMLP(nn.Module):
    """
    Simple MLP baseline for tactical signal classification.
    
    Architecture: 60 → 128 → 64 → 32 → 3 (BUY/HOLD/SELL)
    
    This model serves as:
    1. Proof-of-concept for training pipeline
    2. Baseline for comparing against FLAG-TRADER, CGDT, CQL
    3. Quick validation of data loading and monitoring
    """
    
    def __init__(
        self,
        input_dim: int = 60,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        num_classes: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"BaselineMLP initialized: {input_dim} → {hidden_dims} → {num_classes}")
        logger.info(f"Total parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 60)
            
        Returns:
            Logits tensor of shape (batch_size, 3)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions (0=SELL, 1=HOLD, 2=BUY)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict:
        """Return model configuration for checkpointing."""
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'architecture': 'BaselineMLP'
        }


class BaselineMLPWithUncertainty(BaselineMLP):
    """
    Extended MLP with uncertainty estimation via MC Dropout.
    
    Can be used for:
    - Confidence-based position sizing
    - Abstaining from low-confidence predictions
    """
    
    def __init__(self, *args, mc_samples: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_samples = mc_samples
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout for uncertainty estimation.
        
        Returns:
            predictions: Mean predictions
            uncertainty: Standard deviation (epistemic uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.mc_samples):
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        
        self.eval()
        
        # Stack and compute statistics
        stacked = torch.stack(predictions, dim=0)  # (mc_samples, batch, num_classes)
        mean_pred = stacked.mean(dim=0)
        uncertainty = stacked.std(dim=0).mean(dim=-1)  # Average std across classes
        
        return mean_pred, uncertainty


# Factory function for easy model creation
def create_baseline_model(
    model_type: str = "mlp",
    **kwargs
) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: One of "mlp", "mlp_uncertainty"
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    models = {
        "mlp": BaselineMLP,
        "mlp_uncertainty": BaselineMLPWithUncertainty
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)
