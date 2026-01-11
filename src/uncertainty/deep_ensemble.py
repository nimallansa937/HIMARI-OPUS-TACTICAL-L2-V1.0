"""
HIMARI Layer 2 - Part F: Deep Ensemble
F4: Epistemic uncertainty via ensemble disagreement.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeepEnsembleConfig:
    """Deep ensemble configuration."""
    n_members: int = 5
    hidden_dim: int = 128
    dropout: float = 0.1


class EnsembleMember(nn.Module):
    """Single ensemble member network."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepEnsemble:
    """
    Deep ensemble for epistemic uncertainty quantification.
    Disagreement between models indicates epistemic uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 3,
        config: Optional[DeepEnsembleConfig] = None,
        device: str = 'cpu'
    ):
        self.config = config or DeepEnsembleConfig()
        self.device = device
        
        self.members = nn.ModuleList([
            EnsembleMember(input_dim, output_dim, self.config.hidden_dim)
            for _ in range(self.config.n_members)
        ]).to(device)
        
    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[np.ndarray, float, Dict]:
        """Predict with epistemic uncertainty estimate."""
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        predictions = []
        for member in self.members:
            member.eval()
            pred = member(x).cpu().numpy()
            predictions.append(pred)
            
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        epistemic = float(np.mean(variance))
        
        return mean_pred.squeeze(), epistemic, {'variance': variance}
        
    def get_disagreement(self, x: torch.Tensor) -> float:
        """Get normalized disagreement score (0-1)."""
        _, epistemic, _ = self.predict_with_uncertainty(x)
        return min(epistemic / 0.25, 1.0)
