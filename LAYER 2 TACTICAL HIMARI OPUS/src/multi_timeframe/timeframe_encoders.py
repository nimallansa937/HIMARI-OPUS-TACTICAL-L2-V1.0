"""
HIMARI Layer 2 - Multi-Timeframe LSTM Encoders
Subsystem C: Multi-Timeframe Fusion (Method C1)

Purpose:
    Parallel LSTM encoders for different timeframes (1m, 5m, 1h, 4h).
    Each encoder transforms its timeframe into a fixed-dim latent representation.

Architecture:
    - Input: Time series for specific timeframe
    - LSTM: 2 layers, 256 hidden dim
    - Output: 256-dim encoding vector
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from loguru import logger


class TimeframeEncoder(nn.Module):
    """
    LSTM encoder for single timeframe.
    
    Args:
        input_dim: Feature dimension (e.g., 60)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        logger.debug(f"TimeframeEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (output, (h_n, c_n))
            - output: (batch, seq_len, hidden_dim)
            - h_n: (num_layers, batch, hidden_dim)
            - c_n: (num_layers, batch, hidden_dim)
        """
        lstm_out, hidden_state = self.lstm(x, hidden)
        
        # Layer norm on output
        lstm_out = self.layer_norm(lstm_out)
        
        return lstm_out, hidden_state
    
    def get_last_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
       Get final timestep encoding.
        
        Args:
            x: Input (batch, seq_len, input_dim)
            
        Returns:
            Last hidden state (batch, hidden_dim)
        """
        lstm_out, _ = self.forward(x)
        return lstm_out[:, -1, :]  # (batch, hidden_dim)


class MultiTimeframeEncoder(nn.Module):
    """
    Parallel LSTM encoders for multiple timeframes.
    
    Example:
        >>> encoder = MultiTimeframeEncoder(
        ...     input_dim=60,
        ...     timeframes=["1m", "5m", "1h", "4h"]
        ... )
        >>> encoded = encoder({"1m": x_1m, "5m": x_5m, ...})
    """
    
    def __init__(
        self,
        input_dim: int,
        timeframes: List[str] = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if timeframes is None:
            timeframes = ["1m", "5m", "1h", "4h"]
        
        self.timeframes = timeframes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create encoder for each timeframe
        self.encoders = nn.ModuleDict({
            tf: TimeframeEncoder(input_dim, hidden_dim, num_layers, dropout)
            for tf in timeframes
        })
        
        logger.info(f"MultiTimeframeEncoder initialized with {len(timeframes)} timeframes")
    
    def forward(
        self,
        inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Encode all timeframes.
        
        Args:
            inputs: Dict mapping timeframe -> tensor (batch, seq_len, input_dim)
            
        Returns:
            Dict mapping timeframe -> encoding (batch, hidden_dim)
        """
        encodings = {}
        
        for tf in self.timeframes:
            if tf in inputs:
                encodings[tf] = self.encoders[tf].get_last_hidden(inputs[tf])
            else:
                # Placeholder if timeframe missing
                batch_size = list(inputs.values())[0].shape[0]
                encodings[tf] = torch.zeros(batch_size, self.hidden_dim, device=list(inputs.values())[0].device)
        
        return encodings
    
    def get_concatenated(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get concatenated encodings from all timeframes.
        
        Returns:
            Tensor of shape (batch, num_timeframes * hidden_dim)
        """
        encodings = self.forward(inputs)
        return torch.cat([encodings[tf] for tf in self.timeframes], dim=1)
