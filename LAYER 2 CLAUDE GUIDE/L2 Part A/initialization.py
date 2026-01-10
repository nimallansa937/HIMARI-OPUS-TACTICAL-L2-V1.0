# ============================================================================
# FILE: initialization.py
# PURPOSE: Neural network weight initialization utilities
# STATUS: KEEP from v4.0
# LATENCY: N/A (applied once at init)
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def orthogonal_init(
    module: nn.Module,
    gain: float = 1.0,
    bias_const: float = 0.0
) -> nn.Module:
    """
    Apply orthogonal initialization to a module.
    
    Why orthogonal initialization?
    - Standard random init causes gradient explosion/vanishing
    - Orthogonal matrices preserve vector norms through layers
    - Accelerates convergence by 15-30%
    - Improves final performance by avoiding poor local minima
    
    Theory: For weight matrix W with orthogonal columns,
    ||Wx|| = ||x|| for all x, preserving gradient magnitude.
    
    Args:
        module: PyTorch module to initialize
        gain: Scaling factor (1.0 for tanh, sqrt(2) for ReLU)
        bias_const: Constant for bias initialization
        
    Returns:
        Initialized module
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias_const)
    
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param, bias_const)
    
    return module


def init_weights(
    model: nn.Module,
    init_type: str = 'orthogonal',
    gain: float = 1.0
) -> nn.Module:
    """
    Initialize all weights in a model.
    
    Args:
        model: PyTorch model
        init_type: 'orthogonal', 'xavier', 'kaiming', or 'normal'
        gain: Scaling factor
        
    Returns:
        Initialized model
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        if classname.find('Linear') != -1 or classname.find('Conv') != -1:
            if init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
        elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if init_type == 'orthogonal':
                        nn.init.orthogonal_(param, gain=gain)
                    else:
                        nn.init.xavier_uniform_(param, gain=gain)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    model.apply(init_func)
    return model


class InitializedLinear(nn.Linear):
    """Linear layer with built-in orthogonal initialization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gain: float = 1.0
    ):
        super().__init__(in_features, out_features, bias)
        nn.init.orthogonal_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class InitializedLSTM(nn.LSTM):
    """LSTM with built-in orthogonal initialization."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        gain: float = 1.0
    ):
        super().__init__(
            input_size, hidden_size, num_layers,
            bias, batch_first, dropout, bidirectional
        )
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
