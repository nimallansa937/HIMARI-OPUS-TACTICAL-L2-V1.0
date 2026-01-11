# HIMARI OPUS 2: Layer 2 Transformer-A2C Training Guide
## Tactical Decision Model - Offline Training Implementation

**Document Version:** 1.0  
**Date:** January 2026  
**System:** HIMARI OPUS 2 Seven-Layer Cryptocurrency Trading Architecture  
**Scope:** Layer 2 Transformer-A2C - OFFLINE TRAINING  
**Target Audience:** AI IDE Agents (Cursor, Windsurf, Aider, Claude Code)  
**Expected Performance:** Sharpe 0.85-1.05 OOS (1.08-1.24 in-sample)

---

# CRITICAL: LESSONS FROM LSTM-PPO TRAINING FAILURES

Before implementing, understand what went wrong with previous training attempts:

## Training History Summary

| Run | Steps | Final Sharpe | Issue |
|-----|-------|--------------|-------|
| LSTM-PPO 500k | 500,000 | **+0.046** | ✅ Best result |
| LSTM-PPO 2M | 2,000,000 | -0.078 | ❌ Overfitting |
| Bounded Delta V2 | 500,000 | -0.037 | ❌ Reward over-engineering |
| Bounded Delta V3 | 500,000 | -0.037 | ❌ Same weights issue |

## Root Causes Identified

### Problem 1: Overfitting (2M run)
- Model memorized training sequences instead of learning patterns
- Training Sharpe kept climbing while validation degraded
- **Fix:** Dropout, early stopping, staged training with validation gates

### Problem 2: Over-Engineered Rewards (Bounded Delta)
- 6 reward components with complex weighting
- Regime compliance penalty dominated, causing over-conservative behavior
- **Fix:** Simple Sortino-based reward, constraints at inference not training

### Problem 3: No Validation Monitoring
- No early stopping when validation degraded
- Trained to completion regardless of performance
- **Fix:** Walk-forward validation with automatic early stopping

---

# PART I: ARCHITECTURE SPECIFICATION

## 1.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2 TRANSFORMER-A2C ARCHITECTURE                      │
│                         (TACR-Inspired Design)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         INPUT FEATURES                               │   │
│  │  OHLCV (5) + Technical (20) + Onchain (10) + Sentiment (5) + Regime (4)  │
│  │                        = 44 features per timestep                    │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TRANSFORMER ENCODER (TFT-style)                   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │ Positional    │  │ Multi-Head    │  │ Feed-Forward  │            │   │
│  │  │ Encoding      │──▶ Attention (8) │──▶ Network       │            │   │
│  │  │ (Learned)     │  │ + Flash Attn  │  │ + LayerNorm   │            │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│  │           × 4 Transformer Blocks                                     │   │
│  │  Context: 100 timesteps (8.3 hours @ 5-min bars)                    │   │
│  │  Hidden: 256 dims | Dropout: 0.2                                    │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                    ┌──────────────┴──────────────┐                          │
│                    │                             │                          │
│                    ▼                             ▼                          │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐          │
│  │        ACTOR HEAD           │  │        CRITIC HEAD          │          │
│  │  ┌───────────────────────┐  │  │  ┌───────────────────────┐  │          │
│  │  │ Linear(256 → 128)     │  │  │  │ Linear(256 → 128)     │  │          │
│  │  │ ReLU + Dropout(0.1)   │  │  │  │ ReLU + Dropout(0.1)   │  │          │
│  │  │ Linear(128 → 3)       │  │  │  │ Linear(128 → 1)       │  │          │
│  │  │ Softmax               │  │  │  │ (State Value)         │  │          │
│  │  └───────────────────────┘  │  │  └───────────────────────┘  │          │
│  │  Output: π(a|s) for        │  │  Output: V(s)               │          │
│  │  LONG, SHORT, FLAT         │  │                             │          │
│  └─────────────────────────────┘  └─────────────────────────────┘          │
│                    │                             │                          │
│                    └──────────────┬──────────────┘                          │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         OUTPUT TO LAYER 3                            │   │
│  │     direction: LONG | SHORT | FLAT                                   │   │
│  │     confidence: softmax probability ∈ [0, 1]                        │   │
│  │     value_estimate: V(s) for monitoring                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.2 Key Design Decisions (From SLR)

| Decision | Choice | Source |
|----------|--------|--------|
| Encoder | Transformer (not LSTM) | TACR: +18% Sharpe vs LSTM |
| Attention | Flash Attention | Dao 2022: 15% speedup, 20x memory |
| Context Length | 100 timesteps | TFT: Optimal for 5-min bars |
| Actor-Critic | A2C (not PPO) | Thach 2025: Simpler, equivalent perf |
| Action Space | Discrete (3) | More stable than continuous |
| Regularization | Dropout 0.2 | Prevents overfitting |

---

# PART II: TRAINING CONFIGURATION

## 2.1 Staged Training Protocol

**CRITICAL:** Do not train to a fixed step count. Use staged training with validation gates.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGED TRAINING PROTOCOL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: Initial Learning (0 → 200k steps)                                │
│  ├── Purpose: Learn basic market patterns                                  │
│  ├── Validation Check: Every 50k steps                                     │
│  ├── Gate: val_sharpe > 0.0 AND val_sharpe improving                       │
│  └── Action: If gate fails 2x consecutive → STOP, use best checkpoint      │
│                                                                             │
│  STAGE 2: Refinement (200k → 500k steps)                                   │
│  ├── Purpose: Fine-tune learned patterns                                   │
│  ├── Validation Check: Every 25k steps                                     │
│  ├── Gate: val_sharpe > train_sharpe * 0.7 (max 30% degradation)          │
│  └── Action: If gap > 30% → STOP (overfitting detected)                    │
│                                                                             │
│  STAGE 3: Extended (500k → 1M steps) - OPTIONAL                            │
│  ├── Purpose: Only if Stage 2 showed stable improvement                    │
│  ├── Validation Check: Every 25k steps                                     │
│  ├── Gate: val_sharpe still improving OR plateau < 3 checks                │
│  └── Action: Stop at first sign of degradation                             │
│                                                                             │
│  FINAL: Select checkpoint with BEST VALIDATION Sharpe (not training)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Data Splits (Walk-Forward)

```python
# DO NOT randomly shuffle time series data!

TRAIN_PERIOD = "2020-01-01" to "2023-06-30"   # 3.5 years
VAL_PERIOD   = "2023-07-01" to "2024-03-31"   # 9 months (gap after train)
TEST_PERIOD  = "2024-04-01" to "2024-12-31"   # 9 months (final evaluation)

# Walk-Forward Validation Windows
WALK_FORWARD_CONFIG = {
    "train_window": "6 months",
    "val_window": "1 month", 
    "step_size": "1 month",
    "min_train_samples": 50000,  # ~170 days @ 5-min bars
}
```

## 2.3 Hyperparameters

```python
@dataclass
class TransformerA2CConfig:
    """
    Hyperparameters tuned based on TACR paper + HIMARI constraints.
    
    KEY PRINCIPLE: Conservative defaults to prevent overfitting.
    """
    
    # === MODEL ARCHITECTURE ===
    input_dim: int = 44           # Features per timestep
    hidden_dim: int = 256         # Transformer hidden size
    num_heads: int = 8            # Multi-head attention
    num_layers: int = 4           # Transformer blocks
    context_length: int = 100     # Timesteps (8.3 hours @ 5-min)
    
    # === REGULARIZATION (CRITICAL) ===
    dropout: float = 0.2          # Higher than typical (prevent overfit)
    attention_dropout: float = 0.1
    weight_decay: float = 1e-4    # L2 regularization
    max_grad_norm: float = 0.5    # Gradient clipping
    
    # === A2C PARAMETERS ===
    actor_lr: float = 1e-4        # Conservative (not 3e-4)
    critic_lr: float = 3e-4       # Critic can learn faster
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE parameter
    entropy_coef: float = 0.01    # Exploration bonus
    value_coef: float = 0.5       # Critic loss weight
    
    # === TRAINING SCHEDULE ===
    batch_size: int = 64
    rollout_steps: int = 2048     # Steps before update
    max_steps: int = 500_000      # Soft limit (early stop likely)
    
    # === EARLY STOPPING ===
    patience: int = 3             # Validation checks before stop
    min_improvement: float = 0.01 # Minimum Sharpe improvement
    val_frequency: int = 25_000   # Steps between validation
    
    # === CHECKPOINTING ===
    checkpoint_frequency: int = 50_000
    keep_best_n: int = 5          # Keep top N checkpoints by val_sharpe
```

---

# PART III: REWARD FUNCTION

## 3.1 Simple Sortino Reward (Recommended)

**LESSON LEARNED:** Complex reward shaping hurt performance. Use simple, aligned rewards.

```python
class SimpleSortinoReward:
    """
    Simple Sortino-based reward function.
    
    WHY THIS WORKS:
    - Directly aligned with trading objective (risk-adjusted returns)
    - Penalizes downside deviation only (not upside volatility)
    - No complex regime penalties that can dominate
    - Lets the model discover optimal behavior through trial/error
    
    WHAT WE REMOVED (from failed bounded delta):
    - regime_compliance penalty (caused over-conservatism)
    - smoothness penalty (unnecessary constraint)
    - survival bonus (implicit in returns already)
    """
    
    def __init__(
        self,
        target_return: float = 0.0,
        downside_penalty: float = 2.0,  # Asymmetric: losses hurt 2x
        scale: float = 100.0,           # Scale small returns to meaningful rewards
    ):
        self.target_return = target_return
        self.downside_penalty = downside_penalty
        self.scale = scale
        self._returns_buffer = []
        
    def compute(
        self,
        action: int,           # 0=FLAT, 1=LONG, 2=SHORT
        market_return: float,  # Actual market return this step
        confidence: float,     # Model's confidence in action
    ) -> float:
        """
        Compute reward for this step.
        
        Args:
            action: Discrete action taken
            market_return: Actual market return (e.g., 0.001 = 0.1%)
            confidence: Model's softmax probability for chosen action
            
        Returns:
            reward: Scaled Sortino-style reward
        """
        # Convert action to position
        position = {0: 0.0, 1: 1.0, 2: -1.0}[action]
        
        # Position-weighted return
        position_return = position * market_return
        
        # Confidence-scaled (optional: reward conviction)
        # Disabled by default - can enable if desired
        # position_return *= confidence
        
        # Sortino-style asymmetric reward
        excess = position_return - self.target_return
        
        if excess >= 0:
            reward = excess * self.scale
        else:
            # Penalize losses more heavily
            reward = excess * self.scale * self.downside_penalty
        
        self._returns_buffer.append(position_return)
        return reward
    
    def get_episode_sharpe(self) -> float:
        """Calculate Sharpe for completed episode."""
        if len(self._returns_buffer) < 2:
            return 0.0
        
        returns = np.array(self._returns_buffer)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-8:
            return 0.0
        
        # Annualized (5-min bars: ~105,120 per year)
        sharpe = (mean_ret / std_ret) * np.sqrt(105120)
        return float(np.clip(sharpe, -10, 10))  # Clip extremes
    
    def reset(self):
        """Reset for new episode."""
        self._returns_buffer = []


class SortinoWithDrawdownPenalty(SimpleSortinoReward):
    """
    Extended version with mild drawdown penalty.
    
    USE IF: SimpleSortinoReward allows excessive drawdowns.
    AVOID: Heavy penalty weights that dominate the reward.
    """
    
    def __init__(
        self,
        target_return: float = 0.0,
        downside_penalty: float = 2.0,
        scale: float = 100.0,
        drawdown_threshold: float = 0.05,  # 5% before penalty kicks in
        drawdown_penalty: float = 0.5,     # MILD penalty (not 2.0!)
    ):
        super().__init__(target_return, downside_penalty, scale)
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty = drawdown_penalty
        self._peak_equity = 1.0
        self._current_equity = 1.0
    
    def compute(
        self,
        action: int,
        market_return: float,
        confidence: float,
    ) -> float:
        # Base Sortino reward
        base_reward = super().compute(action, market_return, confidence)
        
        # Track equity
        position = {0: 0.0, 1: 1.0, 2: -1.0}[action]
        position_return = position * market_return
        self._current_equity *= (1 + position_return)
        self._peak_equity = max(self._peak_equity, self._current_equity)
        
        # Calculate drawdown
        drawdown = 1.0 - (self._current_equity / self._peak_equity)
        
        # Mild penalty only if drawdown exceeds threshold
        if drawdown > self.drawdown_threshold:
            excess_dd = drawdown - self.drawdown_threshold
            penalty = -excess_dd * self.drawdown_penalty * self.scale
            return base_reward + penalty
        
        return base_reward
    
    def reset(self):
        super().reset()
        self._peak_equity = 1.0
        self._current_equity = 1.0
```

## 3.2 Reward Function Comparison

| Reward Type | Complexity | LSTM-PPO Result | Recommendation |
|-------------|------------|-----------------|----------------|
| Raw P&L | Simple | Sharpe +0.046 ✅ | Good baseline |
| SimpleSortinoReward | Simple | Expected +0.05-0.10 | **Recommended** |
| BoundedDeltaReward (6 components) | Complex | Sharpe -0.037 ❌ | Avoid |
| CVaR-constrained | Medium | Untested | Research only |

---

# PART IV: MODEL IMPLEMENTATION

## 4.1 Transformer Encoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Learned positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feedforward
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x


class TacticalTransformerEncoder(nn.Module):
    """
    Transformer encoder for Layer 2 tactical decisions.
    
    Based on TFT (Lim et al. 2019) and TACR (Kim et al. 2023).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim,
            max_len=config.context_length + 50,
            dropout=config.dropout,
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.hidden_dim,
                num_heads=config.num_heads,
                d_ff=config.hidden_dim * 4,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            encoded: [batch, hidden_dim] (last timestep representation)
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Return last timestep (for decision making)
        return x[:, -1, :]
```

## 4.2 Actor-Critic Heads

```python
class ActorHead(nn.Module):
    """
    Actor network for discrete actions (LONG/SHORT/FLAT).
    
    Output: Probability distribution over actions.
    """
    
    def __init__(self, hidden_dim: int, num_actions: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Small initialization for output layer
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] from encoder
            
        Returns:
            logits: [batch, num_actions] (unnormalized log probabilities)
        """
        return self.network(x)
    
    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: [batch] selected action indices
            log_prob: [batch] log probability of selected action
            probs: [batch, num_actions] full probability distribution
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        return action, log_prob, probs


class CriticHead(nn.Module):
    """
    Critic network for state value estimation.
    
    Output: V(s) estimate.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Standard initialization for value function
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] from encoder
            
        Returns:
            value: [batch, 1] state value estimate
        """
        return self.network(x)
```

## 4.3 Complete Model

```python
class TransformerA2C(nn.Module):
    """
    Complete Transformer-A2C model for Layer 2 tactical decisions.
    
    Architecture:
        Input → Transformer Encoder → [Actor Head, Critic Head]
        
    Output:
        - action: LONG (1), SHORT (2), FLAT (0)
        - confidence: softmax probability of chosen action
        - value: V(s) estimate for training
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = TacticalTransformerEncoder(config)
        self.actor = ActorHead(config.hidden_dim, num_actions=3, dropout=config.dropout * 0.5)
        self.critic = CriticHead(config.hidden_dim, dropout=config.dropout * 0.5)
    
    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> dict:
        """
        Forward pass for training and inference.
        
        Args:
            states: [batch, seq_len, input_dim]
            deterministic: If True, take argmax action (for inference)
            
        Returns:
            dict with: action, log_prob, value, probs, confidence
        """
        # Encode sequence
        encoded = self.encoder(states)
        
        # Get action from actor
        action, log_prob, probs = self.actor.get_action(encoded, deterministic)
        
        # Get value from critic
        value = self.critic(encoded).squeeze(-1)
        
        # Confidence is the probability of the selected action
        confidence = probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "probs": probs,
            "confidence": confidence,
            "encoded": encoded,  # For debugging/analysis
        }
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for A2C update.
        
        Args:
            states: [batch, seq_len, input_dim]
            actions: [batch] action indices
            
        Returns:
            log_probs: [batch] log probability of actions
            values: [batch] state values
            entropy: [batch] policy entropy
        """
        encoded = self.encoder(states)
        
        # Get logits and values
        logits = self.actor(encoded)
        values = self.critic(encoded).squeeze(-1)
        
        # Compute log probs and entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy()
        
        return log_probs, values, entropy
    
    @torch.no_grad()
    def predict(self, states: torch.Tensor) -> dict:
        """
        Inference-only prediction (no gradients).
        
        For production use in online pipeline.
        """
        self.eval()
        return self.forward(states, deterministic=True)
```

---

# PART V: TRAINING LOOP

## 5.1 A2C Trainer with Early Stopping

```python
import os
import json
import logging
from collections import deque
from dataclasses import asdict
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.optim as optim

# Optional: wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TransformerA2CTrainer:
    """
    A2C trainer with staged training and early stopping.
    
    KEY FEATURES (from lessons learned):
    1. Validation-based early stopping
    2. Best checkpoint selection by validation Sharpe
    3. Train/val gap monitoring for overfitting detection
    4. Simple Sortino reward (not complex shaping)
    """
    
    def __init__(
        self,
        config,
        train_env,
        val_env,
        device: str = "cuda",
        output_dir: str = "./output/transformer_a2c",
    ):
        self.config = config
        self.train_env = train_env
        self.val_env = val_env
        self.device = torch.device(device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = TransformerA2C(config).to(self.device)
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.AdamW(
            list(self.model.encoder.parameters()) + list(self.model.actor.parameters()),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = optim.AdamW(
            self.model.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )
        
        # Reward function
        self.reward_fn = SimpleSortinoReward(
            target_return=0.0,
            downside_penalty=2.0,
            scale=100.0,
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_sharpe = -float('inf')
        self.patience_counter = 0
        self.checkpoints = []
        
        # Metrics buffers
        self.train_sharpes = deque(maxlen=100)
        self.val_sharpes = deque(maxlen=20)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        if WANDB_AVAILABLE:
            wandb.init(
                project="himari-layer2-transformer-a2c",
                config=asdict(self.config),
                name=f"transformer_a2c_{self.config.hidden_dim}d_{self.config.num_layers}L",
            )
    
    def collect_rollout(self, env, steps: int) -> Dict:
        """Collect experience from environment."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state, _ = env.reset()
        self.reward_fn.reset()
        
        for _ in range(steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from model
            with torch.no_grad():
                output = self.model(state_tensor, deterministic=False)
            
            action = output["action"].item()
            log_prob = output["log_prob"].item()
            value = output["value"].item()
            confidence = output["confidence"].item()
            
            # Step environment
            next_state, market_return, done, info = env.step(action)
            
            # Compute reward
            reward = self.reward_fn.compute(action, market_return, confidence)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state
            
            if done:
                state, _ = env.reset()
                self.train_sharpes.append(self.reward_fn.get_episode_sharpe())
                self.reward_fn.reset()
        
        # Get final value for bootstrapping
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            final_value = self.model(state_tensor)["value"].item()
        
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones),
            "final_value": final_value,
        }
    
    def compute_returns_and_advantages(self, rollout: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE returns and advantages."""
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        final_value = rollout["final_value"]
        
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        # Append final value for bootstrapping
        values_ext = np.append(values, final_value)
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * next_non_terminal - values_ext[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return returns, advantages
    
    def update(self, rollout: Dict, returns: np.ndarray, advantages: np.ndarray) -> Dict:
        """Perform A2C update."""
        # Convert to tensors
        states = torch.FloatTensor(rollout["states"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Evaluate actions with current policy
        log_probs, values, entropy = self.model.evaluate_actions(states, actions)
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages_t.detach()).mean()
        
        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns_t)
        
        # Combined loss
        total_loss = (
            actor_loss +
            self.config.entropy_coef * entropy_loss +
            self.config.value_coef * critic_loss
        )
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss_with_entropy = actor_loss + self.config.entropy_coef * entropy_loss
        actor_loss_with_entropy.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.encoder.parameters()) + list(self.model.actor.parameters()),
            self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.critic.parameters(),
            self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
        }
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return Sharpe ratio."""
        self.model.eval()
        
        state, _ = self.val_env.reset()
        reward_fn = SimpleSortinoReward(scale=100.0)
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.model(state_tensor, deterministic=True)
            
            action = output["action"].item()
            confidence = output["confidence"].item()
            
            next_state, market_return, done, _ = self.val_env.step(action)
            reward_fn.compute(action, market_return, confidence)
            
            state = next_state
        
        val_sharpe = reward_fn.get_episode_sharpe()
        self.val_sharpes.append(val_sharpe)
        
        self.model.train()
        return val_sharpe
    
    def save_checkpoint(self, val_sharpe: float, tag: str = ""):
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_{self.global_step}{tag}.pt"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "val_sharpe": val_sharpe,
            "config": asdict(self.config),
        }, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append({
            "path": checkpoint_path,
            "step": self.global_step,
            "val_sharpe": val_sharpe,
        })
        
        # Keep only best N checkpoints
        self.checkpoints.sort(key=lambda x: x["val_sharpe"], reverse=True)
        while len(self.checkpoints) > self.config.keep_best_n:
            removed = self.checkpoints.pop()
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])
        
        self.logger.info(f"💾 Saved checkpoint: {checkpoint_name} (val_sharpe={val_sharpe:.4f})")
        
        return checkpoint_path
    
    def check_early_stopping(self, val_sharpe: float) -> bool:
        """
        Check if training should stop early.
        
        Returns True if should stop.
        """
        improved = val_sharpe > self.best_val_sharpe + self.config.min_improvement
        
        if improved:
            self.best_val_sharpe = val_sharpe
            self.patience_counter = 0
            self.save_checkpoint(val_sharpe, tag="_best")
            return False
        else:
            self.patience_counter += 1
            self.logger.warning(
                f"⚠️ No improvement for {self.patience_counter}/{self.config.patience} checks"
            )
            
            if self.patience_counter >= self.config.patience:
                self.logger.info("🛑 Early stopping triggered")
                return True
            return False
    
    def check_overfitting(self) -> bool:
        """
        Detect overfitting by comparing train/val gap.
        
        Returns True if overfitting detected.
        """
        if len(self.train_sharpes) < 10 or len(self.val_sharpes) < 3:
            return False
        
        train_sharpe = np.mean(list(self.train_sharpes)[-20:])
        val_sharpe = np.mean(list(self.val_sharpes)[-3:])
        
        # If validation is less than 70% of training, likely overfitting
        if train_sharpe > 0.1 and val_sharpe < train_sharpe * 0.7:
            self.logger.warning(
                f"⚠️ Overfitting detected: train={train_sharpe:.4f}, val={val_sharpe:.4f}"
            )
            return True
        
        return False
    
    def train(self):
        """
        Main training loop with staged training and early stopping.
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting Transformer-A2C Training")
        self.logger.info("=" * 70)
        self.logger.info(f"Max steps: {self.config.max_steps:,}")
        self.logger.info(f"Validation frequency: {self.config.val_frequency:,}")
        self.logger.info(f"Early stopping patience: {self.config.patience}")
        self.logger.info("=" * 70)
        
        self.model.train()
        
        while self.global_step < self.config.max_steps:
            # Collect rollout
            rollout = self.collect_rollout(self.train_env, self.config.rollout_steps)
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(rollout)
            
            # Update model
            losses = self.update(rollout, returns, advantages)
            
            self.global_step += self.config.rollout_steps
            
            # Logging
            if self.global_step % 10000 == 0:
                train_sharpe = np.mean(list(self.train_sharpes)[-10:]) if self.train_sharpes else 0
                self.logger.info(
                    f"Step {self.global_step:,}/{self.config.max_steps:,} | "
                    f"Train Sharpe: {train_sharpe:.4f} | "
                    f"Actor Loss: {losses['actor_loss']:.4f} | "
                    f"Critic Loss: {losses['critic_loss']:.4f}"
                )
                
                if WANDB_AVAILABLE:
                    wandb.log({
                        "global_step": self.global_step,
                        "train_sharpe": train_sharpe,
                        **losses,
                    })
            
            # Validation check
            if self.global_step % self.config.val_frequency == 0:
                val_sharpe = self.validate()
                train_sharpe = np.mean(list(self.train_sharpes)[-20:]) if self.train_sharpes else 0
                
                self.logger.info(
                    f"📊 Validation @ {self.global_step:,}: "
                    f"val_sharpe={val_sharpe:.4f}, train_sharpe={train_sharpe:.4f}"
                )
                
                if WANDB_AVAILABLE:
                    wandb.log({
                        "val_sharpe": val_sharpe,
                        "train_val_gap": train_sharpe - val_sharpe,
                    })
                
                # Check early stopping
                if self.check_early_stopping(val_sharpe):
                    break
                
                # Check overfitting
                if self.check_overfitting():
                    self.logger.warning("Overfitting detected, stopping training")
                    break
            
            # Regular checkpoint
            if self.global_step % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(
                    np.mean(list(self.val_sharpes)[-3:]) if self.val_sharpes else 0
                )
        
        # Training complete
        self.logger.info("=" * 70)
        self.logger.info("Training Complete!")
        self.logger.info("=" * 70)
        self.logger.info(f"Total steps: {self.global_step:,}")
        self.logger.info(f"Best validation Sharpe: {self.best_val_sharpe:.4f}")
        self.logger.info(f"Best checkpoint: {self.checkpoints[0]['path'] if self.checkpoints else 'N/A'}")
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        return self.checkpoints[0] if self.checkpoints else None
```

---

# PART VI: USAGE GUIDE

## 6.1 Training Command

```bash
# Basic training
python train_transformer_a2c.py \
    --data ./data/btc_5min_2020_2024.pkl \
    --output ./output/transformer_a2c_v1 \
    --device cuda \
    --max_steps 500000

# Resume from checkpoint
python train_transformer_a2c.py \
    --data ./data/btc_5min_2020_2024.pkl \
    --output ./output/transformer_a2c_v1 \
    --checkpoint ./output/transformer_a2c_v1/checkpoint_200000_best.pt \
    --device cuda
```

## 6.2 Expected Training Progression

| Stage | Steps | Expected Val Sharpe | Notes |
|-------|-------|---------------------|-------|
| Early | 0-100k | -0.5 to +0.2 | Learning basic patterns |
| Middle | 100k-300k | +0.2 to +0.6 | Refining strategy |
| Late | 300k-500k | +0.5 to +0.8 | May plateau or overfit |
| **Target** | Best checkpoint | **0.85-1.05** | Selected by val_sharpe |

## 6.3 Warning Signs to Watch

| Metric | Healthy | Warning | Action |
|--------|---------|---------|--------|
| Val Sharpe | Improving | Declining 3+ checks | Early stop |
| Train-Val Gap | <30% | >30% | Increase dropout |
| Entropy | >0.5 | <0.2 | Increase entropy_coef |
| Actor Loss | Decreasing | Spiking | Reduce learning rate |

## 6.4 Integration with Layer 3

```python
# After training, the model outputs to Layer 3:

output = model.predict(market_features)

layer3_input = {
    "direction": output["action"].item(),   # 0=FLAT, 1=LONG, 2=SHORT
    "confidence": output["confidence"].item(),  # 0.0 to 1.0
    "value_estimate": output["value"].item(),  # For monitoring
}

# Layer 3 then applies:
# - Tier 1: Volatility targeting (deterministic base)
# - Tier 2: Bounded delta adjustment (if using RL enhancement)
# - Tier 3-5: Hard constraints and circuit breakers
```

---

# PART VII: CHECKLIST FOR AI IDE IMPLEMENTATION

## Pre-Training Checklist

- [ ] Data prepared with proper train/val/test splits by time
- [ ] No data leakage (future data in features)
- [ ] Feature normalization configured
- [ ] GPU available and CUDA working
- [ ] Wandb account configured (optional but recommended)

## Training Checklist

- [ ] Start with default hyperparameters
- [ ] Monitor validation Sharpe every 25k steps
- [ ] Watch for train-val gap widening (overfitting)
- [ ] Let early stopping trigger naturally
- [ ] Don't manually extend training if validation degrades

## Post-Training Checklist

- [ ] Select checkpoint with best **validation** Sharpe (not training)
- [ ] Test on held-out test set (2024 data)
- [ ] Verify test Sharpe within 70-100% of validation
- [ ] Export model for inference pipeline
- [ ] Document final hyperparameters and metrics

---

# APPENDIX: QUICK REFERENCE

## A. Key Hyperparameters to Tune (if needed)

| Parameter | Default | If Overfitting | If Underfitting |
|-----------|---------|----------------|-----------------|
| dropout | 0.2 | Increase to 0.3 | Decrease to 0.1 |
| weight_decay | 1e-4 | Increase to 1e-3 | Decrease to 1e-5 |
| hidden_dim | 256 | Decrease to 128 | Increase to 512 |
| num_layers | 4 | Decrease to 2 | Increase to 6 |
| actor_lr | 1e-4 | Decrease to 5e-5 | Increase to 3e-4 |

## B. Reward Function Selection

| Situation | Recommended Reward |
|-----------|-------------------|
| First attempt | SimpleSortinoReward |
| Excessive drawdowns | SortinoWithDrawdownPenalty (mild) |
| Need exploration | Increase entropy_coef to 0.05 |
| **Never use** | Complex 6-component shaping |

## C. Training Time Estimates (A100 GPU)

| Steps | Approximate Time |
|-------|------------------|
| 100k | 15-20 minutes |
| 500k | 1.5-2 hours |
| 1M | 3-4 hours |

---

**Document Version:** 1.0  
**Created:** January 2026  
**Based on:** SLR findings (TACR, TFT, A2C papers) + LSTM-PPO training lessons  
**For:** HIMARI OPUS 2 Layer 2 Tactical Decision Engine
