#!/usr/bin/env python3
"""
HIMARI OPUS 2 - Layer 3 Tier 2 Bounded Delta Training Script
============================================================

This script trains an LSTM-PPO model to output BOUNDED DELTAS (±30%)
that adjust the Tier 1 volatility-targeted base position.

Key Differences from Raw Position Training:
1. Output is delta ∈ [-0.30, +0.30], not raw position size
2. Reward is risk-adjusted (Sortino-style) + regime compliance
3. Asymmetric loss penalty (losses hurt 2x more)
4. Explicit crisis reduction reward

Expected Performance: Sharpe 0.25-0.35 (vs -0.078 from raw training)

Usage:
    python bounded_delta_training.py --steps 500000 --device cuda
    python bounded_delta_training.py --steps 1000000 --checkpoint /path/to/500k.pt
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from enum import Enum
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# Attempt wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Regime(Enum):
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    CASCADE = "cascade"
    BULL = "bull"
    BEAR = "bear"


@dataclass
class BoundedDeltaConfig:
    """Master configuration for bounded delta training."""
    
    # Delta bounds (Layer 3 Tier 2 specification)
    delta_lower: float = -0.30
    delta_upper: float = +0.30
    
    # Reward component weights
    w_risk_adjusted: float = 1.0      # Sortino-based return
    w_delta_efficiency: float = 0.3   # Reward good delta choices
    w_drawdown_penalty: float = 2.0   # Asymmetric loss aversion
    w_regime_compliance: float = 0.5  # Reward regime-appropriate behavior
    w_smoothness: float = 0.1         # Penalize erratic delta changes
    w_survival_bonus: float = 0.2     # Bonus for avoiding ruin
    
    # Sortino/Calmar parameters
    target_return: float = 0.0        # MAR for Sortino
    drawdown_threshold: float = 0.02  # 2% drawdown triggers penalty
    max_drawdown_cap: float = 0.10    # 10% max DD for scaling
    
    # Regime-specific optimal deltas
    regime_optimal_delta: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.0,       # Neutral in normal conditions
        "high_vol": -0.15,   # Reduce 15% in high vol
        "crisis": -0.25,     # Reduce 25% in crisis  
        "cascade": -0.30,    # Max reduction in cascade
        "bull": +0.10,       # Slight increase in bull
        "bear": -0.10,       # Slight reduction in bear
        "mixed": 0.0,        # Neutral in mixed
        "volatility_cluster": -0.20,  # Reduce in vol clusters
    })


@dataclass
class Tier1Config:
    """Configuration for Tier 1 Volatility Targeting (Deterministic Base)."""
    target_vol_annual: float = 0.15
    lookback_short: int = 5
    lookback_long: int = 20
    base_fraction: float = 0.5  # Half-Kelly (conservative)
    min_position_pct: float = 0.01
    max_position_pct: float = 0.10
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0


@dataclass 
class NetworkConfig:
    """Neural network architecture configuration."""
    state_dim: int = 32
    hidden_dim: int = 256
    lstm_layers: int = 2
    sequence_length: int = 20
    dropout: float = 0.1
    
    # PPO hyperparameters
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training
    batch_size: int = 64
    update_epochs: int = 4
    buffer_size: int = 2048


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    total_steps: int = 500_000
    checkpoint_interval: int = 50_000
    log_interval: int = 10
    eval_interval: int = 100
    
    # Paths
    data_path: str = "/tmp/synthetic_data/stress_scenarios.pkl"
    model_dir: str = "/tmp/models/bounded_delta"
    checkpoint_path: Optional[str] = None  # Resume from checkpoint
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ============================================================================
# TIER 1: VOLATILITY TARGETING (DETERMINISTIC BASE)
# ============================================================================

class VolatilityTargeter:
    """
    Tier 1: Deterministic volatility-targeted position sizing.
    
    This computes the BASE position that the RL delta will adjust.
    Pure arithmetic - no neural networks in the critical path.
    """
    
    def __init__(self, config: Tier1Config):
        self.config = config
        self._vol_buffer: Deque[float] = deque(maxlen=config.lookback_long)
        
    def compute_base_position(
        self,
        portfolio_equity: float,
        realized_vol: float,
        regime: str
    ) -> Tuple[float, Dict]:
        """
        Compute base position size using volatility targeting.
        
        Formula: position = (target_vol / realized_vol) * base_fraction * equity
        
        Returns:
            base_position_pct: Position as fraction of equity [0.01, 0.10]
            diagnostics: Dict with computation details
        """
        # Update volatility buffer
        self._vol_buffer.append(realized_vol)
        
        # Compute blended volatility (short + long lookback)
        if len(self._vol_buffer) >= self.config.lookback_short:
            short_vol = np.mean(list(self._vol_buffer)[-self.config.lookback_short:])
            long_vol = np.mean(list(self._vol_buffer))
            blended_vol = 0.7 * short_vol + 0.3 * long_vol
        else:
            blended_vol = realized_vol
        
        # Prevent division by zero
        blended_vol = max(blended_vol, 0.001)
        
        # Core volatility targeting formula
        raw_position = (self.config.target_vol_annual / blended_vol) * self.config.base_fraction
        
        # Clamp to bounds
        clamped_position = np.clip(
            raw_position,
            self.config.min_position_pct,
            self.config.max_position_pct
        )
        
        diagnostics = {
            "realized_vol": realized_vol,
            "blended_vol": blended_vol,
            "raw_position_pct": raw_position,
            "clamped_position_pct": clamped_position,
            "vol_buffer_len": len(self._vol_buffer)
        }
        
        return clamped_position, diagnostics
    
    def reset(self):
        """Reset volatility buffer for new episode."""
        self._vol_buffer.clear()


# ============================================================================
# BOUNDED DELTA REWARD FUNCTION
# ============================================================================

class BoundedDeltaRewardFunction:
    """
    Reward function for training bounded delta RL policy.
    
    Key difference from raw P&L reward:
    - Rewards QUALITY of delta adjustment, not raw returns
    - Penalizes deviations from regime-optimal behavior
    - Asymmetric: losses hurt 2x more than gains help
    - Smoothness: penalizes erratic delta changes
    
    Output delta is bounded to [-0.30, +0.30] via tanh scaling.
    """
    
    def __init__(self, config: BoundedDeltaConfig):
        self.config = config
        self._returns_history: List[float] = []
        self._delta_history: List[float] = []
        self._peak_equity: float = 1.0
        self._current_equity: float = 1.0
        
    def compute_reward(
        self,
        raw_delta: float,                # Model output (unbounded)
        base_position: float,            # From Tier 1 volatility targeting  
        realized_return: float,          # Actual market return this step
        regime: str,                     # Current market regime
        volatility: float,               # Current volatility estimate
        prev_delta: float = 0.0          # Previous delta for smoothness
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward for bounded delta training.
        
        Returns:
            total_reward: Combined shaped reward
            components: Dict of individual reward components for logging
        """
        
        # 1. Bound the delta (this is what inference will do)
        bounded_delta = float(np.clip(
            np.tanh(raw_delta) * 0.30,
            self.config.delta_lower,
            self.config.delta_upper
        ))
        
        # 2. Compute actual position used  
        actual_position = base_position * (1.0 + bounded_delta)
        position_return = actual_position * realized_return
        
        # Track for metrics
        self._returns_history.append(position_return)
        self._delta_history.append(bounded_delta)
        self._current_equity *= (1 + position_return)
        self._peak_equity = max(self._peak_equity, self._current_equity)
        
        # Current drawdown
        drawdown = 1.0 - (self._current_equity / self._peak_equity) if self._peak_equity > 0 else 0.0
        
        # ===== REWARD COMPONENTS =====
        
        # Component 1: Risk-Adjusted Return (Sortino-style)
        r_risk_adjusted = self._compute_sortino_reward(position_return)
        
        # Component 2: Delta Efficiency  
        r_delta_efficiency = self._compute_delta_efficiency(
            bounded_delta, realized_return, regime
        )
        
        # Component 3: Drawdown Penalty (Asymmetric)
        r_drawdown = self._compute_drawdown_penalty(drawdown)
        
        # Component 4: Regime Compliance
        r_regime = self._compute_regime_compliance(bounded_delta, regime)
        
        # Component 5: Smoothness Penalty
        r_smoothness = self._compute_smoothness_penalty(bounded_delta, prev_delta)
        
        # Component 6: Survival Bonus
        r_survival = self._compute_survival_bonus(drawdown)
        
        # ===== COMBINE COMPONENTS =====
        total_reward = (
            self.config.w_risk_adjusted * r_risk_adjusted +
            self.config.w_delta_efficiency * r_delta_efficiency +
            self.config.w_drawdown_penalty * r_drawdown +
            self.config.w_regime_compliance * r_regime +
            self.config.w_smoothness * r_smoothness +
            self.config.w_survival_bonus * r_survival
        )
        
        components = {
            "risk_adjusted": r_risk_adjusted,
            "delta_efficiency": r_delta_efficiency,
            "drawdown_penalty": r_drawdown,
            "regime_compliance": r_regime,
            "smoothness": r_smoothness,
            "survival_bonus": r_survival,
            "bounded_delta": bounded_delta,
            "position_return": position_return,
            "drawdown": drawdown,
            "equity": self._current_equity,
            "total_reward": total_reward
        }
        
        return total_reward, components
    
    def _compute_sortino_reward(self, position_return: float) -> float:
        """Sortino-style reward: penalize downside deviation only."""
        excess_return = position_return - self.config.target_return
        
        if excess_return >= 0:
            return excess_return * 10  # Scale up small values
        else:
            # Negative return: 2x penalty (asymmetric loss aversion)
            return excess_return * 20
    
    def _compute_delta_efficiency(
        self,
        delta: float,
        realized_return: float,
        regime: str
    ) -> float:
        """Reward delta choices that align with outcomes."""
        
        if regime in ["crisis", "cascade", "volatility_cluster"]:
            # In crisis: reward ANY reduction, regardless of return
            if delta < 0:
                return 0.5 * abs(delta)
            else:
                return -1.0  # Penalize increases in crisis
        
        # Normal conditions: reward alignment
        delta_sign = np.sign(delta)
        return_sign = np.sign(realized_return)
        
        if delta_sign == return_sign:
            return abs(delta * realized_return) * 50
        elif delta_sign == 0:
            return 0.1
        else:
            return -abs(delta * realized_return) * 25
    
    def _compute_drawdown_penalty(self, drawdown: float) -> float:
        """Asymmetric penalty for drawdowns."""
        if drawdown <= self.config.drawdown_threshold:
            return 0.0
        
        excess_dd = drawdown - self.config.drawdown_threshold
        normalized_dd = min(excess_dd / self.config.max_drawdown_cap, 1.0)
        
        # Quadratic penalty
        return -(normalized_dd ** 2) * 2
    
    def _compute_regime_compliance(self, delta: float, regime: str) -> float:
        """Reward deltas that match regime-optimal behavior."""
        optimal_delta = self.config.regime_optimal_delta.get(regime, 0.0)
        deviation = abs(delta - optimal_delta)
        
        # Max deviation is 0.60 (from -0.30 to +0.30)
        return 1.0 - (deviation / 0.60)
    
    def _compute_smoothness_penalty(
        self,
        current_delta: float,
        prev_delta: float
    ) -> float:
        """Penalize erratic delta changes."""
        delta_change = abs(current_delta - prev_delta)
        
        if delta_change < 0.05:
            return 0.0
        else:
            return -(delta_change - 0.05) * 2
    
    def _compute_survival_bonus(self, drawdown: float) -> float:
        """Bonus for staying alive (avoiding ruin)."""
        if drawdown < 0.05:
            return 0.5  # Healthy
        elif drawdown < 0.10:
            return 0.2  # Stressed but OK
        elif drawdown < 0.15:
            return 0.0  # Danger zone
        else:
            return -0.5  # Near ruin
    
    def get_episode_sharpe(self) -> float:
        """Calculate Sharpe ratio for completed episode."""
        if len(self._returns_history) < 2:
            return 0.0
        
        returns = np.array(self._returns_history)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-8:
            return 0.0
        
        # Annualized (assuming 5-min bars, ~105120 per year)
        sharpe = (mean_ret / std_ret) * np.sqrt(105120)
        return float(sharpe)
    
    def reset(self):
        """Reset tracking for new episode."""
        self._returns_history = []
        self._delta_history = []
        self._peak_equity = 1.0
        self._current_equity = 1.0


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class LSTMPolicyNetwork(nn.Module):
    """
    LSTM-based policy network that outputs bounded delta.
    
    Architecture:
    - LSTM encoder for temporal patterns
    - Actor head: outputs delta mean and std
    - Critic head: outputs state value
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        
        # Actor head (outputs delta mean and log_std)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 2)  # mean, log_std
        )
        
        # Critic head (outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(
        self,
        states: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            states: [batch, seq_len, state_dim]
            hidden: LSTM hidden state tuple
            
        Returns:
            delta_mean: [batch, 1]
            delta_std: [batch, 1]  
            value: [batch, 1]
            hidden: Updated LSTM hidden state
        """
        batch_size = states.shape[0]
        
        # Project input
        x = self.input_proj(states)  # [batch, seq_len, hidden]
        
        # LSTM encoding
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last timestep output
        features = lstm_out[:, -1, :]  # [batch, hidden]
        
        # Actor output
        actor_out = self.actor(features)  # [batch, 2]
        delta_mean = actor_out[:, 0:1]
        delta_log_std = actor_out[:, 1:2]
        delta_std = torch.exp(torch.clamp(delta_log_std, -5, 2))
        
        # Critic output
        value = self.critic(features)  # [batch, 1]
        
        return delta_mean, delta_std, value, hidden
    
    def get_action(
        self,
        states: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled delta (raw, unbounded)
            log_prob: Log probability of action
            value: State value estimate
            hidden: Updated hidden state
        """
        delta_mean, delta_std, value, hidden = self.forward(states, hidden)
        
        if deterministic:
            action = delta_mean
            log_prob = torch.zeros_like(action)
        else:
            dist = Normal(delta_mean, delta_std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, hidden
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs: Log probability of actions
            values: State values
            entropy: Policy entropy
        """
        delta_mean, delta_std, value, _ = self.forward(states)
        
        dist = Normal(delta_mean, delta_std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, value, entropy


# ============================================================================
# PPO TRAINER
# ============================================================================

class BoundedDeltaPPOTrainer:
    """
    PPO trainer for bounded delta policy.
    
    Integrates:
    - Tier 1 volatility targeting (deterministic base)
    - Bounded delta reward function
    - LSTM-PPO policy optimization
    """
    
    def __init__(
        self,
        net_config: NetworkConfig,
        delta_config: BoundedDeltaConfig,
        tier1_config: Tier1Config,
        device: str = "cuda"
    ):
        self.net_config = net_config
        self.delta_config = delta_config
        self.device = torch.device(device)
        
        # Components
        self.policy = LSTMPolicyNetwork(net_config).to(self.device)
        self.tier1 = VolatilityTargeter(tier1_config)
        self.reward_fn = BoundedDeltaRewardFunction(delta_config)
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': net_config.actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': net_config.critic_lr},
            {'params': self.policy.lstm.parameters(), 'lr': net_config.actor_lr},
            {'params': self.policy.input_proj.parameters(), 'lr': net_config.actor_lr},
        ])
        
        # Rollout buffer
        self.buffer = RolloutBuffer(net_config.buffer_size)
        
        # Tracking
        self.prev_delta = 0.0
        self.hidden = None
        
    def select_action(
        self,
        state_seq: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[float, float, float]:
        """
        Select action (bounded delta) for current state.
        
        Returns:
            bounded_delta: Delta in [-0.30, +0.30]
            raw_delta: Unbounded delta (for training)
            value: State value estimate
        """
        with torch.no_grad():
            states = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            action, log_prob, value, self.hidden = self.policy.get_action(
                states, self.hidden, deterministic
            )
            
            raw_delta = action.cpu().numpy()[0, 0]
            bounded_delta = float(np.clip(np.tanh(raw_delta) * 0.30, -0.30, 0.30))
            
            return bounded_delta, raw_delta, value.cpu().numpy()[0, 0], log_prob.cpu().numpy()[0, 0]
    
    def step(
        self,
        state_seq: np.ndarray,
        market_return: float,
        volatility: float,
        regime: str,
        portfolio_equity: float = 100000.0
    ) -> Tuple[float, Dict]:
        """
        Execute one training step.
        
        Returns:
            bounded_delta: Action taken
            info: Dict with metrics
        """
        # 1. Get Tier 1 base position
        base_position, tier1_diag = self.tier1.compute_base_position(
            portfolio_equity, volatility, regime
        )
        
        # 2. Select action (bounded delta)
        bounded_delta, raw_delta, value, log_prob = self.select_action(state_seq)
        
        # 3. Compute reward
        reward, reward_components = self.reward_fn.compute_reward(
            raw_delta=raw_delta,
            base_position=base_position,
            realized_return=market_return,
            regime=regime,
            volatility=volatility,
            prev_delta=self.prev_delta
        )
        
        # 4. Store transition
        self.buffer.add(
            state=state_seq,
            action=raw_delta,
            reward=reward,
            value=value,
            log_prob=log_prob
        )
        
        # 5. Update tracking
        self.prev_delta = bounded_delta
        
        # Compile info
        info = {
            "bounded_delta": bounded_delta,
            "raw_delta": raw_delta,
            "base_position": base_position,
            "reward": reward,
            "value": value,
            **reward_components,
            **tier1_diag
        }
        
        return bounded_delta, info
    
    def end_episode(self, final_value: float = 0.0) -> Dict:
        """
        End episode and compute returns.
        
        Returns:
            episode_stats: Dict with episode metrics
        """
        # Compute episode Sharpe
        episode_sharpe = self.reward_fn.get_episode_sharpe()
        final_equity = self.reward_fn._current_equity
        
        # Finalize buffer
        self.buffer.finish_episode(final_value)
        
        # Reset for next episode
        self.tier1.reset()
        self.reward_fn.reset()
        self.prev_delta = 0.0
        self.hidden = None
        
        return {
            "episode_sharpe": episode_sharpe,
            "final_equity": final_equity
        }
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected rollouts.
        
        Returns:
            losses: Dict with loss components
        """
        if len(self.buffer) < self.net_config.batch_size:
            return {}
        
        # Get data from buffer
        states, actions, returns, advantages, old_log_probs = self.buffer.get()
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(-1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(-1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        
        for _ in range(self.net_config.update_epochs):
            # Get current policy outputs
            log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
            
            # Policy loss (clipped PPO)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.net_config.clip_epsilon,
                1 + self.net_config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_bonus = entropy.mean()
            
            # Total loss
            loss = (
                policy_loss +
                self.net_config.value_coef * value_loss -
                self.net_config.entropy_coef * entropy_bonus
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.net_config.max_grad_norm
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy_bonus.item()
        
        # Clear buffer
        self.buffer.clear()
        
        n = self.net_config.update_epochs
        return {
            "total_loss": total_loss / n,
            "policy_loss": policy_loss_sum / n,
            "value_loss": value_loss_sum / n,
            "entropy": entropy_sum / n
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "net_config": asdict(self.net_config),
            "delta_config": asdict(self.delta_config),
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.clear()
        
    def add(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        value: float,
        log_prob: float
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        
    def finish_episode(self, final_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute returns and advantages using GAE."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [final_value])
        
        # GAE computation
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        self.returns = returns.tolist()
        self.advantages = advantages.tolist()
        
    def get(self) -> Tuple[np.ndarray, ...]:
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.returns),
            np.array(self.advantages),
            np.array(self.log_probs)
        )
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        
    def __len__(self):
        return len(self.states)


# ============================================================================
# SYNTHETIC ENVIRONMENT
# ============================================================================

class SyntheticTradingEnv:
    """
    Synthetic trading environment using pre-generated stress scenarios.
    """
    
    def __init__(self, data_path: str, seq_length: int = 20):
        self.seq_length = seq_length
        self.scenarios = self._load_scenarios(data_path)
        self.current_scenario = None
        self.step_idx = 0
        
    def _load_scenarios(self, path: str) -> List[Dict]:
        """Load synthetic scenarios from pickle file."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                scenarios = pickle.load(f)
            logging.info(f"Loaded {len(scenarios)} synthetic scenarios")
            return scenarios
        else:
            logging.warning(f"No scenarios at {path}, generating defaults")
            return self._generate_default_scenarios(500)
    
    def _generate_default_scenarios(self, n: int) -> List[Dict]:
        """Generate default synthetic scenarios if none exist."""
        scenarios = []
        regime_types = ["bull", "bear", "mixed", "crash", "volatility_cluster"]
        
        for i in range(n):
            regime = random.choice(regime_types)
            length = random.randint(500, 1500)
            
            # Generate synthetic price series
            if regime == "bull":
                drift = 0.0002
                vol = 0.02
            elif regime == "bear":
                drift = -0.0001
                vol = 0.025
            elif regime == "crash":
                drift = -0.001
                vol = 0.05
            elif regime == "volatility_cluster":
                drift = 0.0
                vol = 0.04
            else:  # mixed
                drift = 0.0
                vol = 0.03
            
            returns = np.random.normal(drift, vol, length)
            prices = 100 * np.cumprod(1 + returns)
            volatility = np.abs(returns) * np.sqrt(252)
            
            # Generate features (simplified)
            features = np.random.randn(length, 32) * 0.1
            # Add some signal: regime indicator, vol, momentum
            features[:, 0] = volatility
            features[:, 1] = np.convolve(returns, np.ones(20)/20, mode='same')
            
            scenarios.append({
                "regime": regime,
                "prices": prices,
                "returns": returns,
                "volatility": volatility,
                "features": features
            })
        
        return scenarios
    
    def reset(self) -> Tuple[np.ndarray, str]:
        """
        Reset environment to new random scenario.
        
        Returns:
            initial_state: [seq_length, state_dim]
            regime: Scenario regime type
        """
        self.current_scenario = random.choice(self.scenarios)
        self.step_idx = self.seq_length
        
        initial_state = self.current_scenario["features"][:self.seq_length]
        return initial_state, self.current_scenario["regime"]
    
    def step(self) -> Tuple[np.ndarray, float, float, bool, str]:
        """
        Take one step in environment.
        
        Returns:
            state: [seq_length, state_dim]
            return_pct: Market return this step
            volatility: Current volatility
            done: Whether episode is over
            regime: Current regime
        """
        if self.step_idx >= len(self.current_scenario["returns"]) - 1:
            done = True
            state = self.current_scenario["features"][-self.seq_length:]
            return_pct = 0.0
            volatility = 0.02
        else:
            done = False
            start_idx = max(0, self.step_idx - self.seq_length + 1)
            state = self.current_scenario["features"][start_idx:self.step_idx + 1]
            
            # Pad if needed
            if len(state) < self.seq_length:
                pad = np.zeros((self.seq_length - len(state), state.shape[1]))
                state = np.vstack([pad, state])
            
            return_pct = self.current_scenario["returns"][self.step_idx]
            volatility = self.current_scenario["volatility"][self.step_idx]
            
            self.step_idx += 1
        
        regime = self.current_scenario["regime"]
        return state, return_pct, volatility, done, regime


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description="HIMARI Layer 3 Bounded Delta Training")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--data", type=str, default="/tmp/synthetic_data/stress_scenarios.pkl")
    parser.add_argument("--output", type=str, default="/tmp/models/bounded_delta")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    setup_logging()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="himari-layer3-bounded-delta",
            name=f"bounded_delta_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "total_steps": args.steps,
                "device": args.device,
                "architecture": "LSTM-PPO-BoundedDelta",
                "delta_bounds": [-0.30, 0.30],
            }
        )
        logging.info("✅ Weights & Biases logging enabled")
    
    # Initialize components
    net_config = NetworkConfig()
    delta_config = BoundedDeltaConfig()
    tier1_config = Tier1Config()
    
    trainer = BoundedDeltaPPOTrainer(
        net_config=net_config,
        delta_config=delta_config,
        tier1_config=tier1_config,
        device=args.device
    )
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)
        logging.info(f"Loaded checkpoint: {args.checkpoint}")
    
    env = SyntheticTradingEnv(args.data, seq_length=net_config.sequence_length)
    
    # Training metrics
    total_steps = 0
    episode = 0
    episode_rewards = []
    episode_sharpes = []
    recent_rewards = deque(maxlen=100)
    recent_sharpes = deque(maxlen=100)
    
    logging.info("=" * 60)
    logging.info("Starting Bounded Delta Training")
    logging.info("=" * 60)
    logging.info(f"Target steps: {args.steps:,}")
    logging.info(f"Delta bounds: [{delta_config.delta_lower}, {delta_config.delta_upper}]")
    logging.info(f"Device: {args.device}")
    logging.info("=" * 60)
    
    # Main training loop
    while total_steps < args.steps:
        state, regime = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        done = False
        while not done:
            # Get next state and market info
            next_state, market_return, volatility, done, regime = env.step()
            
            # Training step
            bounded_delta, info = trainer.step(
                state_seq=state,
                market_return=market_return,
                volatility=volatility,
                regime=regime
            )
            
            episode_reward += info["reward"]
            episode_steps += 1
            total_steps += 1
            state = next_state
            
            # PPO update when buffer is full
            if len(trainer.buffer) >= net_config.buffer_size:
                losses = trainer.update()
                if WANDB_AVAILABLE and losses:
                    wandb.log({
                        "loss/total": losses["total_loss"],
                        "loss/policy": losses["policy_loss"],
                        "loss/value": losses["value_loss"],
                        "loss/entropy": losses["entropy"],
                    }, step=total_steps)
            
            # Checkpoint
            if total_steps % 50_000 == 0:
                ckpt_path = os.path.join(args.output, f"checkpoint_{total_steps}.pt")
                trainer.save(ckpt_path)
                logging.info(f"Saved checkpoint: {ckpt_path}")
        
        # End episode
        episode_stats = trainer.end_episode()
        episode_sharpe = episode_stats["episode_sharpe"]
        
        episode += 1
        recent_rewards.append(episode_reward)
        recent_sharpes.append(episode_sharpe)
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_sharpe = np.mean(recent_sharpes)
            progress = 100 * total_steps / args.steps
            
            logging.info(
                f"Episode {episode:5d} | "
                f"Steps: {total_steps:7d}/{args.steps} ({progress:5.1f}%) | "
                f"Reward: {episode_reward:7.3f} | "
                f"Sharpe: {episode_sharpe:6.3f} | "
                f"Scenario: {regime}"
            )
            
            if WANDB_AVAILABLE:
                wandb.log({
                    "episode": episode,
                    "total_steps": total_steps,
                    "progress_pct": progress,
                    "episode_reward": episode_reward,
                    "episode_sharpe": episode_sharpe,
                    "avg_reward_100": avg_reward,
                    "avg_sharpe_100": avg_sharpe,
                    "scenario_type": regime,
                }, step=total_steps)
    
    # Final save
    final_path = os.path.join(args.output, "bounded_delta_final.pt")
    trainer.save(final_path)
    
    avg_reward = np.mean(recent_rewards)
    avg_sharpe = np.mean(recent_sharpes)
    
    logging.info("=" * 60)
    logging.info("Bounded Delta Training Complete!")
    logging.info("=" * 60)
    logging.info(f"Total episodes: {episode}")
    logging.info(f"Total steps: {total_steps:,}")
    logging.info(f"Final avg reward: {avg_reward:.3f}")
    logging.info(f"Final avg Sharpe: {avg_sharpe:.3f}")
    logging.info(f"Model saved: {final_path}")
    logging.info("=" * 60)
    
    if WANDB_AVAILABLE:
        wandb.log({
            "final_avg_reward": avg_reward,
            "final_avg_sharpe": avg_sharpe,
            "total_episodes": episode,
        })
        wandb.finish()
        logging.info("✅ W&B run finished")


if __name__ == "__main__":
    main()
