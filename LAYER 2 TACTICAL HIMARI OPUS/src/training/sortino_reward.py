"""
HIMARI Layer 2 - Sortino Reward Function
Simple Sortino-based reward for Transformer-A2C training.

Based on lessons learned from LSTM-PPO training failures:
- Complex reward shaping (6 components) hurt performance
- Over-engineered rewards caused over-conservative behavior
- Simple, aligned rewards work better
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


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

        # Log suspicious values (likely noise, not skill)
        if abs(sharpe) > 5.0:
            logger.warning(
                f"Suspicious Sharpe: {sharpe:.2f} (mean_ret={mean_ret:.6f}, "
                f"std_ret={std_ret:.6f}, n={len(returns)}) - likely noise"
            )

        return float(sharpe)  # Return actual value without clipping
    
    def get_episode_sortino(self) -> float:
        """Calculate Sortino ratio for completed episode."""
        if len(self._returns_buffer) < 2:
            return 0.0

        returns = np.array(self._returns_buffer)
        mean_ret = np.mean(returns)

        # Downside deviation only
        negative_returns = returns[returns < self.target_return]
        if len(negative_returns) < 2:
            # Not enough downside data to calculate Sortino reliably
            # Return 0 instead of 10 to avoid misleading high values
            logger.debug(
                f"Insufficient downside returns ({len(negative_returns)} samples) "
                f"for Sortino calculation"
            )
            return 0.0

        downside_std = np.std(negative_returns)
        if downside_std < 1e-8:
            # Zero downside deviation - undefined Sortino
            return 0.0

        # Annualized
        sortino = (mean_ret / downside_std) * np.sqrt(105120)

        # Log suspicious values
        if abs(sortino) > 5.0:
            logger.warning(
                f"Suspicious Sortino: {sortino:.2f} (mean_ret={mean_ret:.6f}, "
                f"downside_std={downside_std:.6f}) - likely noise"
            )

        return float(sortino)  # Return actual value without clipping
    
    def get_total_return(self) -> float:
        """Calculate total cumulative return."""
        if len(self._returns_buffer) == 0:
            return 0.0
        return float(np.sum(self._returns_buffer))
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown during episode."""
        if len(self._returns_buffer) < 2:
            return 0.0
        
        cumulative = np.cumsum(self._returns_buffer)
        cumulative = cumulative + 1.0  # Start from 1.0
        
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        return float(np.min(drawdowns))  # Most negative
    
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


# ==============================================================================
# Reward Function Factory
# ==============================================================================

def create_reward_function(
    reward_type: str = "sortino",
    **kwargs
) -> SimpleSortinoReward:
    """
    Factory function to create reward function.
    
    Args:
        reward_type: "sortino" or "sortino_drawdown"
        **kwargs: Additional arguments for reward function
        
    Returns:
        Reward function instance
    """
    if reward_type == "sortino":
        return SimpleSortinoReward(**kwargs)
    elif reward_type == "sortino_drawdown":
        return SortinoWithDrawdownPenalty(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
