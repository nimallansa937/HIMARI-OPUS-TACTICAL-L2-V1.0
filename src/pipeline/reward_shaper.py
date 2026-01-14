"""
Sortino-Based Reward Shaper for PPO Training

Shapes raw PnL returns into training rewards that:
1. Penalize downside volatility (Sortino ratio focus)
2. Penalize excessive trading (trade cost)
3. Reward risk-adjusted returns, not just absolute returns

Why Sortino over Sharpe?
    - Sharpe penalizes all volatility equally
    - Sortino only penalizes downside volatility
    - Upside volatility (big wins) should NOT be penalized

Formula:
    shaped_reward = base_reward + sortino_bonus - trade_penalty - drawdown_penalty

Author: HIMARI Development Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque


@dataclass
class RewardComponents:
    """Breakdown of shaped reward."""
    raw_return: float           # Position * price return
    sortino_bonus: float        # Bonus for low downside risk
    trade_penalty: float        # Penalty for position change
    drawdown_penalty: float     # Penalty for being in drawdown
    total_reward: float         # Final shaped reward


class SortinoRewardShaper:
    """
    Shapes rewards to optimize for Sortino ratio.

    The key insight: we want the policy to learn that:
    - Small consistent gains > big volatile gains
    - Avoiding losses > maximizing wins
    - Holding good positions > frequent trading

    Parameters:
        target_return: Minimum acceptable return (MAR), default 0.0
        lookback_window: Window for downside deviation calc (default 50)
        trade_cost: Penalty per position change (default 0.001)
        drawdown_weight: Penalty weight for drawdowns (default 0.5)
        reward_scale: Scale factor for final reward (default 100)
    """

    def __init__(
        self,
        target_return: float = 0.0,
        lookback_window: int = 50,
        trade_cost: float = 0.001,
        drawdown_weight: float = 0.5,
        reward_scale: float = 100.0
    ):
        self.target_return = target_return
        self.lookback_window = lookback_window
        self.trade_cost = trade_cost
        self.drawdown_weight = drawdown_weight
        self.reward_scale = reward_scale

        # Rolling buffers
        self.returns_buffer: deque = deque(maxlen=lookback_window)
        self.equity_buffer: deque = deque(maxlen=lookback_window * 2)

        # State tracking
        self.prev_position: Optional[int] = None
        self.peak_equity: float = 1.0
        self.current_equity: float = 1.0
        self.n_updates: int = 0

        # Statistics
        self.total_trades: int = 0
        self.total_reward: float = 0.0

    def reset(self) -> None:
        """Reset shaper state."""
        self.returns_buffer.clear()
        self.equity_buffer.clear()
        self.prev_position = None
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.n_updates = 0
        self.total_trades = 0
        self.total_reward = 0.0

    def shape(
        self,
        price_return: float,
        position: int,
        regime_id: int = 0,
        regime_confidence: float = 1.0
    ) -> RewardComponents:
        """
        Shape raw return into training reward.

        Args:
            price_return: Raw price return (e.g., 0.01 = +1%)
            position: Current position (-1, 0, +1)
            regime_id: Current regime (0-3) for regime-aware shaping
            regime_confidence: Confidence in regime detection

        Returns:
            RewardComponents with breakdown and total shaped reward
        """
        self.n_updates += 1

        # === 1. BASE REWARD ===
        # Raw PnL: position * return
        raw_return = position * price_return

        # Update equity curve
        self.current_equity *= (1 + raw_return)
        self.equity_buffer.append(self.current_equity)
        self.returns_buffer.append(raw_return)

        # Update peak for drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # === 2. SORTINO BONUS ===
        # Reward for maintaining low downside volatility
        sortino_bonus = self._compute_sortino_bonus()

        # === 3. TRADE PENALTY ===
        # Penalize position changes to discourage overtrading
        trade_penalty = 0.0
        if self.prev_position is not None and position != self.prev_position:
            trade_penalty = self.trade_cost
            self.total_trades += 1

        self.prev_position = position

        # === 4. DRAWDOWN PENALTY ===
        # Penalize being in drawdown
        drawdown_penalty = self._compute_drawdown_penalty()

        # === 5. REGIME-ADJUSTED SCALING ===
        # Scale reward based on regime
        regime_scale = self._get_regime_scale(regime_id, regime_confidence)

        # === COMBINE ===
        total_reward = (
            raw_return * self.reward_scale
            + sortino_bonus
            - trade_penalty * self.reward_scale
            - drawdown_penalty
        ) * regime_scale

        self.total_reward += total_reward

        return RewardComponents(
            raw_return=raw_return,
            sortino_bonus=sortino_bonus,
            trade_penalty=trade_penalty,
            drawdown_penalty=drawdown_penalty,
            total_reward=total_reward
        )

    def _compute_sortino_bonus(self) -> float:
        """
        Compute Sortino-based bonus.

        Returns positive bonus when recent downside deviation is low,
        negative penalty when downside deviation is high.
        """
        if len(self.returns_buffer) < 10:
            return 0.0

        returns = np.array(list(self.returns_buffer))

        # Downside returns (below target)
        downside = returns[returns < self.target_return]

        if len(downside) == 0:
            # No downside risk = bonus
            return 0.1

        # Downside deviation
        downside_std = np.std(downside)

        # Convert to bonus/penalty
        # Low downside std = bonus, high = penalty
        if downside_std < 0.005:
            return 0.05  # Good: low downside risk
        elif downside_std < 0.01:
            return 0.0   # Neutral
        elif downside_std < 0.02:
            return -0.05  # Moderate penalty
        else:
            return -0.1   # High downside risk

    def _compute_drawdown_penalty(self) -> float:
        """
        Compute penalty for being in drawdown.

        Larger drawdowns = larger penalties.
        """
        if self.peak_equity <= 0:
            return 0.0

        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

        if drawdown <= 0.01:
            return 0.0  # Ignore small drawdowns
        elif drawdown < 0.05:
            return drawdown * self.drawdown_weight * 0.5
        elif drawdown < 0.10:
            return drawdown * self.drawdown_weight
        else:
            # Severe drawdown - higher penalty
            return drawdown * self.drawdown_weight * 2.0

    def _get_regime_scale(self, regime_id: int, confidence: float) -> float:
        """
        Scale reward based on current regime.

        In CRISIS regime, we want the policy to be more conservative,
        so we increase the penalty for losses.
        """
        # Regime scales (0=LOW_VOL, 1=TRENDING, 2=HIGH_VOL, 3=CRISIS)
        regime_scales = {
            0: 1.0,    # LOW_VOL: normal rewards
            1: 1.1,    # TRENDING: slightly higher (momentum works)
            2: 0.9,    # HIGH_VOL: slightly lower (more caution)
            3: 0.7     # CRISIS: much lower (survival mode)
        }

        base_scale = regime_scales.get(regime_id, 1.0)

        # Blend with confidence
        return base_scale * confidence + 1.0 * (1 - confidence)

    def compute_episode_sortino(self) -> float:
        """
        Compute Sortino ratio for the entire episode.

        Returns:
            Sortino ratio (higher is better, >1 is good, >2 is excellent)
        """
        if len(self.returns_buffer) < 20:
            return 0.0

        returns = np.array(list(self.returns_buffer))
        excess_returns = returns - self.target_return

        # Mean excess return
        mean_return = np.mean(excess_returns)

        # Downside deviation
        downside = excess_returns[excess_returns < 0]
        if len(downside) == 0:
            return 10.0  # Perfect (no downside)

        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return 10.0

        # Annualize (assuming hourly data)
        annualization = np.sqrt(365 * 24)
        sortino = (mean_return * annualization) / (downside_std * annualization)

        return float(np.clip(sortino, -10, 10))

    def get_statistics(self) -> dict:
        """Get shaping statistics."""
        returns = np.array(list(self.returns_buffer)) if self.returns_buffer else np.array([0])

        return {
            'n_updates': self.n_updates,
            'total_trades': self.total_trades,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.n_updates, 1),
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'max_drawdown': (self.peak_equity - min(self.equity_buffer)) / self.peak_equity if self.equity_buffer else 0,
            'sortino_ratio': self.compute_episode_sortino(),
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns))
        }


class BatchRewardShaper:
    """
    Batch reward shaping for dataset generation.

    Processes entire sequences and returns shaped rewards for all timesteps.
    """

    def __init__(self, **kwargs):
        self.shaper_kwargs = kwargs

    def shape_sequence(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
        regime_ids: np.ndarray = None,
        regime_confidences: np.ndarray = None
    ) -> np.ndarray:
        """
        Shape rewards for entire sequence.

        Args:
            returns: Price returns (T,)
            positions: Position sequence (T,) with values in {-1, 0, 1}
            regime_ids: Optional regime labels (T,)
            regime_confidences: Optional regime confidences (T,)

        Returns:
            shaped_rewards: Array of shape (T,)
        """
        T = len(returns)

        if regime_ids is None:
            regime_ids = np.zeros(T, dtype=np.int32)
        if regime_confidences is None:
            regime_confidences = np.ones(T, dtype=np.float32)

        shaper = SortinoRewardShaper(**self.shaper_kwargs)
        shaped_rewards = np.zeros(T, dtype=np.float32)

        for t in range(T):
            result = shaper.shape(
                price_return=returns[t],
                position=int(positions[t]),
                regime_id=int(regime_ids[t]),
                regime_confidence=float(regime_confidences[t])
            )
            shaped_rewards[t] = result.total_reward

        return shaped_rewards


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sortino Reward Shaper Test")
    print("=" * 60)

    # Create shaper
    shaper = SortinoRewardShaper(
        target_return=0.0,
        lookback_window=50,
        trade_cost=0.001,
        reward_scale=100.0
    )

    # Simulate trading
    np.random.seed(42)
    T = 500

    # Generate price returns
    returns = np.random.randn(T) * 0.01  # 1% daily vol

    # Simple momentum strategy
    positions = np.sign(np.convolve(returns, np.ones(5)/5, mode='same'))
    positions = positions.astype(int)

    # Shape rewards
    total_reward = 0.0
    for t in range(T):
        result = shaper.shape(
            price_return=returns[t],
            position=positions[t],
            regime_id=0
        )
        total_reward += result.total_reward

    # Print statistics
    stats = shaper.get_statistics()
    print("\nShaping Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")

    print(f"\nTotal Shaped Reward: {total_reward:.2f}")

    # Test batch shaper
    print("\n" + "=" * 60)
    print("Batch Reward Shaper Test")
    batch_shaper = BatchRewardShaper(reward_scale=100.0)
    shaped_rewards = batch_shaper.shape_sequence(returns, positions)
    print(f"Batch shaped rewards shape: {shaped_rewards.shape}")
    print(f"Mean reward: {np.mean(shaped_rewards):.4f}")
    print(f"Std reward: {np.std(shaped_rewards):.4f}")

    print("\n✅ Reward Shaper test passed!")
