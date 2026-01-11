"""
HIMARI Layer 2 - Training Infrastructure
Subsystem K: Reward Shaping (Methods K2, K4)

Purpose:
    Augment training data and shape rewards for risk-adjusted learning.

Methods:
    K2: MJD/GARCH Augmentation - Synthetic trajectory generation (delegates to preprocessing)
    K4: Sortino/Calmar Rewards - Risk-adjusted reward functions

Expected Performance:
    - Training data: 10x augmentation multiplier
    - Sortino ratio: >2.0 (vs 1.5 with simple PnL rewards)
    - Calmar ratio: >1.5
"""

from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping"""
    # Method K4: Risk-adjusted rewards
    reward_type: str = "sortino"  # or "calmar", "sharpe", "pnl"

    # Sortino parameters
    sortino_mar: float = 0.0  # Minimum acceptable return (0% risk-free rate)
    sortino_window: int = 100  # Rolling window for downside deviation

    # Calmar parameters
    calmar_window: int = 252 * 288  # 1 year of 5-min bars for max drawdown

    # Sharpe parameters
    sharpe_window: int = 100

    # Reward scaling
    reward_scale: float = 1.0  # Multiplier for final reward

    # Penalty terms
    trading_cost_penalty: float = 0.002  # 0.2% per trade
    drawdown_penalty_weight: float = 2.0  # Extra penalty for drawdowns


class RewardShaper:
    """
    Risk-adjusted reward shaping for RL training.

    Example:
        >>> shaper = RewardShaper(reward_type="sortino")
        >>> reward = shaper.compute_reward(returns_history, current_pnl)
    """

    def __init__(self, config: Optional[RewardShapingConfig] = None):
        self.config = config or RewardShapingConfig()
        self.returns_history = []
        self.pnl_history = []
        self.peak_pnl = 0.0

        logger.info(f"Reward Shaper initialized: type={self.config.reward_type}")

    def reset(self):
        """Reset history for new episode"""
        self.returns_history = []
        self.pnl_history = []
        self.peak_pnl = 0.0

    def compute_sortino_ratio(
        self,
        returns: np.ndarray,
        mar: Optional[float] = None
    ) -> float:
        """
        Compute Sortino ratio (Method K4).

        Sortino = (mean_return - MAR) / downside_deviation

        Only penalizes downside volatility, not upside.

        Args:
            returns: Array of returns
            mar: Minimum acceptable return

        Returns:
            Sortino ratio
        """
        mar = mar or self.config.sortino_mar

        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        excess_returns = returns - mar

        # Downside deviation (only negative excess returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            # No downside, perfect Sortino
            return 10.0 if mean_return > mar else 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 10.0 if mean_return > mar else 0.0

        sortino = (mean_return - mar) / downside_std
        return float(sortino)

    def compute_calmar_ratio(
        self,
        pnl_series: np.ndarray
    ) -> float:
        """
        Compute Calmar ratio (Method K4).

        Calmar = annualized_return / max_drawdown

        Args:
            pnl_series: Cumulative PnL series

        Returns:
            Calmar ratio
        """
        if len(pnl_series) < 2:
            return 0.0

        # Annualized return
        total_return = (pnl_series[-1] - pnl_series[0]) / max(abs(pnl_series[0]), 1.0)
        periods = len(pnl_series)
        periods_per_year = 252 * 288  # 5-min bars
        annualized_return = total_return * (periods_per_year / periods)

        # Max drawdown
        running_max = np.maximum.accumulate(pnl_series)
        drawdowns = (pnl_series - running_max) / np.maximum(running_max, 1.0)
        max_dd = abs(np.min(drawdowns))

        if max_dd == 0:
            return 10.0 if annualized_return > 0 else 0.0

        calmar = annualized_return / max_dd
        return float(calmar)

    def compute_sharpe_ratio(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Compute Sharpe ratio.

        Sharpe = mean_return / std_return

        Args:
            returns: Array of returns

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 10.0 if mean_return > 0 else 0.0

        sharpe = mean_return / std_return
        return float(sharpe)

    def compute_reward(
        self,
        current_return: float,
        traded: bool = False,
        current_pnl: Optional[float] = None
    ) -> float:
        """
        Compute shaped reward based on configured type.

        Args:
            current_return: Return for this step
            traded: Whether a trade was executed (for cost penalty)
            current_pnl: Current cumulative PnL (for Calmar)

        Returns:
            Shaped reward value
        """
        # Update history
        self.returns_history.append(current_return)
        if current_pnl is not None:
            self.pnl_history.append(current_pnl)
            self.peak_pnl = max(self.peak_pnl, current_pnl)

        # Base reward
        if self.config.reward_type == "pnl":
            # Simple PnL
            reward = current_return

        elif self.config.reward_type == "sharpe":
            # Rolling Sharpe ratio
            window = min(len(self.returns_history), self.config.sharpe_window)
            recent_returns = np.array(self.returns_history[-window:])
            reward = self.compute_sharpe_ratio(recent_returns)

        elif self.config.reward_type == "sortino":
            # Rolling Sortino ratio (Method K4)
            window = min(len(self.returns_history), self.config.sortino_window)
            recent_returns = np.array(self.returns_history[-window:])
            reward = self.compute_sortino_ratio(recent_returns)

        elif self.config.reward_type == "calmar":
            # Rolling Calmar ratio (Method K4)
            if len(self.pnl_history) < 2:
                reward = 0.0
            else:
                window = min(len(self.pnl_history), self.config.calmar_window)
                recent_pnl = np.array(self.pnl_history[-window:])
                reward = self.compute_calmar_ratio(recent_pnl)

        else:
            logger.warning(f"Unknown reward type: {self.config.reward_type}, using PnL")
            reward = current_return

        # Trading cost penalty
        if traded:
            reward -= self.config.trading_cost_penalty

        # Drawdown penalty
        if current_pnl is not None and current_pnl < self.peak_pnl:
            drawdown_pct = (self.peak_pnl - current_pnl) / max(self.peak_pnl, 1.0)
            reward -= self.config.drawdown_penalty_weight * drawdown_pct

        # Scale
        reward *= self.config.reward_scale

        return float(reward)

    def get_episode_metrics(self) -> dict:
        """
        Get summary metrics for completed episode.

        Returns:
            Dict with Sharpe, Sortino, Calmar
        """
        if not self.returns_history:
            return {
                'sharpe': 0.0,
                'sortino': 0.0,
                'calmar': 0.0,
                'total_return': 0.0
            }

        returns = np.array(self.returns_history)
        pnl = np.array(self.pnl_history) if self.pnl_history else np.cumsum(returns)

        return {
            'sharpe': self.compute_sharpe_ratio(returns),
            'sortino': self.compute_sortino_ratio(returns),
            'calmar': self.compute_calmar_ratio(pnl),
            'total_return': float(np.sum(returns)),
            'n_steps': len(returns)
        }


class CurriculumLearning:
    """
    Curriculum learning for progressive difficulty (bonus method).

    Starts with easier scenarios (low volatility, trending) and progresses
    to harder scenarios (high volatility, ranging, crisis).
    """

    def __init__(self):
        self.difficulty_level = 0  # 0=easy, 1=medium, 2=hard
        self.performance_threshold = 1.5  # Sharpe needed to advance

    def get_scenario_weights(self) -> dict:
        """
        Get sampling weights for different market scenarios.

        Returns:
            Dict mapping scenario to sampling weight
        """
        if self.difficulty_level == 0:
            # Easy: mostly trending, low volatility
            return {
                'trending_up': 0.4,
                'trending_down': 0.3,
                'ranging': 0.2,
                'crisis': 0.1
            }
        elif self.difficulty_level == 1:
            # Medium: balanced
            return {
                'trending_up': 0.25,
                'trending_down': 0.25,
                'ranging': 0.3,
                'crisis': 0.2
            }
        else:
            # Hard: mostly difficult scenarios
            return {
                'trending_up': 0.15,
                'trending_down': 0.15,
                'ranging': 0.35,
                'crisis': 0.35
            }

    def update_difficulty(self, sharpe_ratio: float):
        """Update difficulty based on performance"""
        if sharpe_ratio > self.performance_threshold and self.difficulty_level < 2:
            self.difficulty_level += 1
            logger.info(f"Curriculum advanced to level {self.difficulty_level}")


__all__ = ['RewardShaper', 'RewardShapingConfig', 'CurriculumLearning']
