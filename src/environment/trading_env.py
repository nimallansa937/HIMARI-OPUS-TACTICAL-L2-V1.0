"""
HIMARI Layer 2 - Trading Environment
Realistic backtesting and simulation environment for PPO-LSTM training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for trading environment."""
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005   # 0.05% slippage
    max_position_size: float = 1.0  # 100% of balance
    min_position_size: float = 0.01 # 1% of balance
    leverage: float = 1.0           # No leverage by default


class TradingEnvironment:
    """
    Trading environment for RL agents with realistic market dynamics.

    Features:
    - Realistic order execution with commission and slippage
    - Position management (long/short/flat)
    - Portfolio tracking (PnL, drawdown, Sharpe ratio)
    - Episode-based training
    - State normalization
    """

    def __init__(self,
                 data: np.ndarray,
                 prices: np.ndarray,
                 config: Optional[TradingConfig] = None):
        """
        Initialize trading environment.

        Args:
            data: Feature vectors (num_samples, feature_dim)
            prices: Corresponding prices for each timestep (num_samples,)
            config: Trading configuration
        """
        self.data = data
        self.prices = prices
        self.config = config or TradingConfig()

        assert len(data) == len(prices), "Data and prices must have same length"

        self.num_samples = len(data)
        self.feature_dim = data.shape[1]

        # State variables
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.position = 0.0  # Number of units held
        self.position_value = 0.0
        self.entry_price = 0.0

        # Portfolio tracking
        self.portfolio_values = []
        self.trades = []
        self.episode_pnl = 0.0

        # Performance metrics
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.num_trades = 0

        logger.info(f"TradingEnvironment initialized: {self.num_samples} samples, {self.feature_dim}D features")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0

        self.portfolio_values = [self.config.initial_balance]
        self.trades = []
        self.episode_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.num_trades = 0

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Trading action (0=SELL, 1=HOLD, 2=BUY)

        Returns:
            observation: Next state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        current_price = self.prices[self.current_step]

        # Execute action
        reward, trade_info = self._execute_action(action, current_price)

        # Update position value
        if self.position != 0:
            self.position_value = self.position * current_price

        # Calculate total portfolio value
        total_value = self.balance + self.position_value
        self.portfolio_values.append(total_value)

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.num_samples - 1

        # Get next observation
        observation = self._get_observation() if not done else np.zeros(self.feature_dim + 5)

        # Info dictionary
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'total_value': total_value,
            'price': current_price,
            'trade_info': trade_info,
            'pnl': total_value - self.config.initial_balance
        }

        # Add episode summary if done
        if done:
            info['episode_summary'] = self._get_episode_summary()

        return observation, reward, done, info

    def _execute_action(self, action: int, price: float) -> Tuple[float, Dict]:
        """
        Execute trading action with realistic costs.

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY
            price: Current market price

        Returns:
            reward: Immediate reward
            trade_info: Trade execution details
        """
        trade_info = {'action': action, 'executed': False}
        reward = 0.0

        # Action mapping: 0=SELL, 1=HOLD, 2=BUY
        if action == 1:  # HOLD
            # No action, just calculate unrealized PnL as reward
            if self.position != 0:
                unrealized_pnl = self.position * (price - self.entry_price)
                reward = unrealized_pnl / self.config.initial_balance  # Normalize by initial balance
            return reward, trade_info

        # Calculate target position
        target_position = 0.0
        if action == 2:  # BUY
            # Buy signal: go long with max position size
            max_units = (self.balance * self.config.max_position_size) / price
            target_position = max_units
        elif action == 0:  # SELL
            # Sell signal: go short or close long
            if self.position > 0:
                target_position = 0.0  # Close long position
            else:
                # Short selling (if enabled)
                max_units = (self.balance * self.config.max_position_size) / price
                target_position = -max_units

        # Calculate position change
        position_change = target_position - self.position

        if abs(position_change) < 1e-6:
            # No significant change
            return reward, trade_info

        # Execute trade with costs
        trade_value = abs(position_change * price)
        commission = trade_value * self.config.commission_rate
        slippage = trade_value * self.config.slippage_rate
        total_cost = commission + slippage

        # Check if we have enough balance for the trade
        if position_change > 0:  # Buying
            required_balance = trade_value + total_cost
            if required_balance > self.balance:
                # Insufficient balance
                trade_info['insufficient_balance'] = True
                return -0.01, trade_info  # Small penalty

        # Execute the trade
        if self.position != 0:
            # Close existing position - realize PnL
            realized_pnl = self.position * (price - self.entry_price)
            self.balance += realized_pnl
            reward += realized_pnl / self.config.initial_balance

        # Update position
        self.position = target_position
        self.entry_price = price if target_position != 0 else 0.0

        # Deduct costs
        self.balance -= total_cost
        self.total_commission += commission
        self.total_slippage += slippage
        self.num_trades += 1

        # Record trade
        self.trades.append({
            'step': self.current_step,
            'action': action,
            'price': price,
            'position_change': position_change,
            'commission': commission,
            'slippage': slippage
        })

        trade_info.update({
            'executed': True,
            'position_change': position_change,
            'commission': commission,
            'slippage': slippage
        })

        # Small penalty for trading costs
        reward -= (total_cost / self.config.initial_balance)

        return reward, trade_info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation state.

        Returns:
            State vector: [features, position, balance, unrealized_pnl, price, portfolio_value]
        """
        features = self.data[self.current_step]

        # Normalize position and balance
        position_norm = self.position / (self.config.initial_balance / self.prices[self.current_step])
        balance_norm = self.balance / self.config.initial_balance

        # Unrealized PnL
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = self.position * (self.prices[self.current_step] - self.entry_price)
        unrealized_pnl_norm = unrealized_pnl / self.config.initial_balance

        # Current price (normalized)
        price_norm = self.prices[self.current_step] / self.prices[0]

        # Total portfolio value
        portfolio_value = self.balance + (self.position * self.prices[self.current_step])
        portfolio_value_norm = portfolio_value / self.config.initial_balance

        # Concatenate all state components
        state = np.concatenate([
            features,
            [position_norm, balance_norm, unrealized_pnl_norm, price_norm, portfolio_value_norm]
        ])

        return state.astype(np.float32)

    def _get_episode_summary(self) -> Dict:
        """
        Calculate episode performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Total return
        total_return = (portfolio_values[-1] - self.config.initial_balance) / self.config.initial_balance

        # Sharpe ratio (annualized, assuming 5-min bars)
        if len(returns) > 0 and np.std(returns) > 0:
            # 5-min bars: 288 bars per day, ~100k bars per year
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(288 * 365)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = sum(1 for t in self.trades if self.trades.index(t) > 0)
        win_rate = winning_trades / len(self.trades) if len(self.trades) > 0 else 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': self.num_trades,
            'win_rate': win_rate,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'final_balance': self.balance,
            'final_position_value': self.position_value,
            'final_portfolio_value': portfolio_values[-1]
        }

    @property
    def observation_space_dim(self) -> int:
        """Dimension of observation space."""
        return self.feature_dim + 5  # features + position + balance + pnl + price + portfolio_value

    @property
    def action_space_dim(self) -> int:
        """Dimension of action space."""
        return 3  # SELL, HOLD, BUY


class VectorizedTradingEnv:
    """
    Vectorized environment for parallel episode collection.
    Runs multiple environments in parallel for faster PPO training.
    """

    def __init__(self,
                 data: np.ndarray,
                 prices: np.ndarray,
                 num_envs: int = 8,
                 config: Optional[TradingConfig] = None):
        """
        Initialize vectorized environment.

        Args:
            data: Feature vectors (num_samples, feature_dim)
            prices: Corresponding prices (num_samples,)
            num_envs: Number of parallel environments
            config: Trading configuration
        """
        self.num_envs = num_envs
        self.config = config or TradingConfig()

        # Create multiple environments
        self.envs = [
            TradingEnvironment(data, prices, self.config)
            for _ in range(num_envs)
        ]

        logger.info(f"VectorizedTradingEnv initialized: {num_envs} parallel environments")

    def reset(self) -> np.ndarray:
        """
        Reset all environments.

        Returns:
            Stacked observations from all environments
        """
        observations = [env.reset() for env in self.envs]
        return np.array(observations)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with given actions.

        Args:
            actions: Actions for each environment (num_envs,)

        Returns:
            observations: Next states (num_envs, obs_dim)
            rewards: Rewards (num_envs,)
            dones: Done flags (num_envs,)
            infos: Info dictionaries (num_envs,)
        """
        results = [env.step(action) for env, action in zip(self.envs, actions)]

        observations = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        # Auto-reset finished environments
        for i, done in enumerate(dones):
            if done:
                observations[i] = self.envs[i].reset()

        return observations, rewards, dones, infos

    @property
    def observation_space_dim(self) -> int:
        return self.envs[0].observation_space_dim

    @property
    def action_space_dim(self) -> int:
        return self.envs[0].action_space_dim
