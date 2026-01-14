"""
HIMARI Layer 2 - Enhanced Sortino Reward with Anti-Overtrading

Experiment 10: Combines Experiment 7 entropy settings with Experiment 9 carry cost
PLUS new anti-overtrading features:
1. Trade cooldown penalty
2. Action persistence reward  
3. Minimum hold time (soft constraint)

For 1-hour data, settings adjusted accordingly.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SortinoAntiOvertrade:
    """
    Sortino reward function with anti-overtrading enhancements.
    
    NEW FEATURES:
    1. Carry cost (from Exp 9): Penalize holding positions
    2. Trade cooldown: Penalize frequent trading
    3. Action persistence: Reward staying in same action
    4. Minimum hold time: Soft penalty for exiting too early
    
    For 1H data (vs 5min):
    - 1 bar = 1 hour (not 5 min)
    - Carry cost per bar should be ~12x lower (1 hour vs 5 min)
    - Suggested carry_cost: 0.000004 (0.0004%/bar = 0.01%/day)
    """
    
    def __init__(
        self,
        # Base Sortino
        target_return: float = 0.0,
        downside_penalty: float = 2.0,
        scale: float = 100.0,
        
        # Transaction costs
        trading_fee: float = 0.001,      # 0.1% per trade
        slippage: float = 0.0005,        # 0.05% slippage
        
        # Carry cost (Exp 9 equivalent for 1H)
        carry_cost: float = 0.000004,    # 0.0004%/bar = 0.01%/day
        
        # NEW: Anti-overtrading
        cooldown_periods: int = 4,       # 4 bars (4 hours) cooldown
        cooldown_penalty: float = 0.0002, # 0.02% penalty within cooldown
        persistence_bonus: float = 0.00005, # 0.005% bonus for same action
        min_hold_bars: int = 2,          # Minimum 2 bars (2 hours) hold
        early_exit_penalty: float = 0.0001, # 0.01% penalty for early exit
    ):
        self.target_return = target_return
        self.downside_penalty = downside_penalty
        self.scale = scale
        
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.carry_cost = carry_cost
        
        # Anti-overtrading params
        self.cooldown_periods = cooldown_periods
        self.cooldown_penalty = cooldown_penalty
        self.persistence_bonus = persistence_bonus
        self.min_hold_bars = min_hold_bars
        self.early_exit_penalty = early_exit_penalty
        
        self.reset()
    
    def reset(self):
        """Reset for new episode."""
        self.returns = []
        self.trade_count = 0
        self.total_costs = 0.0
        self.total_carry = 0.0
        self.gross_return = 0.0
        
        # Anti-overtrading state
        self.previous_action = 0  # Start FLAT
        self.bars_since_last_trade = 999  # Long time since "last trade"
        self.bars_in_current_position = 0
        
        # Tracking
        self.cooldown_penalties_paid = 0.0
        self.persistence_bonuses_earned = 0.0
        self.early_exit_penalties_paid = 0.0
    
    def compute(
        self,
        action: int,           # 0=FLAT, 1=LONG, 2=SHORT
        market_return: float,  # Market return this bar
        confidence: float = 1.0,
    ) -> float:
        """
        Compute reward with anti-overtrading enhancements.
        """
        reward = 0.0
        position_return = 0.0
        
        # 1. Calculate position return
        if action == 1:  # LONG
            position_return = market_return
        elif action == 2:  # SHORT
            position_return = -market_return
        # FLAT = 0
        
        self.gross_return += position_return
        
        # 2. Apply downside penalty (Sortino-style)
        if position_return < self.target_return:
            adjusted_return = position_return * self.downside_penalty
        else:
            adjusted_return = position_return
        
        reward = adjusted_return * self.scale
        
        # 3. Transaction cost on position change
        position_changed = (action != self.previous_action)
        
        if position_changed:
            cost = self.trading_fee + self.slippage
            reward -= cost * self.scale
            self.trade_count += 1
            self.total_costs += cost
            
            # 3a. COOLDOWN PENALTY - trading too soon after last trade
            if self.bars_since_last_trade < self.cooldown_periods:
                cooldown_cost = self.cooldown_penalty
                reward -= cooldown_cost * self.scale
                self.cooldown_penalties_paid += cooldown_cost
                logger.debug(f"Cooldown penalty: {cooldown_cost:.6f}")
            
            # 3b. EARLY EXIT PENALTY - exiting before min hold time
            if self.bars_in_current_position < self.min_hold_bars:
                early_cost = self.early_exit_penalty
                reward -= early_cost * self.scale
                self.early_exit_penalties_paid += early_cost
                logger.debug(f"Early exit penalty: {early_cost:.6f}")
            
            # Reset position tracking
            self.bars_since_last_trade = 0
            self.bars_in_current_position = 0
        else:
            # 3c. PERSISTENCE BONUS - staying in same action
            bonus = self.persistence_bonus
            reward += bonus * self.scale
            self.persistence_bonuses_earned += bonus
            
            # Increment counters
            self.bars_since_last_trade += 1
            self.bars_in_current_position += 1
        
        # 4. Carry cost for non-FLAT positions
        if action != 0:  # Not FLAT
            carry = self.carry_cost
            reward -= carry * self.scale
            self.total_carry += carry
        
        # Track return
        self.returns.append(position_return)
        self.previous_action = action
        
        return reward
    
    def get_episode_sharpe(self) -> float:
        """Calculate Sharpe for completed episode."""
        if len(self.returns) < 2:
            return 0.0
        returns = np.array(self.returns)
        if returns.std() < 1e-10:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(len(returns))
    
    def get_net_return(self) -> float:
        """Net return after all costs."""
        return self.gross_return - self.total_costs - self.total_carry - \
               self.cooldown_penalties_paid - self.early_exit_penalties_paid + \
               self.persistence_bonuses_earned
    
    def get_stats(self) -> dict:
        """Get detailed stats for logging."""
        return {
            "trade_count": self.trade_count,
            "gross_return": self.gross_return,
            "total_trading_costs": self.total_costs,
            "total_carry_costs": self.total_carry,
            "cooldown_penalties": self.cooldown_penalties_paid,
            "persistence_bonuses": self.persistence_bonuses_earned,
            "early_exit_penalties": self.early_exit_penalties_paid,
            "net_return": self.get_net_return(),
            "sharpe": self.get_episode_sharpe(),
        }


# ==============================================================================
# Factory function for Experiment 10 config
# ==============================================================================

def create_exp10_reward(timeframe: str = "1h") -> SortinoAntiOvertrade:
    """
    Create reward function for Experiment 10.
    
    Args:
        timeframe: "1h" or "5m" to adjust parameters accordingly
    
    Returns:
        Configured SortinoAntiOvertrade instance
    """
    if timeframe == "1h":
        # 1 hour bars - costs scaled down from 5m
        return SortinoAntiOvertrade(
            # Sortino base
            target_return=0.0,
            downside_penalty=2.0,
            scale=100.0,
            
            # Transaction costs
            trading_fee=0.001,      # 0.1%
            slippage=0.0005,        # 0.05%
            
            # Carry cost: Exp9 was 0.00005 for 5m = 1.44%/day
            # For 1h: 0.00005 * (5/60) ≈ 0.000004 per bar
            carry_cost=0.000004,
            
            # Anti-overtrading
            cooldown_periods=4,     # 4 hours
            cooldown_penalty=0.0002,
            persistence_bonus=0.00005,
            min_hold_bars=2,        # 2 hours min hold
            early_exit_penalty=0.0001,
        )
    else:  # 5m
        return SortinoAntiOvertrade(
            target_return=0.0,
            downside_penalty=2.0,
            scale=100.0,
            trading_fee=0.001,
            slippage=0.0005,
            carry_cost=0.00005,      # Same as Exp 9
            cooldown_periods=12,     # 1 hour cooldown (12 * 5min)
            cooldown_penalty=0.0002,
            persistence_bonus=0.00005,
            min_hold_bars=6,         # 30 min min hold (6 * 5min)
            early_exit_penalty=0.0001,
        )


if __name__ == "__main__":
    # Test the reward function
    reward_fn = create_exp10_reward("1h")
    
    # Simulate some trading
    actions = [0, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    returns = [0.001, 0.002, -0.001, 0.003, 0.001, -0.002, 0.001, 
               0.002, 0.001, 0.002, -0.001, 0.001, 0.003, 0.0, 0.0, 0.001]
    
    total_reward = 0
    for i, (a, r) in enumerate(zip(actions, returns)):
        rew = reward_fn.compute(a, r)
        total_reward += rew
        print(f"Step {i}: action={a}, return={r:.4f}, reward={rew:.4f}")
    
    print("\n" + "="*50)
    print("Episode Stats:")
    for k, v in reward_fn.get_stats().items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
