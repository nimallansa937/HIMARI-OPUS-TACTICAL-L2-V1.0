"""
HIMARI Layer 2 - Part H: RSS Risk Management
Complete risk management pipeline with adaptive position sizing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# H1: Kelly Criterion Position Sizing
# ============================================================================

@dataclass
class KellyConfig:
    base_fraction: float = 0.25  # Kelly fraction (quarter Kelly for safety)
    max_position: float = 0.20   # Max position as fraction of capital
    min_position: float = 0.01   # Minimum position size
    win_rate_period: int = 100   # Lookback for win rate

class KellySizer:
    """Kelly Criterion position sizing with fractional Kelly for safety."""
    def __init__(self, config=None):
        self.config = config or KellyConfig()
        self.outcomes = deque(maxlen=self.config.win_rate_period)
        self.returns = deque(maxlen=self.config.win_rate_period)
        
    def update(self, pnl: float):
        self.outcomes.append(1 if pnl > 0 else 0)
        self.returns.append(pnl)
        
    def compute_position(self, confidence: float) -> float:
        if len(self.outcomes) < 20:
            return self.config.min_position
            
        win_rate = np.mean(self.outcomes)
        avg_win = np.mean([r for r in self.returns if r > 0]) if any(r > 0 for r in self.returns) else 0.01
        avg_loss = abs(np.mean([r for r in self.returns if r < 0])) if any(r < 0 for r in self.returns) else 0.01
        
        # Kelly formula: f* = (p*b - q) / b where b = avg_win/avg_loss
        b = avg_win / (avg_loss + 1e-8)
        q = 1 - win_rate
        kelly = (win_rate * b - q) / (b + 1e-8)
        
        # Apply fractional Kelly and confidence scaling
        position = kelly * self.config.base_fraction * confidence
        return np.clip(position, self.config.min_position, self.config.max_position)


# ============================================================================
# H2: Volatility Targeting
# ============================================================================

@dataclass
class VolTargetConfig:
    target_vol: float = 0.15  # 15% annualized volatility target
    vol_lookback: int = 20    # Rolling volatility window
    max_leverage: float = 3.0

class VolatilityTargeting:
    """Position sizing to target constant portfolio volatility."""
    def __init__(self, config=None):
        self.config = config or VolTargetConfig()
        self.returns = deque(maxlen=self.config.vol_lookback)
        
    def update(self, daily_return: float):
        self.returns.append(daily_return)
        
    def compute_position(self) -> float:
        if len(self.returns) < 5:
            return 1.0
            
        realized_vol = np.std(self.returns) * np.sqrt(252)  # Annualize
        if realized_vol < 0.01:
            return 1.0
            
        leverage = self.config.target_vol / realized_vol
        return np.clip(leverage, 0.1, self.config.max_leverage)


# ============================================================================
# H3: Drawdown Control
# ============================================================================

@dataclass
class DrawdownConfig:
    max_drawdown: float = 0.15    # Max allowed drawdown
    critical_drawdown: float = 0.20
    recovery_threshold: float = 0.05

class DrawdownController:
    """Reduces exposure during drawdowns."""
    def __init__(self, config=None):
        self.config = config or DrawdownConfig()
        self.peak_equity = 1.0
        self.current_equity = 1.0
        
    def update(self, equity: float) -> float:
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        
        drawdown = (self.peak_equity - equity) / self.peak_equity
        
        if drawdown >= self.config.critical_drawdown:
            return 0.0  # Stop trading
        elif drawdown >= self.config.max_drawdown:
            return 0.25  # Quarter position
        elif drawdown >= self.config.max_drawdown * 0.5:
            return 0.5  # Half position
        return 1.0
    
    @property
    def current_drawdown(self):
        return (self.peak_equity - self.current_equity) / self.peak_equity


# ============================================================================
# H4: Correlation-Based Risk Adjustment
# ============================================================================

class CorrelationRiskAdjuster:
    """Adjusts position based on correlation regime."""
    def __init__(self, window: int = 30):
        self.window = window
        self.asset_returns = {}
        
    def update(self, asset: str, ret: float):
        if asset not in self.asset_returns:
            self.asset_returns[asset] = deque(maxlen=self.window)
        self.asset_returns[asset].append(ret)
        
    def get_diversification_factor(self) -> float:
        if len(self.asset_returns) < 2:
            return 1.0
        
        # Compute average pairwise correlation
        assets = list(self.asset_returns.keys())
        correlations = []
        for i, a1 in enumerate(assets):
            for a2 in assets[i+1:]:
                r1, r2 = list(self.asset_returns[a1]), list(self.asset_returns[a2])
                min_len = min(len(r1), len(r2))
                if min_len >= 10:
                    corr = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if not correlations:
            return 1.0
        avg_corr = np.mean(correlations)
        # Higher correlation = lower diversification = scale down
        return 1.0 - 0.5 * avg_corr


# ============================================================================
# H5: Regime-Adaptive Risk
# ============================================================================

class RegimeRiskScaler:
    """Scales risk based on detected regime."""
    def __init__(self):
        self.regime_multipliers = {
            0: 1.2,   # Trending up - full risk
            1: 1.0,   # Trending down - normal
            2: 0.8,   # Ranging - reduced
            3: 0.4,   # High volatility - low risk
            4: 0.1,   # Crisis - minimal
        }
        
    def get_multiplier(self, regime: int) -> float:
        return self.regime_multipliers.get(regime, 0.5)


# ============================================================================
# H6: Stop Loss Manager
# ============================================================================

@dataclass
class StopLossConfig:
    atr_multiplier: float = 2.0
    max_loss_pct: float = 0.02
    trailing_activation: float = 0.01

class StopLossManager:
    """ATR-based and trailing stop loss management."""
    def __init__(self, config=None):
        self.config = config or StopLossConfig()
        self.entry_price = None
        self.stop_price = None
        self.highest_since_entry = None
        
    def set_entry(self, price: float, atr: float):
        self.entry_price = price
        self.stop_price = price - self.config.atr_multiplier * atr
        self.highest_since_entry = price
        
    def update(self, price: float, atr: float) -> bool:
        """Returns True if stop triggered."""
        if self.entry_price is None:
            return False
            
        # Update trailing stop
        if price > self.highest_since_entry:
            self.highest_since_entry = price
            if (price - self.entry_price) / self.entry_price > self.config.trailing_activation:
                self.stop_price = max(self.stop_price, price - self.config.atr_multiplier * atr)
        
        # Check stops
        if price <= self.stop_price:
            return True
        if (self.entry_price - price) / self.entry_price > self.config.max_loss_pct:
            return True
        return False
    
    def clear(self):
        self.entry_price = None
        self.stop_price = None
        self.highest_since_entry = None


# ============================================================================
# Complete Risk Management Pipeline
# ============================================================================

@dataclass
class RiskConfig:
    target_vol: float = 0.15
    max_drawdown: float = 0.15
    kelly_fraction: float = 0.25
    max_position: float = 0.20

class RSSRiskManager:
    """Complete Risk Management Pipeline."""
    
    def __init__(self, config=None):
        self.config = config or RiskConfig()
        self.kelly = KellySizer(KellyConfig(base_fraction=self.config.kelly_fraction))
        self.vol_target = VolatilityTargeting(VolTargetConfig(target_vol=self.config.target_vol))
        self.drawdown = DrawdownController(DrawdownConfig(max_drawdown=self.config.max_drawdown))
        self.correlation = CorrelationRiskAdjuster()
        self.regime_scaler = RegimeRiskScaler()
        self.stop_loss = StopLossManager()
        
    def compute_position_size(self, confidence: float, regime: int, 
                             equity: float) -> float:
        """Compute final position size with all risk adjustments."""
        # Base position from Kelly
        kelly_pos = self.kelly.compute_position(confidence)
        
        # Vol targeting
        vol_mult = self.vol_target.compute_position()
        
        # Drawdown scaling
        dd_mult = self.drawdown.update(equity)
        
        # Regime scaling
        regime_mult = self.regime_scaler.get_multiplier(regime)
        
        # Correlation adjustment
        corr_mult = self.correlation.get_diversification_factor()
        
        final_pos = kelly_pos * vol_mult * dd_mult * regime_mult * corr_mult
        return np.clip(final_pos, 0.0, self.config.max_position)
    
    def update_pnl(self, pnl: float, daily_return: float):
        self.kelly.update(pnl)
        self.vol_target.update(daily_return)
    
    def get_diagnostics(self) -> Dict:
        return {
            'drawdown': self.drawdown.current_drawdown,
            'vol_leverage': self.vol_target.compute_position(),
            'diversification': self.correlation.get_diversification_factor()
        }
