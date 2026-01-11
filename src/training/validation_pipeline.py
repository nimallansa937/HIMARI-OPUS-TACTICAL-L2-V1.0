"""
HIMARI Layer 2 - Part L: Validation Framework
Comprehensive model validation and testing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# L1: Backtesting Engine
# ============================================================================

@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_pct: float = 0.0005
    max_position_pct: float = 0.20

class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(self, config=None):
        self.config = config or BacktestConfig()
        self.capital = self.config.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = [self.capital]
        
    def execute_signal(self, price: float, signal: int, size: float = 1.0):
        """Execute a trading signal."""
        if signal == 0:
            return
            
        # Apply slippage
        exec_price = price * (1 + self.config.slippage_pct * signal)
        
        # Calculate trade value
        max_value = self.capital * self.config.max_position_pct
        trade_value = min(max_value * size, self.capital * 0.95)
        
        # Commission
        commission = trade_value * self.config.commission_rate
        
        # Update position
        if signal == 1:  # Buy
            self.position += trade_value / exec_price
            self.capital -= trade_value + commission
        elif signal == -1:  # Sell
            if self.position > 0:
                proceeds = self.position * exec_price
                self.capital += proceeds - commission
                self.position = 0
                
        self.trades.append({
            'price': exec_price,
            'signal': signal,
            'capital': self.capital
        })
        
    def update_equity(self, price: float):
        """Update equity curve with current price."""
        equity = self.capital + self.position * price
        self.equity_curve.append(equity)
        
    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] - equity[0]) / equity[0]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
            'final_equity': equity[-1]
        }


# ============================================================================
# L2: Walk-Forward Analysis
# ============================================================================

class WalkForwardAnalyzer:
    """Walk-forward analysis for robust validation."""
    
    def __init__(self, train_ratio: float = 0.7, n_windows: int = 5):
        self.train_ratio = train_ratio
        self.n_windows = n_windows
        self.window_results = []
        
    def analyze(self, data: np.ndarray, model_fn, train_fn, test_fn) -> Dict:
        """Perform walk-forward analysis."""
        n = len(data)
        window_size = n // self.n_windows
        
        for i in range(self.n_windows):
            start = i * window_size
            end = start + window_size
            
            train_end = start + int(window_size * self.train_ratio)
            train_data = data[start:train_end]
            test_data = data[train_end:end]
            
            # Train and test
            model = train_fn(train_data)
            metrics = test_fn(model, test_data)
            
            self.window_results.append({
                'window': i,
                'train_size': len(train_data),
                'test_size': len(test_data),
                **metrics
            })
            
        return {
            'windows': self.window_results,
            'avg_sharpe': np.mean([r.get('sharpe', 0) for r in self.window_results]),
            'stability': np.std([r.get('sharpe', 0) for r in self.window_results])
        }


# ============================================================================
# L3: Statistical Tests
# ============================================================================

class StatisticalValidator:
    """Statistical tests for strategy validation."""
    
    def t_test(self, returns: np.ndarray) -> Tuple[float, float]:
        """Test if mean return is significantly different from zero."""
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        t_stat = mean / (std / np.sqrt(n))
        # Simplified p-value (would use scipy.stats in production)
        p_value = min(1.0, 2 * np.exp(-0.5 * t_stat ** 2))
        return t_stat, p_value
    
    def sharpe_test(self, returns: np.ndarray, benchmark: float = 0) -> Dict:
        """Test Sharpe ratio significance."""
        sharpe = (np.mean(returns) - benchmark) / (np.std(returns) + 1e-8)
        annualized = sharpe * np.sqrt(252)
        # Standard error of Sharpe
        se = np.sqrt((1 + 0.5 * sharpe**2) / len(returns))
        return {
            'sharpe': annualized,
            'std_error': se * np.sqrt(252),
            'significant': abs(annualized) > 2 * se * np.sqrt(252)
        }
    
    def drawdown_test(self, equity: np.ndarray, threshold: float = 0.2) -> Dict:
        """Analyze drawdown characteristics."""
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        
        # Find drawdown periods
        in_dd = drawdown > 0.01
        dd_periods = []
        start = None
        for i, is_dd in enumerate(in_dd):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                dd_periods.append(i - start)
                start = None
                
        return {
            'max_drawdown': drawdown.max(),
            'avg_dd_duration': np.mean(dd_periods) if dd_periods else 0,
            'exceeds_threshold': drawdown.max() > threshold
        }


# ============================================================================
# L4: Cross-Validation
# ============================================================================

class TimeSeriesCrossValidator:
    """Time-aware cross-validation for financial data."""
    
    def __init__(self, n_splits: int = 5, gap: int = 20):
        self.n_splits = n_splits
        self.gap = gap
        
    def split(self, n_samples: int):
        """Generate train/test split indices."""
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            test_start = train_end + self.gap
            test_end = min(test_start + fold_size, n_samples)
            
            yield list(range(train_end)), list(range(test_start, test_end))


# ============================================================================
# L5: Performance Attribution
# ============================================================================

class PerformanceAttributor:
    """Attribute performance to different factors."""
    
    def __init__(self):
        self.factor_returns = {}
        
    def add_factor(self, name: str, returns: np.ndarray):
        self.factor_returns[name] = returns
        
    def attribute(self, strategy_returns: np.ndarray) -> Dict:
        """Decompose strategy returns by factor."""
        if not self.factor_returns:
            return {'strategy': np.mean(strategy_returns)}
            
        # Simple attribution via correlation
        attributions = {}
        remaining = strategy_returns.copy()
        
        for name, factor_rets in self.factor_returns.items():
            min_len = min(len(remaining), len(factor_rets))
            corr = np.corrcoef(remaining[:min_len], factor_rets[:min_len])[0, 1]
            beta = corr * np.std(remaining[:min_len]) / (np.std(factor_rets[:min_len]) + 1e-8)
            factor_contrib = beta * np.mean(factor_rets[:min_len])
            attributions[name] = factor_contrib
            
        attributions['alpha'] = np.mean(strategy_returns) - sum(attributions.values())
        return attributions


# ============================================================================
# L6: Robustness Checks
# ============================================================================

class RobustnessChecker:
    """Check strategy robustness under various conditions."""
    
    def parameter_sensitivity(self, model_fn, param_ranges: Dict, 
                             test_fn) -> Dict:
        """Test sensitivity to parameter changes."""
        results = {}
        
        for param_name, values in param_ranges.items():
            param_results = []
            for value in values:
                metrics = test_fn({param_name: value})
                param_results.append({'value': value, **metrics})
            results[param_name] = param_results
            
        return results
    
    def regime_breakdown(self, returns: np.ndarray, regimes: np.ndarray) -> Dict:
        """Analyze performance by regime."""
        regime_results = {}
        
        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_rets = returns[mask]
            
            if len(regime_rets) > 10:
                regime_results[int(regime)] = {
                    'mean_return': np.mean(regime_rets),
                    'sharpe': np.mean(regime_rets) / (np.std(regime_rets) + 1e-8) * np.sqrt(252),
                    'count': len(regime_rets),
                    'pct_positive': np.mean(regime_rets > 0)
                }
                
        return regime_results


# ============================================================================
# Complete Validation Pipeline
# ============================================================================

@dataclass
class ValidationConfig:
    train_ratio: float = 0.7
    n_windows: int = 5
    n_splits: int = 5

class ValidationPipeline:
    """Complete validation framework."""
    
    def __init__(self, config=None):
        self.config = config or ValidationConfig()
        self.backtest = BacktestEngine()
        self.walk_forward = WalkForwardAnalyzer(
            train_ratio=self.config.train_ratio,
            n_windows=self.config.n_windows
        )
        self.stats = StatisticalValidator()
        self.cv = TimeSeriesCrossValidator(n_splits=self.config.n_splits)
        self.attributor = PerformanceAttributor()
        self.robustness = RobustnessChecker()
        
    def full_validation(self, strategy_returns: np.ndarray,
                       equity_curve: np.ndarray) -> Dict:
        """Run complete validation suite."""
        results = {}
        
        # Statistical tests
        t_stat, p_value = self.stats.t_test(strategy_returns)
        results['t_test'] = {'t_stat': t_stat, 'p_value': p_value}
        results['sharpe_test'] = self.stats.sharpe_test(strategy_returns)
        results['drawdown_test'] = self.stats.drawdown_test(equity_curve)
        
        # Performance metrics
        results['sharpe'] = results['sharpe_test']['sharpe']
        results['significant'] = p_value < 0.05
        
        return results
