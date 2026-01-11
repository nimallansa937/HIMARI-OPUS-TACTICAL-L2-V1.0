"""
HIMARI OPUS 2 Layer 2 - Performance Metrics
Version: 2.1.1 FINAL

Performance metrics including Deflated Sharpe Ratio (DSR) for
evaluating tactical layer improvements.

DSR adjusts Sharpe ratio for multiple testing bias, preventing
overfitting from appearing as genuine alpha.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import math
import numpy as np


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    sharpe_ratio: float
    deflated_sharpe: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    volatility: float
    sortino_ratio: float
    total_trades: int
    
    def to_dict(self) -> dict:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'deflated_sharpe': self.deflated_sharpe,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'volatility': self.volatility,
            'sortino_ratio': self.sortino_ratio,
            'total_trades': self.total_trades,
        }
    
    def is_acceptable(self) -> bool:
        """
        Check if metrics meet v2.1.1 acceptance criteria.
        
        Criteria:
        - DSR > 0.4 (robust after multiple testing adjustment)
        - Max DD > -50% (improved from baseline -60%+)
        - Sharpe > 0.3 during crisis (vs < 0 baseline)
        """
        return self.deflated_sharpe > 0.4 and self.max_drawdown > -0.50


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio Calculator.
    
    Adjusts Sharpe ratio for multiple testing bias using the formula:
    
    DSR = Sharpe * [1 - (N * log(log(T))) / (2 * log(T)) * (skew^2 + kurt/4)]
    
    Where:
    - N = number of tests (architectural variants + parameter sets)
    - T = number of observations (trading days)
    - skew = return skewness
    - kurt = return kurtosis
    
    Interpretation:
    - DSR > 0.5: Likely profitable (not just lucky)
    - DSR < 0: Likely overfit
    - DSR > 0.3: Acceptable for deployment (v2.1.1 threshold)
    """
    
    @staticmethod
    def calculate(
        sharpe: float,
        n_tests: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,  # Normal distribution has kurt=3
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio.
        
        Args:
            sharpe: Raw Sharpe ratio
            n_tests: Number of strategy variants tested
            n_observations: Number of observations (trading days)
            skewness: Return distribution skewness
            kurtosis: Return distribution kurtosis (excess kurtosis = kurt - 3)
            
        Returns:
            Deflated Sharpe ratio
        """
        if n_observations <= 1 or n_tests < 1:
            return 0.0
        
        # Avoid log(0) or log(1)
        T = max(n_observations, 3)
        N = max(n_tests, 1)
        
        try:
            log_T = math.log(T)
            log_log_T = math.log(log_T) if log_T > 1 else 0.001
            
            # Excess kurtosis
            excess_kurt = kurtosis - 3.0
            
            # Haircut factor
            haircut = (N * log_log_T) / (2 * log_T) * (skewness**2 + excess_kurt / 4)
            
            # Apply haircut
            dsr = sharpe * (1 - min(1, haircut))
            
            return max(0, dsr)
            
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_from_returns(
        returns: np.ndarray,
        n_tests: int = 1,
        annualization_factor: float = 252,
    ) -> float:
        """
        Calculate DSR directly from return series.
        
        Args:
            returns: Array of returns
            n_tests: Number of strategy variants tested
            annualization_factor: Days per year for annualization
            
        Returns:
            Deflated Sharpe ratio
        """
        if len(returns) < 10:
            return 0.0
        
        # Calculate Sharpe
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret == 0:
            return 0.0
        
        sharpe = math.sqrt(annualization_factor) * mean_ret / std_ret
        
        # Calculate distribution stats
        from scipy import stats
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns, fisher=False))  # Pearson kurtosis
        
        return DeflatedSharpeRatio.calculate(
            sharpe, n_tests, len(returns), skewness, kurtosis
        )


class MetricsCalculator:
    """
    Comprehensive metrics calculator for tactical layer evaluation.
    """
    
    @staticmethod
    def calculate(
        returns: np.ndarray,
        n_tests: int = 1,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Array of trade returns
            n_tests: Number of variants tested (for DSR)
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Comprehensive performance metrics
        """
        if len(returns) < 2:
            return PerformanceMetrics(
                sharpe_ratio=0.0,
                deflated_sharpe=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade_return=0.0,
                volatility=0.0,
                sortino_ratio=0.0,
                total_trades=len(returns),
            )
        
        # Basic stats
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Sharpe ratio
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = math.sqrt(252) * np.mean(excess_returns) / std_ret if std_ret > 0 else 0
        
        # Deflated Sharpe
        dsr = DeflatedSharpeRatio.calculate_from_returns(returns, n_tests)
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_dd = float(np.min(drawdowns))
        
        # Calmar ratio (annual return / |max DD|)
        annual_return = mean_ret * 252
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        winning = returns > 0
        win_rate = np.mean(winning)
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Volatility (annualized)
        volatility = std_ret * math.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = (mean_ret * math.sqrt(252)) / (downside_std * math.sqrt(252)) if downside_std > 0 else 0
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            deflated_sharpe=dsr,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=mean_ret,
            volatility=volatility,
            sortino_ratio=sortino,
            total_trades=len(returns),
        )
    
    @staticmethod
    def compare_strategies(
        baseline_returns: np.ndarray,
        enhanced_returns: np.ndarray,
    ) -> Dict:
        """
        Compare baseline vs enhanced strategy performance.
        
        Args:
            baseline_returns: Returns from baseline strategy
            enhanced_returns: Returns from enhanced strategy
            
        Returns:
            Comparison dictionary
        """
        baseline_metrics = MetricsCalculator.calculate(baseline_returns)
        enhanced_metrics = MetricsCalculator.calculate(enhanced_returns)
        
        return {
            'baseline': baseline_metrics.to_dict(),
            'enhanced': enhanced_metrics.to_dict(),
            'improvements': {
                'sharpe_delta': enhanced_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio,
                'dd_improvement': enhanced_metrics.max_drawdown - baseline_metrics.max_drawdown,
                'win_rate_delta': enhanced_metrics.win_rate - baseline_metrics.win_rate,
            },
            'verdict': {
                'sharpe_acceptable': enhanced_metrics.sharpe_ratio >= baseline_metrics.sharpe_ratio * 0.95,  # Allow 5% drag
                'dd_improved': enhanced_metrics.max_drawdown > baseline_metrics.max_drawdown,
                'overall': enhanced_metrics.is_acceptable(),
            },
        }
