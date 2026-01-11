"""
HIMARI Layer 2 - Synthetic Crash Generator for Pre-training
Based on Layer 3 research: Pre-train on 70% synthetic, 30% real data.

Generates stress scenarios using:
- Merton Jump-Diffusion (MJD) for flash crashes
- GARCH(1,1) with Student-t for volatility clustering
- Maps to 44-feature format for Transformer-A2C
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CrashScenarioConfig:
    """Configuration for synthetic crash generation."""
    
    # General settings
    n_crash_scenarios: int = 500
    n_baseline_scenarios: int = 1000
    steps_per_scenario: int = 1000
    feature_dim: int = 44
    
    # Merton Jump-Diffusion parameters
    mu: float = 0.0001  # Drift (small positive for crypto)
    sigma: float = 0.02  # Volatility
    lambda_jump: float = 0.05  # Jump intensity (5% of steps)
    mu_jump: float = -0.15  # Mean jump size (negative for crashes)
    sigma_jump: float = 0.10  # Jump size volatility
    
    # GARCH(1,1) parameters
    omega: float = 0.00001  # Long-run variance
    alpha: float = 0.15  # ARCH term
    beta: float = 0.80  # GARCH term
    df: int = 5  # Degrees of freedom for Student-t
    
    # Crash amplification
    crash_amplification: float = 2.0  # Amplify crashes for stress testing
    min_crash_magnitude: float = -0.10  # Min 10% crash per scenario


class SyntheticCrashGenerator:
    """
    Generates synthetic crash scenarios for pre-training.
    
    Based on research:
    - Agent learns that high volatility → reduce position
    - Exposes model to black swan events not in historical data
    - Reduces overfitting risk by 60-70%
    """
    
    def __init__(self, config: CrashScenarioConfig = None):
        self.config = config or CrashScenarioConfig()
        np.random.seed(42)  # For reproducibility
        
    def generate_mjd_returns(
        self,
        n_steps: int,
        amplify_crashes: bool = True,
    ) -> np.ndarray:
        """
        Generate returns using Merton Jump-Diffusion model.
        
        dS/S = (μ - λk)dt + σdW + dJ
        
        Where:
        - μ: drift
        - σ: diffusion volatility  
        - λ: jump intensity
        - J: compound Poisson process with lognormal jumps
        """
        cfg = self.config
        
        # Diffusion component (geometric Brownian motion)
        diffusion = cfg.sigma * np.random.randn(n_steps)
        
        # Jump component (Poisson process with lognormal jumps)
        jump_times = np.random.poisson(cfg.lambda_jump, n_steps)
        jump_sizes = np.zeros(n_steps)
        
        for i in range(n_steps):
            if jump_times[i] > 0:
                # Lognormal jump sizes (biased negative for crashes)
                jump = np.sum(np.random.normal(cfg.mu_jump, cfg.sigma_jump, jump_times[i]))
                jump_sizes[i] = jump
                
        # Combine
        returns = cfg.mu + diffusion + jump_sizes
        
        # Amplify crashes if requested
        if amplify_crashes:
            crash_mask = returns < -0.02  # Identify crash days
            returns[crash_mask] *= cfg.crash_amplification
            
        return returns
    
    def generate_garch_returns(
        self,
        n_steps: int,
    ) -> np.ndarray:
        """
        Generate returns using GARCH(1,1) with Student-t innovations.
        
        σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
        r(t) = σ(t) · z(t), z ~ Student-t(df)
        
        Captures volatility clustering typical in crypto markets.
        """
        cfg = self.config
        
        # Initialize
        returns = np.zeros(n_steps)
        sigma_sq = np.zeros(n_steps)
        sigma_sq[0] = cfg.omega / (1 - cfg.alpha - cfg.beta)  # Unconditional variance
        
        # Student-t innovations
        innovations = np.random.standard_t(cfg.df, n_steps)
        innovations = innovations / np.sqrt(cfg.df / (cfg.df - 2))  # Standardize
        
        # Generate GARCH process
        for t in range(1, n_steps):
            sigma_sq[t] = (
                cfg.omega + 
                cfg.alpha * returns[t-1]**2 + 
                cfg.beta * sigma_sq[t-1]
            )
            returns[t] = np.sqrt(sigma_sq[t]) * innovations[t]
            
        return returns
    
    def generate_crash_scenario(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single crash scenario with 44 features.
        
        Returns:
            features: [n_steps, 44] feature array
            prices: [n_steps] price array
        """
        cfg = self.config
        n_steps = cfg.steps_per_scenario
        
        # Generate MJD returns for price
        returns = self.generate_mjd_returns(n_steps, amplify_crashes=True)
        
        # Ensure minimum crash magnitude
        if returns.sum() > cfg.min_crash_magnitude:
            # Add a flash crash
            crash_idx = np.random.randint(n_steps // 4, 3 * n_steps // 4)
            returns[crash_idx:crash_idx+5] = np.random.uniform(-0.10, -0.03, 5)
        
        # Generate price path
        prices = 50000 * np.cumprod(1 + returns)  # Starting BTC price ~50k
        
        # Generate 44 features
        features = self._generate_features(returns, prices)
        
        return features, prices
    
    def generate_baseline_scenario(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a baseline (non-crash) scenario using GARCH.
        
        Returns:
            features: [n_steps, 44] feature array
            prices: [n_steps] price array
        """
        cfg = self.config
        n_steps = cfg.steps_per_scenario
        
        # Generate GARCH returns (no crash amplification)
        returns = self.generate_garch_returns(n_steps)
        
        # Generate price path
        prices = 50000 * np.cumprod(1 + returns)
        
        # Generate 44 features
        features = self._generate_features(returns, prices)
        
        return features, prices
    
    def _generate_features(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Generate 44 features from returns and prices.
        
        Feature mapping based on Layer 2 Trading Guide:
        0-9: Price-based features (returns, MA ratios)
        10-19: Volatility features (realized vol, ATR)
        20-29: Technical indicators (RSI, MACD, Bollinger)
        30-39: Volume/orderflow features (synthetic)
        40-43: Regime/position features
        """
        n_steps = len(returns)
        features = np.zeros((n_steps, self.config.feature_dim), dtype=np.float32)
        
        # 0-4: Return features
        features[:, 0] = returns  # 1-step return
        features[:, 1] = np.roll(returns, 1)  # Lagged return
        features[:, 2] = _rolling_mean(returns, 5)  # 5-step MA
        features[:, 3] = _rolling_mean(returns, 20)  # 20-step MA
        features[:, 4] = _rolling_mean(returns, 50)  # 50-step MA
        
        # 5-9: Price ratios
        ma_5 = _rolling_mean(prices, 5)
        ma_20 = _rolling_mean(prices, 20)
        features[:, 5] = prices / ma_5 - 1  # Price/MA5 ratio
        features[:, 6] = prices / ma_20 - 1  # Price/MA20 ratio
        features[:, 7] = ma_5 / ma_20 - 1  # MA5/MA20 crossover
        features[:, 8] = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)  # Normalized price
        features[:, 9] = np.log(prices / np.roll(prices, 1) + 1e-8)  # Log return
        
        # 10-19: Volatility features
        features[:, 10] = _rolling_std(returns, 5)  # 5-step vol
        features[:, 11] = _rolling_std(returns, 20)  # 20-step vol
        features[:, 12] = _rolling_std(returns, 50)  # 50-step vol
        features[:, 13] = features[:, 10] / (features[:, 11] + 1e-8)  # Vol ratio
        high = prices * (1 + np.abs(np.random.randn(n_steps) * 0.01))
        low = prices * (1 - np.abs(np.random.randn(n_steps) * 0.01))
        features[:, 14] = _atr(high, low, prices, 14)  # ATR
        features[:, 15] = returns / (features[:, 10] + 1e-8)  # Return/Vol (Sharpe-like)
        features[:, 16] = _rolling_max(returns, 20) - _rolling_min(returns, 20)  # Range
        features[:, 17] = _ewma(features[:, 10], 0.1)  # EMA of volatility
        features[:, 18] = np.where(features[:, 10] > features[:, 11], 1, 0)  # Vol regime
        features[:, 19] = _rolling_skew(returns, 20)  # Skewness
        
        # 20-29: Technical indicators
        features[:, 20] = _rsi(returns, 14)  # RSI
        macd, signal = _macd(prices)
        features[:, 21] = macd
        features[:, 22] = signal
        features[:, 23] = macd - signal  # MACD histogram
        bb_upper, bb_lower = _bollinger_bands(prices, 20)
        features[:, 24] = (prices - bb_lower) / (bb_upper - bb_lower + 1e-8)  # Bollinger %B
        features[:, 25] = bb_upper - bb_lower  # Bollinger width
        features[:, 26] = _stochastic(prices, 14)  # Stochastic
        features[:, 27] = _williams_r(prices, 14)  # Williams %R
        features[:, 28] = _cci(high, low, prices, 20)  # CCI
        features[:, 29] = _momentum(prices, 10)  # Momentum
        
        # 30-39: Volume/orderflow (synthetic)
        synth_volume = np.abs(returns) * 1e9 + np.random.exponential(1e8, n_steps)
        features[:, 30] = synth_volume / _rolling_mean(synth_volume, 20)  # Vol ratio
        features[:, 31] = np.random.randn(n_steps) * 0.01  # Order imbalance (synthetic)
        features[:, 32] = _ewma(features[:, 30], 0.2)  # EMA volume
        features[:, 33] = np.random.uniform(-0.01, 0.01, n_steps)  # Funding rate (synthetic)
        features[:, 34] = np.random.randn(n_steps) * 0.1  # OI delta (synthetic)
        features[:, 35] = np.where(synth_volume > _rolling_mean(synth_volume, 5), 1, 0)  # Volume spike
        features[:, 36] = _rolling_mean(features[:, 31], 10)  # Avg imbalance
        features[:, 37] = returns * synth_volume / 1e10  # Dollar volume
        features[:, 38] = np.random.choice([0, 1], n_steps, p=[0.7, 0.3])  # Large trade flag
        features[:, 39] = np.random.randn(n_steps) * 0.05  # Spread estimate
        
        # 40-43: Regime/position features
        features[:, 40] = _rolling_mean(features[:, 18], 20)  # Regime probability
        features[:, 41] = np.zeros(n_steps)  # Current position (placeholder)
        features[:, 42] = np.zeros(n_steps)  # Current PnL (placeholder)
        features[:, 43] = np.random.uniform(0.5, 1.0, n_steps)  # Confidence score
        
        # Handle NaNs
        features = np.nan_to_num(features, 0.0)
        
        return features
    
    def generate_all_scenarios(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate all crash and baseline scenarios.
        
        Returns:
            all_features: [n_scenarios, n_steps, 44]
            all_prices: [n_scenarios, n_steps]
        """
        cfg = self.config
        
        logger.info(f"Generating {cfg.n_crash_scenarios} crash scenarios...")
        crash_features = []
        crash_prices = []
        for i in range(cfg.n_crash_scenarios):
            if (i + 1) % 100 == 0:
                logger.info(f"  Crash scenario {i + 1}/{cfg.n_crash_scenarios}")
            f, p = self.generate_crash_scenario()
            crash_features.append(f)
            crash_prices.append(p)
            
        logger.info(f"Generating {cfg.n_baseline_scenarios} baseline scenarios...")
        baseline_features = []
        baseline_prices = []
        for i in range(cfg.n_baseline_scenarios):
            if (i + 1) % 200 == 0:
                logger.info(f"  Baseline scenario {i + 1}/{cfg.n_baseline_scenarios}")
            f, p = self.generate_baseline_scenario()
            baseline_features.append(f)
            baseline_prices.append(p)
            
        # Combine (crash scenarios first for emphasis)
        all_features = np.array(crash_features + baseline_features)
        all_prices = np.array(crash_prices + baseline_prices)
        
        logger.info(f"Generated {len(all_features)} total scenarios")
        logger.info(f"  Shape: {all_features.shape}")
        
        return all_features, all_prices
    
    def save_scenarios(self, output_path: str):
        """Generate and save scenarios to pickle file."""
        import pickle
        
        features, prices = self.generate_all_scenarios()
        
        data = {
            'features': features,
            'prices': prices,
            'config': self.config.__dict__,
            'n_crash': self.config.n_crash_scenarios,
            'n_baseline': self.config.n_baseline_scenarios,
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved scenarios to {output_path}")
        
    @staticmethod
    def load_scenarios(input_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load scenarios from pickle file."""
        import pickle
        
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            
        return data['features'], data['prices']


# ============================================================================
# Helper functions for feature generation
# ============================================================================

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with edge handling."""
    result = np.convolve(x, np.ones(window)/window, mode='same')
    return result

def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.std(x[start:i+1])
    return result

def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum."""
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.max(x[start:i+1])
    return result

def _rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum."""
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.min(x[start:i+1])
    return result

def _rolling_skew(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling skewness."""
    from scipy.stats import skew
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        if i - start >= 2:
            result[i] = skew(x[start:i+1])
    return result

def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average."""
    result = np.zeros_like(x)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    return result

def _rsi(returns: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    
    avg_gain = _ewma(gains, 1/period)
    avg_loss = _ewma(losses, 1/period)
    
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    return rsi / 100  # Normalize to [0, 1]

def _macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """MACD indicator."""
    ema_fast = _ewma(prices, 2/(fast+1))
    ema_slow = _ewma(prices, 2/(slow+1))
    macd_line = (ema_fast - ema_slow) / prices
    signal_line = _ewma(macd_line, 2/(signal+1))
    return macd_line, signal_line

def _bollinger_bands(prices: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Bollinger Bands."""
    ma = _rolling_mean(prices, period)
    std = _rolling_std(prices, period)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, lower

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr = _ewma(tr, 1/period)
    return atr / close  # Normalize

def _stochastic(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Stochastic oscillator."""
    highest = _rolling_max(prices, period)
    lowest = _rolling_min(prices, period)
    k = (prices - lowest) / (highest - lowest + 1e-8)
    return k

def _williams_r(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Williams %R."""
    return _stochastic(prices, period) - 0.5  # Center around 0

def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    tp_ma = _rolling_mean(tp, period)
    tp_std = _rolling_std(tp, period)
    cci = (tp - tp_ma) / (0.015 * tp_std + 1e-8)
    return cci / 100  # Scale

def _momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Momentum indicator."""
    return prices / np.roll(prices, period) - 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic crash scenarios")
    parser.add_argument("--output", type=str, default="./data/synthetic_crashes.pkl")
    parser.add_argument("--n_crashes", type=int, default=500)
    parser.add_argument("--n_baseline", type=int, default=1000)
    
    args = parser.parse_args()
    
    config = CrashScenarioConfig(
        n_crash_scenarios=args.n_crashes,
        n_baseline_scenarios=args.n_baseline,
    )
    
    generator = SyntheticCrashGenerator(config)
    generator.save_scenarios(args.output)
