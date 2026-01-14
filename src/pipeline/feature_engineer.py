"""
Feature Engineering for BTC Trading

Generates 44 technical features from raw OHLCV data.
These features are designed to capture:
    - Price momentum and trends
    - Volatility regimes
    - Volume patterns
    - Mean reversion signals

Author: HIMARI Development Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Momentum windows
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Volatility windows
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0

    # Volume windows
    vwap_period: int = 20
    obv_smooth: int = 10

    # Returns windows
    returns_windows: Tuple[int, ...] = (1, 4, 12, 24, 48, 168)  # 1h to 1w
    vol_windows: Tuple[int, ...] = (12, 24, 48, 168)


class FeatureEngineer:
    """
    Generates 44 features from OHLCV data.

    Feature groups:
        - Returns (6): Multi-horizon returns
        - Momentum (8): RSI, MACD, Stochastic, ROC
        - Volatility (8): ATR, Bollinger, Historical vol
        - Volume (6): OBV, VWAP deviation, Volume ratios
        - Price patterns (8): Support/resistance, Candle patterns
        - Statistical (8): Skewness, Kurtosis, Autocorr

    Total: 44 features
    """

    FEATURE_DIM = 44

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names = self._create_feature_names()

    def _create_feature_names(self) -> list:
        """Create feature name list."""
        names = []

        # Returns (0-5)
        for w in self.config.returns_windows:
            names.append(f'returns_{w}h')

        # Momentum (6-13)
        names.extend([
            'rsi_14', 'rsi_norm',
            'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d',
            'roc_12'
        ])

        # Volatility (14-21)
        names.extend([
            'atr_14', 'atr_norm',
            'bb_upper_dist', 'bb_lower_dist', 'bb_width',
            'vol_12h', 'vol_24h', 'vol_ratio'
        ])

        # Volume (22-27)
        names.extend([
            'obv_norm', 'obv_slope',
            'vwap_dev', 'volume_ma_ratio',
            'volume_trend', 'volume_zscore'
        ])

        # Price patterns (28-35)
        names.extend([
            'high_low_ratio', 'close_position',
            'gap_up', 'gap_down',
            'upper_shadow', 'lower_shadow',
            'body_size', 'range_position'
        ])

        # Statistical (36-43)
        names.extend([
            'skewness_24h', 'kurtosis_24h',
            'autocorr_1', 'autocorr_4',
            'hurst_approx', 'mean_rev_score',
            'zscore_24h', 'zscore_168h'
        ])

        return names

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 44 features from OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Index should be DatetimeIndex

        Returns:
            DataFrame with 44 feature columns
        """
        features = pd.DataFrame(index=df.index)

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df['volume']

        # === RETURNS (6 features) ===
        for w in self.config.returns_windows:
            features[f'returns_{w}h'] = close.pct_change(w)

        # === MOMENTUM (8 features) ===
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        features['rsi_norm'] = (features['rsi_14'] - 50) / 50  # Normalize to [-1, 1]

        # MACD
        ema_fast = close.ewm(span=self.config.macd_fast).mean()
        ema_slow = close.ewm(span=self.config.macd_slow).mean()
        features['macd'] = (ema_fast - ema_slow) / close * 100  # Percentage
        features['macd_signal'] = features['macd'].ewm(span=self.config.macd_signal).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Stochastic
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        features['stoch_k'] = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # Rate of Change
        features['roc_12'] = close.pct_change(12) * 100

        # === VOLATILITY (8 features) ===
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(self.config.atr_period).mean()
        features['atr_norm'] = features['atr_14'] / close  # Normalized ATR

        # Bollinger Bands
        bb_ma = close.rolling(self.config.bb_period).mean()
        bb_std = close.rolling(self.config.bb_period).std()
        bb_upper = bb_ma + self.config.bb_std * bb_std
        bb_lower = bb_ma - self.config.bb_std * bb_std
        features['bb_upper_dist'] = (close - bb_upper) / close
        features['bb_lower_dist'] = (close - bb_lower) / close
        features['bb_width'] = (bb_upper - bb_lower) / bb_ma

        # Historical Volatility
        returns = close.pct_change()
        features['vol_12h'] = returns.rolling(12).std() * np.sqrt(24 * 365)  # Annualized
        features['vol_24h'] = returns.rolling(24).std() * np.sqrt(24 * 365)
        features['vol_ratio'] = features['vol_12h'] / (features['vol_24h'] + 1e-10)

        # === VOLUME (6 features) ===
        # OBV
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_ma = obv.rolling(50).mean()
        features['obv_norm'] = (obv - obv_ma) / (obv_ma.abs() + 1e-10)
        features['obv_slope'] = obv.diff(self.config.obv_smooth) / self.config.obv_smooth

        # VWAP
        vwap = (volume * (high + low + close) / 3).rolling(self.config.vwap_period).sum() / \
               volume.rolling(self.config.vwap_period).sum()
        features['vwap_dev'] = (close - vwap) / close

        # Volume patterns
        vol_ma = volume.rolling(20).mean()
        features['volume_ma_ratio'] = volume / (vol_ma + 1e-10)
        features['volume_trend'] = volume.rolling(10).mean() / (volume.rolling(50).mean() + 1e-10) - 1
        features['volume_zscore'] = (volume - vol_ma) / (volume.rolling(20).std() + 1e-10)

        # === PRICE PATTERNS (8 features) ===
        features['high_low_ratio'] = (high - low) / (close + 1e-10)
        features['close_position'] = (close - low) / (high - low + 1e-10)
        features['gap_up'] = (open_ - close.shift(1)).clip(lower=0) / close
        features['gap_down'] = (close.shift(1) - open_).clip(lower=0) / close

        body = abs(close - open_)
        full_range = high - low + 1e-10
        features['upper_shadow'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range
        features['lower_shadow'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range
        features['body_size'] = body / full_range

        # Range position (where is current price in recent range)
        rolling_high = high.rolling(24).max()
        rolling_low = low.rolling(24).min()
        features['range_position'] = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)

        # === STATISTICAL (8 features) ===
        returns_24h = returns.rolling(24)
        features['skewness_24h'] = returns_24h.skew()
        features['kurtosis_24h'] = returns_24h.kurt()

        # Autocorrelation
        features['autocorr_1'] = returns.rolling(48).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        features['autocorr_4'] = returns.rolling(48).apply(
            lambda x: x.autocorr(lag=4) if len(x) > 4 else 0, raw=False
        )

        # Hurst approximation (simplified)
        features['hurst_approx'] = self._compute_hurst_approx(returns, 24)

        # Mean reversion score
        ma_50 = close.rolling(50).mean()
        features['mean_rev_score'] = (close - ma_50) / (close.rolling(50).std() + 1e-10)

        # Z-scores
        features['zscore_24h'] = (close - close.rolling(24).mean()) / (close.rolling(24).std() + 1e-10)
        features['zscore_168h'] = (close - close.rolling(168).mean()) / (close.rolling(168).std() + 1e-10)

        # Clean up NaN and Inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)

        return features

    def _compute_hurst_approx(self, returns: pd.Series, window: int) -> pd.Series:
        """Simplified Hurst exponent approximation using R/S ratio."""
        def hurst_rs(x):
            if len(x) < 10:
                return 0.5
            mean = np.mean(x)
            cumdev = np.cumsum(x - mean)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(x)
            if s == 0:
                return 0.5
            rs = r / s
            if rs <= 0:
                return 0.5
            return np.log(rs) / np.log(len(x))

        return returns.rolling(window).apply(hurst_rs, raw=True)


def download_btc_data_ccxt(
    start_date: str,
    end_date: str,
    timeframe: str = '1h',
    exchange: str = 'binance'
) -> pd.DataFrame:
    """
    Download BTC/USDT data via CCXT.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Candle timeframe (1h, 4h, 1d, etc.)
        exchange: Exchange name

    Returns:
        DataFrame with OHLCV data
    """
    import ccxt
    from datetime import datetime
    import time

    print(f"Downloading BTC/USDT {timeframe} data from {exchange}...")
    print(f"Period: {start_date} to {end_date}")

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange)
    ex = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    # Fetch in batches
    all_ohlcv = []
    current_ts = start_ts
    batch_size = 1000  # Most exchanges limit to 1000 candles

    while current_ts < end_ts:
        try:
            ohlcv = ex.fetch_ohlcv(
                'BTC/USDT',
                timeframe=timeframe,
                since=current_ts,
                limit=batch_size
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1  # Next candle

            print(f"  Downloaded {len(all_ohlcv)} candles...", end='\r')
            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"\nError at {current_ts}: {e}")
            time.sleep(1)
            continue

    print(f"\n  Total: {len(all_ohlcv)} candles")

    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df


def create_44_feature_dataset(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    output_path: str = None
) -> pd.DataFrame:
    """
    Download BTC data and create 44-feature dataset.

    Args:
        start_date: Start date
        end_date: End date
        output_path: Optional path to save .pkl

    Returns:
        DataFrame with OHLCV + 44 features
    """
    # Download data
    df = download_btc_data_ccxt(start_date, end_date, '1h')

    print(f"\nComputing 44 features...")

    # Compute features
    engineer = FeatureEngineer()
    features = engineer.compute_features(df)

    # Combine OHLCV with features
    result = pd.concat([df, features], axis=1)

    # Drop rows with NaN (warmup period)
    result = result.dropna()

    print(f"Final dataset: {len(result)} rows, {len(result.columns)} columns")
    print(f"Date range: {result.index[0]} to {result.index[-1]}")

    # Save if path provided
    if output_path:
        result.to_pickle(output_path)
        print(f"Saved to: {output_path}")

    return result


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Feature Engineer Test")
    print("=" * 60)

    # Create sample OHLCV data
    np.random.seed(42)
    n = 500

    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    price = 40000 + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        'open': price + np.random.randn(n) * 50,
        'high': price + abs(np.random.randn(n) * 100),
        'low': price - abs(np.random.randn(n) * 100),
        'close': price,
        'volume': np.abs(np.random.randn(n) * 1000 + 5000)
    }, index=dates)

    # Compute features
    engineer = FeatureEngineer()
    features = engineer.compute_features(df)

    print(f"\nFeatures shape: {features.shape}")
    print(f"Feature names ({len(engineer.feature_names)}):")
    for i, name in enumerate(engineer.feature_names):
        print(f"  {i:2d}: {name}")

    print(f"\nSample (first 5 rows, first 10 features):")
    print(features.iloc[:5, :10])

    print("\n" + "=" * 60)
    print("Feature Engineer test passed!")
