"""
Download BTC 5-min data locally (run on your PC, not Vast.ai)
Then upload to Google Drive for use on Vast.ai
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time

def download_btc_5min(start_date='2025-01-01', end_date='2026-01-13', output_path='btc_5min_2025_2026.pkl'):
    """Download BTC 5-min data from Binance."""
    try:
        import ccxt
    except ImportError:
        print("Install ccxt first: pip install ccxt")
        return
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    start_ts = exchange.parse8601(f'{start_date}T00:00:00Z')
    end_ts = exchange.parse8601(f'{end_date}T00:00:00Z')
    
    all_ohlcv = []
    current_ts = start_ts
    
    print(f"Downloading BTC/USDT 5m data from {start_date} to {end_date}...")
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', since=current_ts, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            print(f"Downloaded {len(all_ohlcv)} bars... ({datetime.fromtimestamp(current_ts/1000)})")
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"\nDownloaded {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Engineer features
    print("Engineering features...")
    features, prices = engineer_features(df)
    
    # Save as pickle
    data = {
        'features': features,
        'prices': prices,
        'timestamps': df['timestamp'].values,
        'ohlcv': df.values,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved to {output_path}")
    print(f"Features shape: {features.shape}")
    print(f"Upload this file to Google Drive for Vast.ai!")


def engineer_features(df: pd.DataFrame) -> tuple:
    """Engineer 44 features matching training data."""
    
    features = pd.DataFrame(index=df.index)
    
    # Price returns
    features['return_1'] = df['close'].pct_change(1)
    features['return_5'] = df['close'].pct_change(5)
    features['return_15'] = df['close'].pct_change(15)
    features['return_60'] = df['close'].pct_change(60)
    
    # Volatility
    features['volatility_15'] = df['close'].pct_change().rolling(15).std()
    features['volatility_60'] = df['close'].pct_change().rolling(60).std()
    features['volatility_240'] = df['close'].pct_change().rolling(240).std()
    
    # Volume
    features['volume_ratio_15'] = df['volume'] / df['volume'].rolling(15).mean()
    features['volume_ratio_60'] = df['volume'] / df['volume'].rolling(60).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi_14'] = gain / (gain + loss + 1e-8)
    
    gain28 = (delta.where(delta > 0, 0)).rolling(28).mean()
    loss28 = (-delta.where(delta < 0, 0)).rolling(28).mean()
    features['rsi_28'] = gain28 / (gain28 + loss28 + 1e-8)
    
    # MA ratios
    features['ma_5'] = df['close'] / df['close'].rolling(5).mean() - 1
    features['ma_15'] = df['close'] / df['close'].rolling(15).mean() - 1
    features['ma_60'] = df['close'] / df['close'].rolling(60).mean() - 1
    features['ma_240'] = df['close'] / df['close'].rolling(240).mean() - 1
    
    # Bollinger
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    features['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std + 1e-8)
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['close']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    features['atr_14'] = tr.rolling(14).mean() / df['close']
    features['atr_60'] = tr.rolling(60).mean() / df['close']
    
    # Candle patterns
    features['range_normalized'] = (df['high'] - df['low']) / df['close']
    features['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Trends
    features['trend_15'] = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    features['trend_60'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    features['trend_240'] = (df['close'] - df['close'].shift(240)) / df['close'].shift(240)
    
    # Volume-price correlation
    features['volume_price_corr'] = df['close'].rolling(60).corr(df['volume'])
    
    # Momentum
    features['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close']
    features['momentum_15'] = (df['close'] - df['close'].shift(15)) / df['close']
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    features['stoch_k'] = (df['close'] - low_14) / (high_14 - low_14 + 1e-8)
    features['stoch_d'] = features['stoch_k'].rolling(3).mean()
    features['williams_r'] = (high_14 - df['close']) / (high_14 - low_14 + 1e-8)
    
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    features['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-8)
    
    # ROC
    features['roc_5'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-8)
    features['roc_15'] = (df['close'] - df['close'].shift(15)) / (df['close'].shift(15) + 1e-8)
    
    # Pad to 44
    while len(features.columns) < 44:
        features[f'padding_{len(features.columns)}'] = 0.0
    
    features = features.iloc[:, :44]
    features = features.fillna(0).clip(-10, 10)
    
    return features.values.astype(np.float32), df['close'].values.astype(np.float32)


if __name__ == '__main__':
    download_btc_5min()
    print("\nDone! Upload btc_5min_2025_2026.pkl to Google Drive")
