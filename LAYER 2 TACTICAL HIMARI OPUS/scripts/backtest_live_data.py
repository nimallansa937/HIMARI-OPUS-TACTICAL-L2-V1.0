"""
Live Data Backtest with Equity Curve Visualization

This script:
1. Downloads BTC 5-min data from Jan 2025 - Jan 2026 (truly unseen!)
2. Engineers the same 44 features used in training
3. Runs the Experiment 9 model
4. Generates equity curve and performance graphs
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def download_btc_data(start_date: str = '2025-01-01', end_date: str = '2026-01-13'):
    """Download BTC 5-min OHLCV data from Binance."""
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        return None
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    start_ts = exchange.parse8601(f'{start_date}T00:00:00Z')
    end_ts = exchange.parse8601(f'{end_date}T00:00:00Z')
    
    all_ohlcv = []
    current_ts = start_ts
    
    logger.info(f"Downloading BTC/USDT 5m data from {start_date} to {end_date}...")
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', since=current_ts, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            logger.info(f"Downloaded {len(all_ohlcv)} bars...")
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    logger.info(f"Downloaded {len(df)} total bars from {df.index[0]} to {df.index[-1]}")
    
    return df


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    """Engineer the same 44 features used in training."""
    
    features = pd.DataFrame(index=df.index)
    
    # Price-based features (normalized)
    features['return_1'] = df['close'].pct_change(1)
    features['return_5'] = df['close'].pct_change(5)
    features['return_15'] = df['close'].pct_change(15)
    features['return_60'] = df['close'].pct_change(60)
    
    # Volatility features
    features['volatility_15'] = df['close'].pct_change().rolling(15).std()
    features['volatility_60'] = df['close'].pct_change().rolling(60).std()
    features['volatility_240'] = df['close'].pct_change().rolling(240).std()
    
    # Volume features
    features['volume_ratio_15'] = df['volume'] / df['volume'].rolling(15).mean()
    features['volume_ratio_60'] = df['volume'] / df['volume'].rolling(60).mean()
    
    # Price momentum
    features['rsi_14'] = compute_rsi(df['close'], 14)
    features['rsi_28'] = compute_rsi(df['close'], 28)
    
    # Moving average crossovers
    features['ma_5'] = df['close'] / df['close'].rolling(5).mean() - 1
    features['ma_15'] = df['close'] / df['close'].rolling(15).mean() - 1
    features['ma_60'] = df['close'] / df['close'].rolling(60).mean() - 1
    features['ma_240'] = df['close'] / df['close'].rolling(240).mean() - 1
    
    # Bollinger bands
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
    
    # High-low range
    features['range_normalized'] = (df['high'] - df['low']) / df['close']
    
    # Candle patterns
    features['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Trend features
    features['trend_15'] = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    features['trend_60'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    features['trend_240'] = (df['close'] - df['close'].shift(240)) / df['close'].shift(240)
    
    # Volume-price correlation
    features['volume_price_corr'] = df['close'].rolling(60).corr(df['volume'])
    
    # More momentum indicators
    features['momentum_5'] = df['close'] - df['close'].shift(5)
    features['momentum_15'] = df['close'] - df['close'].shift(15)
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    features['stoch_k'] = (df['close'] - low_14) / (high_14 - low_14 + 1e-8)
    features['stoch_d'] = features['stoch_k'].rolling(3).mean()
    
    # Williams %R
    features['williams_r'] = (high_14 - df['close']) / (high_14 - low_14 + 1e-8)
    
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    features['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-8)
    
    # Rate of change
    features['roc_5'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-8)
    features['roc_15'] = (df['close'] - df['close'].shift(15)) / (df['close'].shift(15) + 1e-8)
    
    # Pad to 44 features if needed
    while len(features.columns) < 44:
        features[f'padding_{len(features.columns)}'] = 0.0
    
    # Take first 44 features
    features = features.iloc[:, :44]
    
    # Fill NaN and clip extremes
    features = features.fillna(0)
    features = features.clip(-10, 10)
    
    return features.values.astype(np.float32), df['close'].values.astype(np.float32)


def compute_rsi(prices, period):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return rs / (1 + rs)  # Normalized to 0-1


def run_backtest(
    model,
    features: np.ndarray,
    prices: np.ndarray,
    device: str = 'cuda',
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    temperature: float = 0.5,
):
    """Run backtest and return equity curve."""
    from src.environment.transformer_a2c_env import TransformerEnvConfig, TransformerA2CEnv
    
    config = TransformerEnvConfig(context_length=100, feature_dim=features.shape[1])
    env = TransformerA2CEnv(features, prices, config)
    
    model.eval()
    
    # Tracking
    equity_curve = [initial_capital]
    positions = []
    actions_history = []
    returns_history = []
    
    capital = initial_capital
    prev_action = 0  # Start FLAT
    
    obs, info = env.reset()
    done = False
    step = 0
    
    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(obs_tensor, deterministic=False)
            probs = output['probs'] / temperature
            probs = torch.softmax(probs, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        obs, market_return, done, info = env.step(action)
        
        # Calculate position return
        if prev_action == 1:  # LONG
            position_return = market_return
        elif prev_action == 2:  # SHORT
            position_return = -market_return
        else:
            position_return = 0.0
        
        # Apply transaction cost if position changed
        if action != prev_action:
            capital *= (1 - fee_rate)
        
        # Update capital
        capital *= (1 + position_return)
        
        equity_curve.append(capital)
        positions.append(prev_action)
        actions_history.append(action)
        returns_history.append(position_return)
        
        prev_action = action
        step += 1
    
    return {
        'equity_curve': np.array(equity_curve),
        'positions': np.array(positions),
        'actions': np.array(actions_history),
        'returns': np.array(returns_history),
    }


def plot_results(results: dict, prices: np.ndarray, save_path: str = 'backtest_results.png'):
    """Generate visualization of backtest results."""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. Equity Curve
    ax1 = axes[0]
    equity = results['equity_curve']
    ax1.plot(equity, color='blue', linewidth=1.5)
    ax1.axhline(y=equity[0], color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(range(len(equity)), equity[0], equity, where=equity >= equity[0], 
                     color='green', alpha=0.3)
    ax1.fill_between(range(len(equity)), equity[0], equity, where=equity < equity[0], 
                     color='red', alpha=0.3)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'Equity Curve - Final: ${equity[-1]:,.0f} ({(equity[-1]/equity[0]-1)*100:+.1f}%)')
    ax1.grid(True, alpha=0.3)
    
    # 2. BTC Price (Buy & Hold comparison)
    ax2 = axes[1]
    btc_normalized = prices[:len(equity)-1] / prices[0] * equity[0]
    ax2.plot(btc_normalized, color='orange', linewidth=1.5, label='BTC Buy & Hold')
    ax2.plot(equity[:-1], color='blue', linewidth=1.5, label='Model Strategy')
    ax2.set_ylabel('Value ($)')
    ax2.set_title('Strategy vs Buy & Hold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position over time
    ax3 = axes[2]
    positions = results['positions']
    colors = ['gray', 'green', 'red']  # FLAT, LONG, SHORT
    for i, (pos, color) in enumerate(zip([0, 1, 2], colors)):
        mask = positions == pos
        ax3.fill_between(range(len(positions)), 0, 1, where=mask, color=color, alpha=0.5, 
                        label=['FLAT', 'LONG', 'SHORT'][i])
    ax3.set_ylabel('Position')
    ax3.set_title('Position Over Time')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 1)
    
    # 4. Drawdown
    ax4 = axes[3]
    rolling_max = np.maximum.accumulate(equity)
    drawdown = (equity - rolling_max) / rolling_max * 100
    ax4.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.5)
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_xlabel('Time (5-min bars)')
    ax4.set_title(f'Drawdown - Max: {np.min(drawdown):.1f}%')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {save_path}")
    plt.show()
    
    return fig


def main():
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Backtest on unseen 2025-2026 data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default=None, help='Path to pickle file with features/prices (skip download)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--start-date', type=str, default='2025-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2026-01-13', help='End date')
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='Starting capital')
    parser.add_argument('--output', type=str, default='backtest_2025_results.png', help='Output path')
    args = parser.parse_args()
    
    # Load data from pickle or download
    if args.data:
        logger.info(f"Loading data from {args.data}...")
        with open(args.data, 'rb') as f:
            data = pickle.load(f)
        features = np.array(data['features'], dtype=np.float32)
        prices = np.array(data['prices'], dtype=np.float32)
        logger.info(f"Loaded {len(features)} samples")
    else:
        # Download data
        df = download_btc_data(args.start_date, args.end_date)
        if df is None or len(df) < 1000:
            logger.error("Not enough data downloaded")
            return
        
        # Engineer features
        logger.info("Engineering features...")
        features, prices = engineer_features(df)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Features shape: {features.shape}")
    
    # Load model
    from src.models.transformer_a2c import TransformerA2C, TransformerA2CConfig
    
    config = TransformerA2CConfig(
        input_dim=44,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        context_length=100,
    )
    
    model = TransformerA2C(config).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Run backtest
    logger.info("Running backtest...")
    results = run_backtest(model, features, prices, args.device, args.initial_capital)
    
    # Calculate metrics
    equity = results['equity_curve']
    returns = results['returns']
    
    total_return = (equity[-1] / equity[0] - 1) * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288)
    max_dd = np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)) * 100
    
    actions = results['actions']
    long_pct = np.mean(actions == 1) * 100
    short_pct = np.mean(actions == 2) * 100
    flat_pct = np.mean(actions == 0) * 100
    trades = np.sum(np.diff(actions) != 0)
    
    logger.info("="*60)
    logger.info("BACKTEST RESULTS (2025-2026 UNSEEN DATA)")
    logger.info("="*60)
    logger.info(f"Initial Capital: ${args.initial_capital:,.0f}")
    logger.info(f"Final Capital: ${equity[-1]:,.0f}")
    logger.info(f"Total Return: {total_return:+.1f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_dd:.1f}%")
    logger.info(f"Total Trades: {trades}")
    logger.info(f"Actions: FLAT={flat_pct:.1f}%, LONG={long_pct:.1f}%, SHORT={short_pct:.1f}%")
    
    # Plot
    plot_results(results, prices, args.output)
    
    return results


if __name__ == '__main__':
    main()
