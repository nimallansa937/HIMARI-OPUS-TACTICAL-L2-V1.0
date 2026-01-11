"""
HIMARI Layer 2 - Model Evaluation Script
Tests trained model on held-out test set and compares to baselines.
"""

import sys
import pickle
import numpy as np
import torch
import logging
from collections import defaultdict

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(checkpoint_path: str, data_path: str = 'data/btc_5min_2020_2024.pkl'):
    """
    Comprehensive evaluation of trained model.
    
    Tests:
    1. Performance on held-out test set
    2. Comparison to buy-and-hold baseline
    3. Trade-by-trade analysis
    """
    
    # Load data
    logger.info("Loading data...")
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    features = np.array(raw_data.get('features', list(raw_data.values())[0]), dtype=np.float32)
    prices = np.array(raw_data.get('prices', features[:, 0]), dtype=np.float32)
    
    logger.info(f"Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Create test environment
    from src.environment.transformer_a2c_env import WalkForwardSplitter, TransformerEnvConfig
    from src.models.transformer_a2c import TransformerA2C, TransformerA2CConfig
    
    splitter = WalkForwardSplitter(features, prices, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    _, _, test_env = splitter.create_envs(
        config=TransformerEnvConfig(context_length=100, feature_dim=features.shape[1])
    )
    
    test_data, test_prices = splitter.get_test_data()
    logger.info(f"Test set: {len(test_data)} samples")
    
    # Load model
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    config = TransformerA2CConfig(**checkpoint['config'])
    model = TransformerA2C(config).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from step {checkpoint['global_step']}")
    
    # =========================================================================
    # TEST 1: Run model on test set
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Model Performance on Test Set")
    logger.info("=" * 70)
    
    state, _ = test_env.reset()
    
    actions = []
    returns = []
    confidences = []
    positions = []
    
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
        
        with torch.no_grad():
            output = model(state_tensor, deterministic=True)
        
        action = output["action"].item()
        confidence = output["confidence"].item()
        
        next_state, market_return, done, info = test_env.step(action)
        
        # Position mapping: FLAT=0, LONG=1, SHORT=2 -> position: 0, 1, -1
        position = {0: 0, 1: 1, 2: -1}[action]
        position_return = position * market_return
        
        actions.append(action)
        returns.append(position_return)
        confidences.append(confidence)
        positions.append(position)
        
        state = next_state
    
    # Calculate metrics
    returns = np.array(returns)
    actions = np.array(actions)
    
    total_return = np.sum(returns)
    cumulative = np.cumsum(returns)
    
    # Sharpe
    if np.std(returns) > 1e-8:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(105120)  # Annualized
    else:
        sharpe = 0.0
    
    # Max drawdown
    peak = np.maximum.accumulate(cumulative + 1)
    drawdown = (cumulative + 1 - peak) / peak
    max_dd = np.min(drawdown)
    
    # Action distribution
    action_counts = {0: np.sum(actions == 0), 1: np.sum(actions == 1), 2: np.sum(actions == 2)}
    total_actions = len(actions)
    
    logger.info(f"\n📊 MODEL RESULTS:")
    logger.info(f"  Total Return: {total_return * 100:.2f}%")
    logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
    logger.info(f"  Max Drawdown: {max_dd * 100:.2f}%")
    logger.info(f"  Action Distribution:")
    logger.info(f"    FLAT:  {action_counts[0]:,} ({action_counts[0]/total_actions*100:.1f}%)")
    logger.info(f"    LONG:  {action_counts[1]:,} ({action_counts[1]/total_actions*100:.1f}%)")
    logger.info(f"    SHORT: {action_counts[2]:,} ({action_counts[2]/total_actions*100:.1f}%)")
    
    # =========================================================================
    # TEST 2: Buy-and-Hold Baseline
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Buy-and-Hold Baseline Comparison")
    logger.info("=" * 70)
    
    # Calculate buy-and-hold returns
    bah_returns = np.diff(test_prices) / test_prices[:-1]
    bah_total_return = (test_prices[-1] / test_prices[0]) - 1
    
    if np.std(bah_returns) > 1e-8:
        bah_sharpe = (np.mean(bah_returns) / np.std(bah_returns)) * np.sqrt(105120)
    else:
        bah_sharpe = 0.0
    
    bah_cumulative = np.cumsum(bah_returns)
    bah_peak = np.maximum.accumulate(bah_cumulative + 1)
    bah_drawdown = (bah_cumulative + 1 - bah_peak) / bah_peak
    bah_max_dd = np.min(bah_drawdown)
    
    logger.info(f"\n📈 BUY-AND-HOLD RESULTS:")
    logger.info(f"  Total Return: {bah_total_return * 100:.2f}%")
    logger.info(f"  Sharpe Ratio: {bah_sharpe:.4f}")
    logger.info(f"  Max Drawdown: {bah_max_dd * 100:.2f}%")
    
    logger.info(f"\n🔍 COMPARISON:")
    if total_return > bah_total_return:
        logger.info(f"  ✅ Model outperforms buy-and-hold by {(total_return - bah_total_return) * 100:.2f}%")
    else:
        logger.info(f"  ❌ Model underperforms buy-and-hold by {(bah_total_return - total_return) * 100:.2f}%")
    
    if sharpe > bah_sharpe:
        logger.info(f"  ✅ Model has better risk-adjusted returns (Sharpe: {sharpe:.2f} vs {bah_sharpe:.2f})")
    else:
        logger.info(f"  ❌ Model has worse risk-adjusted returns (Sharpe: {sharpe:.2f} vs {bah_sharpe:.2f})")
    
    # =========================================================================
    # TEST 3: Trade-by-Trade Analysis
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Trade-by-Trade Analysis")
    logger.info("=" * 70)
    
    # Find trade segments (continuous positions)
    trades = []
    current_position = 0
    entry_idx = 0
    trade_return = 0
    
    for i, pos in enumerate(positions):
        if pos != current_position:
            if current_position != 0:  # Closing a position
                trades.append({
                    'position': 'LONG' if current_position == 1 else 'SHORT',
                    'entry': entry_idx,
                    'exit': i,
                    'duration': i - entry_idx,
                    'return': trade_return
                })
            if pos != 0:  # Opening new position
                entry_idx = i
                trade_return = 0
            current_position = pos
        elif current_position != 0:
            trade_return += returns[i]
    
    # Close final trade if any
    if current_position != 0:
        trades.append({
            'position': 'LONG' if current_position == 1 else 'SHORT',
            'entry': entry_idx,
            'exit': len(positions),
            'duration': len(positions) - entry_idx,
            'return': trade_return
        })
    
    if trades:
        trade_returns = [t['return'] for t in trades]
        winning_trades = [t for t in trades if t['return'] > 0]
        losing_trades = [t for t in trades if t['return'] <= 0]
        
        logger.info(f"\n📉 TRADE STATISTICS:")
        logger.info(f"  Total Trades: {len(trades)}")
        logger.info(f"  Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
        logger.info(f"  Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
        logger.info(f"  Average Trade Return: {np.mean(trade_returns)*100:.4f}%")
        logger.info(f"  Best Trade: {max(trade_returns)*100:.4f}%")
        logger.info(f"  Worst Trade: {min(trade_returns)*100:.4f}%")
        
        # Trade duration
        durations = [t['duration'] for t in trades]
        logger.info(f"  Average Duration: {np.mean(durations):.0f} bars ({np.mean(durations)*5:.0f} minutes)")
        
        # Show first 10 trades
        logger.info(f"\n📋 FIRST 10 TRADES:")
        for i, trade in enumerate(trades[:10]):
            emoji = "✅" if trade['return'] > 0 else "❌"
            logger.info(f"  {emoji} {trade['position']}: bars {trade['entry']}-{trade['exit']}, "
                       f"duration={trade['duration']}, return={trade['return']*100:.4f}%")
    else:
        logger.info("  ⚠️ No trades executed (model stayed FLAT)")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)
    
    is_valid = True
    issues = []
    
    if action_counts[0] / total_actions > 0.9:
        issues.append("Policy collapsed to 90%+ FLAT")
        is_valid = False
    if action_counts[1] / total_actions > 0.9:
        issues.append("Policy collapsed to 90%+ LONG")
        is_valid = False
    if action_counts[2] / total_actions > 0.9:
        issues.append("Policy collapsed to 90%+ SHORT")
        is_valid = False
    if abs(sharpe) > 10:
        issues.append(f"Sharpe {sharpe:.1f} is unrealistic (likely noise)")
        is_valid = False
    if total_return < bah_total_return and sharpe < bah_sharpe:
        issues.append("Model underperforms simple buy-and-hold")
    
    if is_valid and len(issues) == 0:
        logger.info("✅ MODEL APPEARS VALID - Shows real trading behavior")
    else:
        logger.info("⚠️ MODEL HAS ISSUES:")
        for issue in issues:
            logger.info(f"  - {issue}")
    
    return {
        "model_total_return": total_return,
        "model_sharpe": sharpe,
        "model_max_dd": max_dd,
        "bah_total_return": bah_total_return,
        "bah_sharpe": bah_sharpe,
        "bah_max_dd": bah_max_dd,
        "action_distribution": action_counts,
        "num_trades": len(trades) if trades else 0,
        "win_rate": len(winning_trades) / len(trades) if trades else 0,
        "is_valid": is_valid,
        "issues": issues
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained Transformer-A2C model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data", type=str, default="data/btc_5min_2020_2024.pkl", help="Path to data file")
    
    args = parser.parse_args()
    
    results = evaluate_model(args.checkpoint, args.data)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model Return: {results['model_total_return']*100:.2f}% | Sharpe: {results['model_sharpe']:.2f}")
    print(f"B&H Return:   {results['bah_total_return']*100:.2f}% | Sharpe: {results['bah_sharpe']:.2f}")
    print(f"Trades: {results['num_trades']} | Win Rate: {results['win_rate']*100:.1f}%")
    print(f"Valid: {'✅' if results['is_valid'] else '❌'}")
