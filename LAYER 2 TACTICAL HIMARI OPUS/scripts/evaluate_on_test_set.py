"""
Experiment 9 Test Set Evaluation Script
Run this on Vast.ai or Google Colab to check for overfitting.

This script:
1. Loads the best checkpoint from Experiment 9
2. Runs it on the UNSEEN test set (2023-2024 data)
3. Compares performance to validation set
4. Performs shuffle test to detect overfitting
"""

import sys
import pickle
import numpy as np
import torch
import logging
from typing import Dict, Tuple

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load BTC 5-minute data."""
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    features = np.array(raw_data.get('features', list(raw_data.values())[0]), dtype=np.float32)
    prices = np.array(raw_data.get('prices', features[:, 0]), dtype=np.float32)
    
    logger.info(f"Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
    return features, prices


def evaluate_on_split(
    model,
    features: np.ndarray,
    prices: np.ndarray,
    start_idx: int,
    end_idx: int,
    device: str = 'cuda',
    temperature: float = 0.5,
) -> Dict:
    """Evaluate model on a specific data split."""
    from src.environment.transformer_a2c_env import TransformerEnvConfig, TransformerA2CEnv
    from src.training.sortino_reward import SortinoWithCarryCost
    
    # Create environment for this split
    split_features = features[start_idx:end_idx]
    split_prices = prices[start_idx:end_idx]
    
    config = TransformerEnvConfig(context_length=100, feature_dim=features.shape[1])
    env = TransformerA2CEnv(split_features, split_prices, config)
    
    # Initialize reward function
    reward_fn = SortinoWithCarryCost(
        trading_fee=0.001,
        slippage=0.0005,
        carry_cost=0.00005
    )
    
    model.eval()
    
    # Collect actions and returns
    actions_taken = []
    returns = []

    obs, info = env.reset()
    done = False

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(obs_tensor, deterministic=False)
            probs = output['probs'] / temperature
            probs = torch.softmax(probs, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        obs, reward, done, info = env.step(action)

        actions_taken.append(action)
        # The step() returns market_return as the reward directly
        returns.append(reward)
    
    # Compute metrics
    actions_array = np.array(actions_taken)
    returns_array = np.array(returns)
    
    # Action distribution
    flat_pct = np.mean(actions_array == 0) * 100
    long_pct = np.mean(actions_array == 1) * 100
    short_pct = np.mean(actions_array == 2) * 100
    
    # Sharpe ratio
    if len(returns_array) > 1 and np.std(returns_array) > 0:
        sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252 * 288)
    else:
        sharpe = 0.0
    
    # Trade count
    trades = np.sum(np.diff(actions_array) != 0)
    
    # Total return
    total_return = np.sum(returns_array)
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'trades': trades,
        'flat_pct': flat_pct,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'n_samples': len(actions_array),
    }


def shuffle_test(
    model,
    features: np.ndarray,
    prices: np.ndarray,
    start_idx: int,
    end_idx: int,
    device: str = 'cuda',
    n_shuffles: int = 5,
) -> Dict:
    """Run shuffle test to detect overfitting."""
    from src.environment.transformer_a2c_env import TransformerEnvConfig, TransformerA2CEnv
    
    shuffle_sharpes = []
    
    for i in range(n_shuffles):
        # Shuffle the features randomly
        split_features = features[start_idx:end_idx].copy()
        split_prices = prices[start_idx:end_idx].copy()
        
        # Shuffle rows
        perm = np.random.permutation(len(split_features))
        shuffled_features = split_features[perm]
        shuffled_prices = split_prices[perm]
        
        # Quick evaluation
        config = TransformerEnvConfig(context_length=100, feature_dim=features.shape[1])
        env = TransformerA2CEnv(shuffled_features, shuffled_prices, config)
        
        model.eval()
        obs, info = env.reset()
        returns = []

        for _ in range(min(10000, len(shuffled_features) - 100)):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(obs_tensor, deterministic=True)
                action = output['action'].item()

            obs, reward, done, info = env.step(action)
            returns.append(info.get('return', 0))

            if done:
                break
        
        returns_array = np.array(returns)
        if len(returns_array) > 1 and np.std(returns_array) > 0:
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252 * 288)
        else:
            sharpe = 0.0
        
        shuffle_sharpes.append(sharpe)
        logger.info(f"Shuffle test {i+1}/{n_shuffles}: Sharpe = {sharpe:.2f}")
    
    return {
        'mean_shuffle_sharpe': np.mean(shuffle_sharpes),
        'std_shuffle_sharpe': np.std(shuffle_sharpes),
        'shuffle_sharpes': shuffle_sharpes,
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Experiment 9 on test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data', type=str, default='data/btc_5min_2020_2024.pkl', help='Path to data file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--shuffle-test', action='store_true', help='Run shuffle test')
    args = parser.parse_args()
    
    # Load data
    features, prices = load_data(args.data)
    
    # Define splits (60/20/20)
    n_samples = len(features)
    train_end = int(n_samples * 0.6)  # 316,200
    val_end = int(n_samples * 0.8)    # 421,600
    
    logger.info(f"Train: [0:{train_end}], Val: [{train_end}:{val_end}], Test: [{val_end}:{n_samples}]")
    
    # Load model
    from src.models.transformer_a2c import TransformerA2C, TransformerA2CConfig
    
    config = TransformerA2CConfig(
        input_dim=features.shape[1],
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        context_length=100,
    )
    
    model = TransformerA2C(config).to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluate on validation set (for comparison)
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SET EVALUATION (should match training)")
    logger.info("="*60)
    
    val_results = evaluate_on_split(model, features, prices, train_end, val_end, args.device)
    logger.info(f"Val Sharpe: {val_results['sharpe']:.2f}")
    logger.info(f"Val Return: {val_results['total_return']*100:.1f}%")
    logger.info(f"Val Trades: {val_results['trades']}")
    logger.info(f"Val Actions: FLAT={val_results['flat_pct']:.1f}%, LONG={val_results['long_pct']:.1f}%, SHORT={val_results['short_pct']:.1f}%")
    
    # Evaluate on TEST set (UNSEEN data!)
    logger.info("\n" + "="*60)
    logger.info("TEST SET EVALUATION (UNSEEN 2023-2024 DATA)")
    logger.info("="*60)
    
    test_results = evaluate_on_split(model, features, prices, val_end, n_samples, args.device)
    logger.info(f"Test Sharpe: {test_results['sharpe']:.2f}")
    logger.info(f"Test Return: {test_results['total_return']*100:.1f}%")
    logger.info(f"Test Trades: {test_results['trades']}")
    logger.info(f"Test Actions: FLAT={test_results['flat_pct']:.1f}%, LONG={test_results['long_pct']:.1f}%, SHORT={test_results['short_pct']:.1f}%")
    
    # Overfitting check
    logger.info("\n" + "="*60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*60)
    
    sharpe_drop = val_results['sharpe'] - test_results['sharpe']
    sharpe_drop_pct = (sharpe_drop / val_results['sharpe'] * 100) if val_results['sharpe'] > 0 else 0
    
    logger.info(f"Sharpe Drop: {sharpe_drop:.2f} ({sharpe_drop_pct:.1f}%)")
    
    if sharpe_drop_pct > 50:
        logger.warning("⚠️ SEVERE OVERFITTING: Test Sharpe dropped >50%")
    elif sharpe_drop_pct > 30:
        logger.warning("⚠️ MODERATE OVERFITTING: Test Sharpe dropped 30-50%")
    elif sharpe_drop_pct > 10:
        logger.info("⚠️ MILD OVERFITTING: Test Sharpe dropped 10-30%")
    else:
        logger.info("✅ NO OVERFITTING: Test Sharpe within 10% of validation")
    
    # Shuffle test
    if args.shuffle_test:
        logger.info("\n" + "="*60)
        logger.info("SHUFFLE TEST (detecting spurious patterns)")
        logger.info("="*60)
        
        shuffle_results = shuffle_test(model, features, prices, val_end, n_samples, args.device)
        logger.info(f"Shuffle Sharpe Mean: {shuffle_results['mean_shuffle_sharpe']:.2f}")
        logger.info(f"Shuffle Sharpe Std: {shuffle_results['std_shuffle_sharpe']:.2f}")
        
        if abs(shuffle_results['mean_shuffle_sharpe']) < 5:
            logger.info("✅ SHUFFLE TEST PASSED: Model doesn't work on random data")
        else:
            logger.warning("⚠️ SHUFFLE TEST FAILED: Model produces signal on random data!")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Validation Sharpe: {val_results['sharpe']:.2f}")
    logger.info(f"Test Sharpe: {test_results['sharpe']:.2f}")
    logger.info(f"Sharpe Degradation: {sharpe_drop_pct:.1f}%")
    
    if test_results['sharpe'] > 10:
        logger.info("✅ MODEL SHOWS REAL EDGE ON UNSEEN DATA")
    elif test_results['sharpe'] > 0:
        logger.info("⚠️ MODEL SHOWS WEAK EDGE ON UNSEEN DATA")
    else:
        logger.warning("❌ MODEL LOSES MONEY ON UNSEEN DATA - OVERFITTING DETECTED")
    
    return {
        'validation': val_results,
        'test': test_results,
        'sharpe_drop_pct': sharpe_drop_pct,
    }


if __name__ == '__main__':
    main()
