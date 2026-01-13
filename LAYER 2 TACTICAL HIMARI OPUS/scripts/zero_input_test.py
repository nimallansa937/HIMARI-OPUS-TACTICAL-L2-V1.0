"""
Zero Input Test - Check if model has learned a constant positional bias.

If the model outputs the same action distribution (~54% LONG) regardless
of input, it means the model hasn't learned real patterns - just bias.
"""

import sys
import numpy as np
import torch
import logging

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def zero_input_test(checkpoint_path: str, device: str = 'cuda', n_samples: int = 10000):
    """Test if model outputs constant actions regardless of input."""
    
    from src.models.transformer_a2c import TransformerA2C, TransformerA2CConfig
    
    # Load model
    config = TransformerA2CConfig(
        input_dim=44,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        context_length=100,
    )
    
    model = TransformerA2C(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # Test 1: All zeros
    logger.info("\n" + "="*60)
    logger.info("TEST 1: ALL ZEROS INPUT")
    logger.info("="*60)
    
    zero_obs = torch.zeros(1, 100, 44).to(device)  # (batch, context, features)
    
    actions_zero = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model(zero_obs, deterministic=False)
            action = output['action'].item()
            actions_zero.append(action)
    
    actions_zero = np.array(actions_zero)
    flat_pct = np.mean(actions_zero == 0) * 100
    long_pct = np.mean(actions_zero == 1) * 100
    short_pct = np.mean(actions_zero == 2) * 100
    
    logger.info(f"Actions on ZERO input: FLAT={flat_pct:.1f}%, LONG={long_pct:.1f}%, SHORT={short_pct:.1f}%")
    
    # Test 2: Random noise
    logger.info("\n" + "="*60)
    logger.info("TEST 2: RANDOM NOISE INPUT")
    logger.info("="*60)
    
    actions_random = []
    for _ in range(n_samples):
        random_obs = torch.randn(1, 100, 44).to(device)
        with torch.no_grad():
            output = model(random_obs, deterministic=False)
            action = output['action'].item()
            actions_random.append(action)
    
    actions_random = np.array(actions_random)
    flat_pct_r = np.mean(actions_random == 0) * 100
    long_pct_r = np.mean(actions_random == 1) * 100
    short_pct_r = np.mean(actions_random == 2) * 100
    
    logger.info(f"Actions on RANDOM input: FLAT={flat_pct_r:.1f}%, LONG={long_pct_r:.1f}%, SHORT={short_pct_r:.1f}%")
    
    # Test 3: Constant positive input (all 1s)
    logger.info("\n" + "="*60)
    logger.info("TEST 3: ALL ONES INPUT")
    logger.info("="*60)
    
    ones_obs = torch.ones(1, 100, 44).to(device)
    
    actions_ones = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model(ones_obs, deterministic=False)
            action = output['action'].item()
            actions_ones.append(action)
    
    actions_ones = np.array(actions_ones)
    flat_pct_o = np.mean(actions_ones == 0) * 100
    long_pct_o = np.mean(actions_ones == 1) * 100
    short_pct_o = np.mean(actions_ones == 2) * 100
    
    logger.info(f"Actions on ONES input: FLAT={flat_pct_o:.1f}%, LONG={long_pct_o:.1f}%, SHORT={short_pct_o:.1f}%")
    
    # Analysis
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    
    # Check if distributions are similar (constant bias test)
    long_variation = np.std([long_pct, long_pct_r, long_pct_o])
    
    logger.info(f"LONG % variation across inputs: {long_variation:.1f}%")
    
    if long_variation < 5:
        logger.warning("⚠️ MODEL HAS CONSTANT BIAS - Actions don't depend on input!")
        logger.warning("The model learned 'always go LONG' instead of patterns.")
        return False
    else:
        logger.info("✅ MODEL RESPONDS TO INPUT - Actions vary with features")
        return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    zero_input_test(args.checkpoint, args.device)
