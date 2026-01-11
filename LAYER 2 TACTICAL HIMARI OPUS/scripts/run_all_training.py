#!/usr/bin/env python3
"""
HIMARI Layer 2 - Run All Models Training
Sequentially trains all 5 models with W&B logging.

Usage:
    python run_all_training.py --wandb-entity charithliyanage52-himari
    
Models trained in order:
1. BaselineMLP - Simple classification baseline (~10 min)
2. CQL - Conservative Q-Learning (~15 min)
3. CGDT - Decision Transformer (~20 min)
4. FLAG-TRADER - Large transformer with LoRA (~30 min)
5. PPO-LSTM - Online RL (~20 min)
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Model training configurations
MODELS = [
    {
        'name': 'baseline',
        'display': 'BaselineMLP',
        'epochs': 50,
        'desc': 'Simple MLP baseline',
        'est_time': '1 hour'
    },
    {
        'name': 'cql',
        'display': 'CQL',
        'epochs': 100,
        'desc': 'Conservative Q-Learning (Offline RL)',
        'est_time': '3 hours'
    },
    {
        'name': 'cgdt',
        'display': 'CGDT',
        'epochs': 50,
        'desc': 'Critic-Guided Decision Transformer',
        'est_time': '5 hours'
    },
    {
        'name': 'flag-trader',
        'display': 'FLAG-TRADER',
        'epochs': 30,
        'desc': 'Large Transformer with LoRA (135M params)',
        'est_time': '8 hours'
    },
    {
        'name': 'ppo',
        'display': 'PPO-LSTM',
        'env_episodes': 1000,
        'desc': 'Proximal Policy Optimization with LSTM',
        'est_time': '10 hours'
    },
]


def run_model_training(
    model_config: dict,
    data_dir: str,
    checkpoint_dir: str,
    wandb_entity: str = None,
    device: str = 'cuda'
) -> dict:
    """Run training for a single model."""
    
    model_name = model_config['name']
    display_name = model_config['display']
    
    logger.info("=" * 70)
    logger.info(f"[START] Training {display_name}")
    logger.info(f"        {model_config['desc']}")
    logger.info(f"        Estimated time: {model_config['est_time']}")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable,
        'scripts/train_all_models.py',
        '--model', model_name,
        '--data-dir', data_dir,
        '--checkpoint-dir', checkpoint_dir,
        '--device', device,
    ]

    # Add model-specific args
    if model_name == 'ppo':
        # PPO uses episodes, not epochs
        cmd.extend(['--env-episodes', str(model_config.get('env_episodes', 1000))])
    else:
        # All other models use epochs
        cmd.extend(['--epochs', str(model_config.get('epochs', 50))])

    if model_name == 'cgdt':
        cmd.extend(['--context-length', '64'])

    if model_name == 'flag-trader':
        cmd.extend(['--context-length', '256'])
        cmd.extend(['--lora-rank', '16'])
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        success = True
        error = None
    except subprocess.CalledProcessError as e:
        success = False
        error = str(e)
        logger.error(f"[ERROR] {display_name} training failed: {error}")
    except Exception as e:
        success = False
        error = str(e)
        logger.error(f"[ERROR] {display_name} training exception: {error}")
    
    elapsed = time.time() - start_time
    
    logger.info("-" * 70)
    if success:
        logger.info(f"[DONE] {display_name} completed in {elapsed/60:.1f} minutes")
    else:
        logger.info(f"[FAIL] {display_name} failed after {elapsed/60:.1f} minutes")
    logger.info("-" * 70)
    logger.info("")
    
    return {
        'model': display_name,
        'success': success,
        'elapsed_seconds': elapsed,
        'error': error
    }


def main():
    parser = argparse.ArgumentParser(description="Run all HIMARI model trainings")
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (optional)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--models', nargs='+', default=None, 
                        help='Specific models to train (default: all)')
    parser.add_argument('--skip', nargs='+', default=[], 
                        help='Models to skip')
    
    args = parser.parse_args()
    
    # Header
    logger.info("")
    logger.info("*" * 70)
    logger.info("*" + " " * 68 + "*")
    logger.info("*" + "      HIMARI LAYER 2 - AUTOMATED MODEL TRAINING".center(68) + "*")
    logger.info("*" + " " * 68 + "*")
    logger.info("*" * 70)
    logger.info("")
    logger.info(f"   Data Directory:  {args.data_dir}")
    logger.info(f"   Checkpoint Dir:  {args.checkpoint_dir}")
    logger.info(f"   Device:          {args.device}")
    logger.info(f"   W&B Entity:      {args.wandb_entity or 'disabled'}")
    logger.info("")
    
    # Filter models
    models_to_train = []
    for model in MODELS:
        if args.models and model['name'] not in args.models:
            continue
        if model['name'] in args.skip:
            logger.info(f"   [SKIP] {model['display']}")
            continue
        models_to_train.append(model)
    
    logger.info(f"   Training {len(models_to_train)} models:")
    for i, m in enumerate(models_to_train, 1):
        logger.info(f"     {i}. {m['display']} ({m['est_time']})")
    logger.info("")
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Train all models
    results = []
    total_start = time.time()
    
    for i, model_config in enumerate(models_to_train, 1):
        logger.info(f"[{i}/{len(models_to_train)}] Training {model_config['display']}...")
        
        result = run_model_training(
            model_config=model_config,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            wandb_entity=args.wandb_entity,
            device=args.device
        )
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    logger.info("")
    logger.info("*" * 70)
    logger.info("*" + "                     TRAINING SUMMARY".center(68) + "*")
    logger.info("*" * 70)
    logger.info("")
    logger.info(f"   Total Training Time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    logger.info("")
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    logger.info(f"   Results: {success_count} succeeded, {fail_count} failed")
    logger.info("")
    
    for result in results:
        status = "[OK]  " if result['success'] else "[FAIL]"
        elapsed = result['elapsed_seconds']
        logger.info(f"   {status} {result['model']:20s} - {elapsed/60:.1f} min")
        if result['error']:
            logger.info(f"          Error: {result['error'][:50]}...")
    
    logger.info("")
    logger.info(f"   Checkpoints saved to: {args.checkpoint_dir}")
    logger.info("")
    logger.info("*" * 70)
    logger.info("*" + "               ALL TRAINING COMPLETE!".center(68) + "*")
    logger.info("*" * 70)
    logger.info("")
    
    # Exit with error if any failed
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
