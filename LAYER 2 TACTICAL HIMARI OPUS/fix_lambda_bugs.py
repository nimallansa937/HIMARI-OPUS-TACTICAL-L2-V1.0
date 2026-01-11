#!/usr/bin/env python3
"""
Auto-fix script for Lambda Labs - Fixes Sharpe/Sortino bugs
Run this on Lambda to patch the training code
"""

import os
import shutil
from pathlib import Path

def fix_sortino_reward():
    """Fix sortino_reward.py - remove clipping bugs"""

    file_path = Path("~/HIMARI-OPUS-TACTICAL-L2-V1.0/src/training/sortino_reward.py").expanduser()

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    # Backup original
    backup_path = file_path.with_suffix('.py.backup')
    shutil.copy(file_path, backup_path)
    print(f"✅ Backed up to: {backup_path}")

    # Read file
    with open(file_path, 'r') as f:
        content = f.read()

    # Fix 1: Remove Sharpe clipping (line ~98)
    content = content.replace(
        'return float(np.clip(sharpe, -10, 10))  # Clip extremes',
        'return float(sharpe)  # Return actual value without clipping'
    )

    # Fix 2: Remove Sortino 10.0 edge cases (lines ~112, 116)
    content = content.replace(
        'return 10.0 if mean_ret > 0 else 0.0',
        'return 0.0'
    )

    # Fix 3: Remove Sortino clipping (line ~120)
    content = content.replace(
        'return float(np.clip(sortino, -10, 10))',
        'return float(sortino)  # Return actual value without clipping'
    )

    # Write fixed file
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✅ Fixed: {file_path}")
    return True


def fix_trainer_logging():
    """Fix transformer_a2c_trainer.py - add validation logging"""

    file_path = Path("~/HIMARI-OPUS-TACTICAL-L2-V1.0/src/training/transformer_a2c_trainer.py").expanduser()

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    # Backup original
    backup_path = file_path.with_suffix('.py.backup')
    shutil.copy(file_path, backup_path)
    print(f"✅ Backed up to: {backup_path}")

    # Read file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the validation function and add logging
    modified = False
    for i, line in enumerate(lines):
        # Look for "val_sharpe = reward_fn.get_episode_sharpe()"
        if 'val_sharpe = reward_fn.get_episode_sharpe()' in line and not modified:
            # Insert logging after this line
            indent = '        '
            logging_code = f'''
{indent}# DEBUG: Log validation metrics
{indent}returns = np.array(reward_fn._returns_buffer)
{indent}logger.info(
{indent}    f"Validation metrics: returns_mean={{np.mean(returns):.6f}}, "
{indent}    f"returns_std={{np.std(returns):.6f}}, n_samples={{len(returns)}}, "
{indent}    f"total_return={{reward_fn.get_total_return():.4f}}, "
{indent}    f"max_dd={{reward_fn.get_max_drawdown():.4f}}"
{indent})

'''
            lines.insert(i + 1, logging_code)
            modified = True
            break

    if modified:
        # Write fixed file
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"✅ Fixed: {file_path}")
        return True
    else:
        print(f"⚠️  Could not find insertion point in {file_path}")
        return False


def main():
    print("=" * 60)
    print("🔧 Auto-fixing Lambda Labs Training Bugs")
    print("=" * 60)

    success = True

    # Fix sortino_reward.py
    print("\n📝 Fixing sortino_reward.py...")
    if not fix_sortino_reward():
        success = False

    # Fix transformer_a2c_trainer.py
    print("\n📝 Fixing transformer_a2c_trainer.py...")
    if not fix_trainer_logging():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✅ All fixes applied successfully!")
        print("\nNext steps:")
        print("1. Re-run training:")
        print("   python scripts/train_transformer_a2c.py --data ~/data/btc_5min.pkl --max_steps 100000 --device cuda --output ./output/btc_real_train")
        print("\n2. You should now see realistic Sharpe values (0.3-1.5)")
        print("   instead of 10.0!")
    else:
        print("❌ Some fixes failed - check errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
