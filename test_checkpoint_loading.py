#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Checkpoint Loading Test

Tests that all 6 trained model checkpoints load correctly:
1. PPO Policy
2. Student-T AHHMM
3. EKF Denoiser
4. Sortino Shaper
5. Position Sizer
6. Risk Manager

Usage:
    python test_checkpoint_loading.py
"""

import os
import sys
import pickle
import torch
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple

# Base directory
BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")

# Checkpoint paths (relative to BASE_DIR)
CHECKPOINTS = {
    "PPO Policy": {
        "path": "L2V1 PPO FINAL/himari_ppo_final.pt",
        "type": "torch",
        "expected_keys": ["model_state_dict"]  # Common keys to verify
    },
    "PPO Best Model": {
        "path": "L2V1 PPO FINAL/best_model.pt",
        "type": "torch",
        "expected_keys": []
    },
    "Student-T AHHMM (Percentile)": {
        "path": "L2V1 AHHMM FINAL/student_t_ahhmm_percentile.pkl",
        "type": "pickle",
        "expected_keys": []
    },
    "Student-T AHHMM (Supervised)": {
        "path": "L2V1 AHHMM FINAL/student_t_ahhmm_supervised.pkl",
        "type": "pickle",
        "expected_keys": []
    },
    "Student-T AHHMM (Trained)": {
        "path": "L2V1 AHHMM FINAL/student_t_ahhmm_trained.pkl",
        "type": "pickle",
        "expected_keys": []
    },
    "EKF Denoiser": {
        "path": "L2V1 EKF FINAL/ekf_config_calibrated.pkl",
        "type": "pickle",
        "expected_keys": []
    },
    "Sortino Shaper": {
        "path": "L2V1 SORTINO FINAL/sortino_config_calibrated.pkl",
        "type": "pickle",
        "expected_keys": []
    },
    "Position Sizer (Best)": {
        "path": "L2 POSTION FINAL MODELS/orkspace/checkpoints/best_model.pt",
        "type": "torch",
        "expected_keys": []
    },
    "Position Sizer (Bounded Delta)": {
        "path": "L2 POSTION FINAL MODELS/orkspace/checkpoints/l3_bounded_delta_final.pt",
        "type": "torch",
        "expected_keys": []
    },
    "Risk Manager": {
        "path": "L2V1 RISK MANAGER FINAL/risk_manager_config.pkl",
        "type": "pickle",
        "expected_keys": []
    },
}


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def load_torch_checkpoint(path: Path) -> Tuple[bool, Dict[str, Any], str]:
    """Load a PyTorch checkpoint and return info."""
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        info = {
            "file_size": format_size(path.stat().st_size),
            "type": "torch"
        }

        if isinstance(checkpoint, dict):
            info["keys"] = list(checkpoint.keys())

            # If it has model_state_dict, analyze it
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                info["n_parameters"] = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                info["layer_shapes"] = {k: list(v.shape) for k, v in list(state_dict.items())[:5]}
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                info["n_parameters"] = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                info["layer_shapes"] = {k: list(v.shape) for k, v in list(state_dict.items())[:5]}
            else:
                # Maybe the checkpoint IS the state dict
                if all(isinstance(v, torch.Tensor) for v in list(checkpoint.values())[:3]):
                    info["n_parameters"] = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
                    info["layer_shapes"] = {k: list(v.shape) for k, v in list(checkpoint.items())[:5]}
        else:
            info["type"] = str(type(checkpoint))

        return True, info, ""
    except Exception as e:
        return False, {}, str(e)


def load_pickle_checkpoint(path: Path) -> Tuple[bool, Dict[str, Any], str]:
    """Load a pickle file and return info."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        info = {
            "file_size": format_size(path.stat().st_size),
            "type": "pickle",
            "data_type": str(type(data).__name__)
        }

        if isinstance(data, dict):
            info["keys"] = list(data.keys())
            # Show sample values
            info["sample_values"] = {}
            for k, v in list(data.items())[:5]:
                if isinstance(v, (int, float, str, bool)):
                    info["sample_values"][k] = v
                elif isinstance(v, (list, tuple)):
                    info["sample_values"][k] = f"{type(v).__name__}[{len(v)}]"
                else:
                    info["sample_values"][k] = str(type(v).__name__)
        elif hasattr(data, '__dict__'):
            info["attributes"] = list(data.__dict__.keys())[:10]

        return True, info, ""
    except Exception as e:
        return False, {}, str(e)


def test_all_checkpoints():
    """Test loading all checkpoints."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - Checkpoint Loading Test")
    print("=" * 70)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print()

    results = {}
    passed = 0
    failed = 0

    for name, config in CHECKPOINTS.items():
        full_path = BASE_DIR / config["path"]
        print(f"\n{'-' * 60}")
        print(f"Testing: {name}")
        print(f"Path: {config['path']}")

        if not full_path.exists():
            print(f"  [FAIL] File not found!")
            results[name] = {"status": "NOT_FOUND", "path": str(full_path)}
            failed += 1
            continue

        if config["type"] == "torch":
            success, info, error = load_torch_checkpoint(full_path)
        else:
            success, info, error = load_pickle_checkpoint(full_path)

        if success:
            print(f"  [PASS] Loaded successfully!")
            print(f"  Size: {info.get('file_size', 'N/A')}")

            if 'keys' in info:
                print(f"  Keys: {info['keys']}")
            if 'n_parameters' in info:
                print(f"  Parameters: {info['n_parameters']:,}")
            if 'layer_shapes' in info:
                print(f"  Sample Layers:")
                for layer, shape in list(info['layer_shapes'].items())[:3]:
                    print(f"    - {layer}: {shape}")
            if 'sample_values' in info:
                print(f"  Sample Values:")
                for k, v in info['sample_values'].items():
                    print(f"    - {k}: {v}")
            if 'attributes' in info:
                print(f"  Attributes: {info['attributes']}")

            results[name] = {"status": "PASS", "info": info}
            passed += 1
        else:
            print(f"  [FAIL] Load error: {error}")
            results[name] = {"status": "FAIL", "error": error}
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(CHECKPOINTS)}")
    print(f"Failed: {failed}/{len(CHECKPOINTS)}")

    if failed == 0:
        print("\n[SUCCESS] All checkpoints loaded successfully!")
    else:
        print("\n[WARNING] Some checkpoints failed to load:")
        for name, result in results.items():
            if result["status"] != "PASS":
                print(f"  - {name}: {result['status']}")

    return passed, failed, results


if __name__ == "__main__":
    try:
        passed, failed, results = test_all_checkpoints()
        sys.exit(0 if failed == 0 else 1)
    except Exception as e:
        print(f"\n[ERROR] Test script failed: {e}")
        traceback.print_exc()
        sys.exit(1)
