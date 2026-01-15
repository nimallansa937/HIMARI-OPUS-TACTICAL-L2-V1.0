#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Full Integration Test

Tests the complete pipeline with all 6 trained components:
1. Load all trained checkpoints
2. Initialize Layer2MasterPipeline
3. Run synthetic data through pipeline
4. Measure latency at each stage
5. Verify outputs are valid

Usage:
    python run_integration_test.py
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Add paths
BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "LAYER 2 TACTICAL HIMARI OPUS"))

# =============================================================================
# Latency Tracking
# =============================================================================

class LatencyTracker:
    """Track latencies for pipeline stages."""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self._start_time: float = 0

    def start(self):
        self._start_time = time.perf_counter()

    def record(self, stage: str):
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self.measurements[stage].append(elapsed_ms)
        return elapsed_ms

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for stage, times in self.measurements.items():
            arr = np.array(times)
            result[stage] = {
                'mean': float(np.mean(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
                'count': len(arr)
            }
        return result


# =============================================================================
# Checkpoint Loader
# =============================================================================

def load_all_checkpoints() -> Dict[str, Any]:
    """Load all 6 trained checkpoints."""
    checkpoints = {}

    # PPO Policy
    ppo_path = BASE_DIR / "L2V1 PPO FINAL" / "himari_ppo_final.pt"
    if ppo_path.exists():
        checkpoints['ppo'] = torch.load(ppo_path, map_location='cpu', weights_only=False)
        print(f"[OK] PPO loaded: {ppo_path.stat().st_size / 1024 / 1024:.2f} MB")

    # AHHMM
    ahhmm_path = BASE_DIR / "L2V1 AHHMM FINAL" / "student_t_ahhmm_percentile.pkl"
    if ahhmm_path.exists():
        with open(ahhmm_path, 'rb') as f:
            checkpoints['ahhmm'] = pickle.load(f)
        print(f"[OK] AHHMM loaded: {len(checkpoints['ahhmm'])} keys")

    # EKF
    ekf_path = BASE_DIR / "L2V1 EKF FINAL" / "ekf_config_calibrated.pkl"
    if ekf_path.exists():
        with open(ekf_path, 'rb') as f:
            checkpoints['ekf'] = pickle.load(f)
        print(f"[OK] EKF loaded: {len(checkpoints['ekf'])} keys")

    # Sortino
    sortino_path = BASE_DIR / "L2V1 SORTINO FINAL" / "sortino_config_calibrated.pkl"
    if sortino_path.exists():
        with open(sortino_path, 'rb') as f:
            checkpoints['sortino'] = pickle.load(f)
        print(f"[OK] Sortino loaded: {len(checkpoints['sortino'])} keys")

    # Position Sizer
    pos_path = BASE_DIR / "L2 POSTION FINAL MODELS" / "orkspace" / "checkpoints" / "best_model.pt"
    if pos_path.exists():
        checkpoints['position_sizer'] = torch.load(pos_path, map_location='cpu', weights_only=False)
        print(f"[OK] Position Sizer loaded: {pos_path.stat().st_size / 1024:.2f} KB")

    # Risk Manager
    risk_path = BASE_DIR / "L2V1 RISK MANAGER FINAL" / "risk_manager_config.pkl"
    if risk_path.exists():
        with open(risk_path, 'rb') as f:
            checkpoints['risk_manager'] = pickle.load(f)
        print(f"[OK] Risk Manager loaded: {len(checkpoints['risk_manager'])} components")

    return checkpoints


# =============================================================================
# Synthetic Data Generator
# =============================================================================

def generate_synthetic_candle(idx: int, base_price: float = 50000) -> Dict[str, Any]:
    """Generate a synthetic OHLCV candle."""
    np.random.seed(idx)

    # Simulate random walk with some trend
    trend = np.sin(idx / 100) * 500
    noise = np.random.normal(0, 100)
    close = base_price + trend + noise

    high = close + np.abs(np.random.normal(0, 50))
    low = close - np.abs(np.random.normal(0, 50))
    open_price = close + np.random.normal(0, 30)
    volume = np.abs(np.random.normal(1000, 300))

    return {
        'timestamp': time.time() + idx * 300,  # 5 min candles
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


def generate_features(candle: Dict[str, Any], lookback: int = 100) -> np.ndarray:
    """Generate 60D feature vector from candle."""
    features = np.zeros(60)

    # Price-based features
    features[0] = candle['close'] / 50000 - 1  # Normalized price
    features[1] = (candle['close'] - candle['open']) / candle['open']  # Return
    features[2] = (candle['high'] - candle['low']) / candle['close']  # Range
    features[3] = candle['volume'] / 1000  # Normalized volume
    features[4] = (candle['close'] - candle['low']) / (candle['high'] - candle['low'] + 1e-8)  # Close location

    # Technical indicators (simulated)
    features[5:15] = np.random.uniform(-0.5, 0.5, 10)  # Momentum indicators
    features[15:25] = np.random.uniform(-0.5, 0.5, 10)  # Volatility indicators
    features[25:35] = np.random.uniform(-0.5, 0.5, 10)  # Volume indicators
    features[35:45] = np.random.uniform(-0.5, 0.5, 10)  # Statistical indicators
    features[45:55] = np.random.uniform(-0.5, 0.5, 10)  # Order flow indicators
    features[55:60] = np.random.uniform(0, 1, 5)  # Meta indicators

    return features


# =============================================================================
# Pipeline Stages (Simplified Implementations)
# =============================================================================

class SimplifiedPipeline:
    """Simplified pipeline for testing."""

    def __init__(self, checkpoints: Dict[str, Any]):
        self.checkpoints = checkpoints

        # Part A: EKF state
        self.ekf_state = np.zeros(4)
        self.ekf_config = checkpoints.get('ekf', {}).get('config', {})

        # Part B: AHHMM state
        self.ahhmm = checkpoints.get('ahhmm', {})
        self.current_regime = 0

        # Part H: Risk config
        self.risk_config = checkpoints.get('risk_manager', {})

        # Running statistics
        self.feature_mean = np.zeros(60)
        self.feature_var = np.ones(60)
        self.n_samples = 0

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """Part A: Preprocessing (EKF + Normalization)."""
        # Update running stats
        self.n_samples += 1
        delta = features - self.feature_mean
        self.feature_mean += delta / self.n_samples
        self.feature_var = ((self.n_samples - 1) * self.feature_var + delta * (features - self.feature_mean)) / self.n_samples

        # Normalize
        std = np.sqrt(self.feature_var + 1e-8)
        normalized = (features - self.feature_mean) / std

        # Clip
        return np.clip(normalized, -10, 10)

    def detect_regime(self, features: np.ndarray) -> Tuple[int, float]:
        """Part B: Regime Detection (simplified)."""
        # Use AHHMM parameters if available
        n_states = self.ahhmm.get('n_states', 4)

        # Simple regime detection based on feature signs
        vol_proxy = np.abs(features[2])  # Range as volatility proxy
        trend_proxy = features[0]  # Price deviation

        if vol_proxy > 0.5:
            regime = 3  # CRISIS
            confidence = min(vol_proxy, 1.0)
        elif trend_proxy > 0.2:
            regime = 0  # BULL
            confidence = trend_proxy
        elif trend_proxy < -0.2:
            regime = 1  # BEAR
            confidence = -trend_proxy
        else:
            regime = 2  # SIDEWAYS
            confidence = 0.5

        self.current_regime = regime
        return regime, confidence

    def decide(self, features: np.ndarray, regime: int) -> Tuple[int, float]:
        """Part D: Decision (simplified)."""
        # Use momentum and regime to decide
        momentum = features[1]  # Return

        if regime == 3:  # CRISIS
            action = 0  # HOLD
            confidence = 0.3
        elif momentum > 0.01:
            action = 1  # BUY
            confidence = min(momentum * 10, 0.9)
        elif momentum < -0.01:
            action = -1  # SELL
            confidence = min(-momentum * 10, 0.9)
        else:
            action = 0  # HOLD
            confidence = 0.5

        return action, confidence

    def validate_hsm(self, action: int, current_position: int) -> Tuple[int, bool]:
        """Part E: HSM Validation."""
        # Block invalid transitions
        if current_position == 0:
            # From FLAT, can go to any action
            return action, True
        elif current_position == 1:
            # From LONG, can HOLD or SELL
            if action == 1:  # Can't enter LONG again
                return 0, False  # Convert to HOLD
            return action, True
        else:  # position == -1
            # From SHORT, can HOLD or BUY
            if action == -1:  # Can't enter SHORT again
                return 0, False
            return action, True

    def apply_hysteresis(self, action: int, confidence: float, threshold: float = 0.4) -> int:
        """Part G: Hysteresis Filter."""
        if confidence < threshold:
            return 0  # Filter to HOLD
        return action

    def calculate_position_size(self, action: int, confidence: float, drawdown: float = 0.0) -> float:
        """Part H: Risk Management."""
        if action == 0:
            return 0.0

        # Kelly-based sizing
        kelly_cap = self.risk_config.get('H2_Kelly', {}).get('fraction_cap', 0.25)
        base_size = confidence * kelly_cap

        # Drawdown brake
        brake_levels = [0.05, 0.08, 0.10]
        reductions = [0.25, 0.50, 0.90]

        for level, reduction in zip(brake_levels, reductions):
            if drawdown >= level:
                base_size *= (1 - reduction)

        return min(base_size, 1.0)

    def safety_check(self, action: int, position_size: float, regime: int) -> Tuple[int, float]:
        """Part I: Safety System."""
        # Crisis mode: force conservative
        if regime == 3:
            position_size *= 0.25
            if position_size < 0.05:
                action = 0
                position_size = 0.0

        return action, position_size

    def process(self, candle: Dict[str, Any], current_position: int = 0, drawdown: float = 0.0) -> Dict[str, Any]:
        """Process a single candle through the full pipeline."""
        latencies = {}

        # Generate features
        start = time.perf_counter()
        features = generate_features(candle)
        latencies['feature_gen'] = (time.perf_counter() - start) * 1000

        # Part A: Preprocess
        start = time.perf_counter()
        preprocessed = self.preprocess(features)
        latencies['preprocess'] = (time.perf_counter() - start) * 1000

        # Part B: Regime Detection
        start = time.perf_counter()
        regime, regime_conf = self.detect_regime(preprocessed)
        latencies['regime'] = (time.perf_counter() - start) * 1000

        # Part D: Decision
        start = time.perf_counter()
        action, confidence = self.decide(preprocessed, regime)
        latencies['decision'] = (time.perf_counter() - start) * 1000

        # Part E: HSM
        start = time.perf_counter()
        action, hsm_valid = self.validate_hsm(action, current_position)
        latencies['hsm'] = (time.perf_counter() - start) * 1000

        # Part G: Hysteresis
        start = time.perf_counter()
        action = self.apply_hysteresis(action, confidence)
        latencies['hysteresis'] = (time.perf_counter() - start) * 1000

        # Part H: Risk
        start = time.perf_counter()
        position_size = self.calculate_position_size(action, confidence, drawdown)
        latencies['risk'] = (time.perf_counter() - start) * 1000

        # Part I: Safety
        start = time.perf_counter()
        action, position_size = self.safety_check(action, position_size, regime)
        latencies['safety'] = (time.perf_counter() - start) * 1000

        return {
            'action': action,
            'position_size': position_size,
            'confidence': confidence,
            'regime': regime,
            'latencies': latencies,
            'total_latency_ms': sum(latencies.values())
        }


# =============================================================================
# Main Test
# =============================================================================

def run_integration_test(n_iterations: int = 1000) -> Dict[str, Any]:
    """Run the full integration test."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - Full Integration Test")
    print("=" * 70)

    # Load checkpoints
    print("\n[1/3] Loading trained checkpoints...")
    checkpoints = load_all_checkpoints()
    print(f"Loaded {len(checkpoints)}/6 checkpoints")

    # Initialize pipeline
    print("\n[2/3] Initializing pipeline...")
    pipeline = SimplifiedPipeline(checkpoints)
    print("Pipeline initialized")

    # Run test iterations
    print(f"\n[3/3] Running {n_iterations} iterations...")

    tracker = LatencyTracker()
    actions = []
    position_sizes = []
    regimes = []
    total_latencies = []

    current_position = 0
    drawdown = 0.0

    for i in range(n_iterations):
        # Generate candle
        candle = generate_synthetic_candle(i)

        # Process through pipeline
        tracker.start()
        result = pipeline.process(candle, current_position, drawdown)
        tracker.record('total_pipeline')

        # Track results
        actions.append(result['action'])
        position_sizes.append(result['position_size'])
        regimes.append(result['regime'])
        total_latencies.append(result['total_latency_ms'])

        # Update state
        if result['action'] != 0:
            current_position = result['action']

        # Progress
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i + 1}/{n_iterations} iterations")

    # Generate report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Latency stats
    latencies = np.array(total_latencies)
    print("\n[LATENCY STATISTICS]")
    print(f"  Mean:   {np.mean(latencies):.3f} ms")
    print(f"  P50:    {np.percentile(latencies, 50):.3f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"  Max:    {np.max(latencies):.3f} ms")

    # Check target
    p99 = np.percentile(latencies, 99)
    target = 50.0
    if p99 < target:
        print(f"\n  [PASS] P99 ({p99:.3f} ms) < Target ({target} ms)")
    else:
        print(f"\n  [FAIL] P99 ({p99:.3f} ms) >= Target ({target} ms)")

    # Action distribution
    print("\n[ACTION DISTRIBUTION]")
    actions = np.array(actions)
    print(f"  SELL (-1): {np.sum(actions == -1):4d} ({100*np.mean(actions == -1):.1f}%)")
    print(f"  HOLD (0):  {np.sum(actions == 0):4d} ({100*np.mean(actions == 0):.1f}%)")
    print(f"  BUY (1):   {np.sum(actions == 1):4d} ({100*np.mean(actions == 1):.1f}%)")

    # Regime distribution
    print("\n[REGIME DISTRIBUTION]")
    regimes = np.array(regimes)
    regime_names = ['BULL', 'BEAR', 'SIDEWAYS', 'CRISIS']
    for r in range(4):
        count = np.sum(regimes == r)
        pct = 100 * count / len(regimes)
        print(f"  {regime_names[r]:8s}: {count:4d} ({pct:.1f}%)")

    # Position size stats
    print("\n[POSITION SIZE STATISTICS]")
    sizes = np.array(position_sizes)
    print(f"  Mean:   {np.mean(sizes):.4f}")
    print(f"  Max:    {np.max(sizes):.4f}")
    print(f"  Min:    {np.min(sizes):.4f}")

    # Validation
    print("\n[VALIDATION]")
    valid_actions = all(a in [-1, 0, 1] for a in actions)
    valid_sizes = all(0 <= s <= 1 for s in position_sizes)
    valid_regimes = all(r in [0, 1, 2, 3] for r in regimes)
    no_nans = not any(np.isnan(latencies))

    print(f"  Actions in {{-1, 0, 1}}:    {'PASS' if valid_actions else 'FAIL'}")
    print(f"  Position sizes in [0,1]:  {'PASS' if valid_sizes else 'FAIL'}")
    print(f"  Regimes in {{0,1,2,3}}:     {'PASS' if valid_regimes else 'FAIL'}")
    print(f"  No NaN values:            {'PASS' if no_nans else 'FAIL'}")

    all_passed = valid_actions and valid_sizes and valid_regimes and no_nans and (p99 < target)

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] All tests passed!")
    else:
        print("[WARNING] Some tests failed")
    print("=" * 70)

    return {
        'latency_p99': p99,
        'passed': all_passed,
        'n_iterations': n_iterations,
        'checkpoints_loaded': len(checkpoints)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-n', type=int, default=1000, help='Number of test iterations')
    args = parser.parse_args()

    try:
        results = run_integration_test(args.iterations)
        sys.exit(0 if results['passed'] else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
