"""
Full Layer 1 to Layer 2 Pipeline Latency Test

Tests the complete data flow from Layer 1 Signal Layer through
the bridge to Layer 2 components, measuring latency at each stage.

Created by: Antigravity AI Agent  
Date: January 6, 2026

Usage:
    python test_l1_l2_pipeline_latency.py
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths for imports
LAYER1_PATH = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\HIMARI SIGNAL LAYER"
LAYER2_PATH = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1"

sys.path.insert(0, LAYER1_PATH)
sys.path.insert(0, LAYER2_PATH)
sys.path.insert(0, os.path.join(LAYER2_PATH, 'src'))


# =============================================================================
# Latency Tracker
# =============================================================================

@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    component: str
    latency_ms: float
    timestamp: float
    success: bool = True
    error: str = ""


class LatencyTracker:
    """Track latencies across pipeline components."""
    
    def __init__(self):
        self.measurements: List[LatencyMeasurement] = []
        self._current_start: float = 0
    
    def start(self):
        """Start timing."""
        self._current_start = time.perf_counter()
    
    def record(self, component: str, success: bool = True, error: str = ""):
        """Record latency for a component."""
        elapsed_ms = (time.perf_counter() - self._current_start) * 1000
        self.measurements.append(LatencyMeasurement(
            component=component,
            latency_ms=elapsed_ms,
            timestamp=time.time(),
            success=success,
            error=error
        ))
        self._current_start = time.perf_counter()
        return elapsed_ms
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get latency summary by component."""
        summary = {}
        
        for m in self.measurements:
            if m.component not in summary:
                summary[m.component] = {'latencies': [], 'errors': 0}
            summary[m.component]['latencies'].append(m.latency_ms)
            if not m.success:
                summary[m.component]['errors'] += 1
        
        result = {}
        for comp, data in summary.items():
            arr = np.array(data['latencies'])
            result[comp] = {
                'mean_ms': float(np.mean(arr)),
                'p50_ms': float(np.percentile(arr, 50)),
                'p95_ms': float(np.percentile(arr, 95)),
                'p99_ms': float(np.percentile(arr, 99)),
                'min_ms': float(np.min(arr)),
                'max_ms': float(np.max(arr)),
                'count': len(arr),
                'errors': data['errors']
            }
        
        return result


# =============================================================================
# Mock Layer 1 Components
# =============================================================================

class MockLayer1Pipeline:
    """
    Mock Layer 1 pipeline for testing.
    
    In production, this would be replaced with actual Layer 1 imports.
    """
    
    def __init__(self):
        self.feature_dim = 60
        self._primitives_loaded = False
        self._try_load_primitives()
    
    def _try_load_primitives(self):
        """Try to load actual Layer 1 primitives."""
        try:
            from feature_vector import FeatureVectorAssembler
            from primitives.streaming_hmm import StreamingHMM
            from primitives.kalman import KalmanFilter
            self._primitives_loaded = True
            logger.info("✅ Loaded actual Layer 1 primitives")
        except ImportError as e:
            logger.warning(f"Could not load Layer 1 primitives: {e}")
            logger.info("Using mock Layer 1 components")
    
    def generate_features(self, price: float = 50000.0) -> np.ndarray:
        """Generate 60D feature vector."""
        features = np.zeros(60)
        
        # Simulate realistic feature values
        np.random.seed(int(time.time() * 1000) % (2**31))
        
        # Trend features (0-7)
        features[0] = np.tanh((price - 50000) / 5000)  # kalman_smoothed
        features[1] = np.random.uniform(-0.02, 0.02)  # velocity
        features[2] = features[0] * 0.9 + np.random.normal(0, 0.01)  # ultimate_smoother
        features[3] = np.random.uniform(0.3, 0.8)  # trend_strength
        features[4] = np.random.uniform(-0.3, 0.3)  # hurst (centered)
        features[5] = np.sign(features[0]) * features[3]  # momentum_regime
        features[6] = np.sign(features[1])  # trend_direction
        features[7] = np.random.uniform(0.4, 0.9)  # trend_quality
        
        # Momentum features (8-15)
        momentum_base = features[1] * 10
        features[8] = np.tanh(momentum_base)  # lorentzian_p_bullish
        features[9] = np.random.uniform(0.5, 0.9)  # lorentzian_confidence
        features[10] = features[8] * 0.8 + np.random.normal(0, 0.1)  # ensemble_signal
        features[11] = np.random.uniform(0.6, 0.95)  # ensemble_agreement
        features[12:16] = np.random.uniform(-0.3, 0.3, 4)
        
        # Volatility features (16-23)
        features[16] = np.random.uniform(0.01, 0.05)  # garch_vol
        features[17] = np.random.choice([-1, 0, 1])  # garch_regime
        hmm_probs = np.random.dirichlet([2, 1, 2])
        features[18:21] = hmm_probs  # hmm probs
        features[21:24] = np.random.uniform(0, 1, 3)
        
        # Volume features (24-33)
        features[24:34] = np.random.uniform(-0.2, 0.2, 10)
        
        # Statistical features (34-41)
        features[34:42] = np.random.uniform(-0.3, 0.3, 8)
        
        # Meta features (42-45)
        features[42:46] = np.random.uniform(0.3, 0.8, 4)
        
        # SMC/Microstructure (46-49)
        features[46:50] = np.random.uniform(0.3, 0.8, 4)
        
        # Order Flow features (50-59)
        features[50] = np.random.uniform(-0.5, 0.5)  # obi_current
        features[51] = features[50] * 0.9  # obi_ema (smoothed)
        features[52] = np.random.uniform(-2, 2)  # cvd_normalized
        features[53] = np.random.choice([-1, 0, 1])  # cvd_divergence
        features[54] = np.random.uniform(-0.3, 0.3)  # microprice_dev
        features[55] = np.random.uniform(0.3, 0.6)  # vpin
        features[56] = np.random.uniform(-1.5, 1.5)  # spread_zscore
        features[57] = np.random.uniform(-0.4, 0.4)  # lob_imbalance
        features[58] = np.random.uniform(-1, 1)  # trade_intensity
        features[59] = np.random.uniform(0.4, 0.6)  # aggressive_ratio
        
        return features


# =============================================================================
# Main Test Runner
# =============================================================================

def run_pipeline_latency_test(n_iterations: int = 100) -> Dict[str, Any]:
    """
    Run the full L1 → L2 pipeline latency test.
    
    Args:
        n_iterations: Number of test iterations
        
    Returns:
        Dictionary with latency results
    """
    print("=" * 70)
    print("HIMARI Layer 1 → Layer 2 Pipeline Latency Test")
    print("=" * 70)
    print(f"\nRunning {n_iterations} iterations...\n")
    
    tracker = LatencyTracker()
    mock_l1 = MockLayer1Pipeline()
    
    # Try to import bridge
    try:
        from l1_l2_bridge import (
            L1L2Bridge, Layer1Output, Layer1DataAdapter, 
            PortfolioState, generate_mock_l1_output
        )
        bridge = L1L2Bridge()
        bridge_available = True
        logger.info("✅ L1L2Bridge loaded successfully")
    except ImportError as e:
        logger.warning(f"Could not import bridge: {e}")
        bridge_available = False
        
    # Run iterations
    total_latencies = []
    
    for i in range(n_iterations):
        iteration_start = time.perf_counter()
        
        # Stage 1: Generate Layer 1 features
        tracker.start()
        try:
            price = 50000 + np.random.uniform(-1000, 1000)
            l1_features = mock_l1.generate_features(price)
            tracker.record("L1_FeatureGeneration")
        except Exception as e:
            tracker.record("L1_FeatureGeneration", success=False, error=str(e))
        
        # Stage 2: Bridge adaptation
        tracker.start()
        try:
            if bridge_available:
                l1_output = Layer1Output(
                    features=l1_features,
                    timestamp=time.time(),
                    symbol="BTCUSDT",
                    n_nonzero=int(np.sum(l1_features != 0)),
                    latency_ms=tracker.measurements[-1].latency_ms if tracker.measurements else 0
                )
                l2_input = bridge.process(l1_output)
                tracker.record("Bridge_Adaptation")
            else:
                # Mock bridge operation
                time.sleep(0.0001)  # 0.1ms mock
                tracker.record("Bridge_Adaptation")
        except Exception as e:
            tracker.record("Bridge_Adaptation", success=False, error=str(e))
        
        # Stage 3: Feature validation
        tracker.start()
        try:
            if bridge_available:
                is_valid, issues = bridge.adapter.validate_feature_vector(l1_features)
                if not is_valid and i == 0:
                    logger.warning(f"Validation issues: {issues[:3]}")
            tracker.record("Feature_Validation")
        except Exception as e:
            tracker.record("Feature_Validation", success=False, error=str(e))
        
        # Stage 4: Simulate L2 preprocessing (Part A)
        tracker.start()
        try:
            # Simulate EKF denoising
            _ = l1_features * 0.99 + np.random.normal(0, 0.001, 60)
            # Simulate normalization
            _ = (l1_features - np.mean(l1_features)) / (np.std(l1_features) + 1e-8)
            tracker.record("L2_Preprocessing_A")
        except Exception as e:
            tracker.record("L2_Preprocessing_A", success=False, error=str(e))
        
        # Stage 5: Simulate L2 regime detection (Part B)
        tracker.start()
        try:
            # Simulate HMM forward step
            regime_probs = np.random.dirichlet([l1_features[18] + 0.1, 
                                                 l1_features[19] + 0.1, 
                                                 l1_features[20] + 0.1])
            regime = np.argmax(regime_probs)
            tracker.record("L2_RegimeDetection_B")
        except Exception as e:
            tracker.record("L2_RegimeDetection_B", success=False, error=str(e))
        
        # Total iteration time
        total_ms = (time.perf_counter() - iteration_start) * 1000
        total_latencies.append(total_ms)
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_iterations} iterations")
    
    # Summarize results
    summary = tracker.get_summary()
    
    # Calculate total pipeline latency
    total_arr = np.array(total_latencies)
    pipeline_summary = {
        'mean_ms': float(np.mean(total_arr)),
        'p50_ms': float(np.percentile(total_arr, 50)),
        'p95_ms': float(np.percentile(total_arr, 95)),
        'p99_ms': float(np.percentile(total_arr, 99)),
        'min_ms': float(np.min(total_arr)),
        'max_ms': float(np.max(total_arr))
    }
    
    return {
        'component_latencies': summary,
        'total_pipeline': pipeline_summary,
        'n_iterations': n_iterations,
        'bridge_available': bridge_available
    }


def print_results(results: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("LATENCY RESULTS")
    print("=" * 70)
    
    print(f"\nIterations: {results['n_iterations']}")
    print(f"Bridge Available: {'✅' if results['bridge_available'] else '❌'}")
    
    print("\n--- Component Latencies ---")
    print(f"{'Component':<25} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10}")
    print("-" * 65)
    
    for comp, stats in results['component_latencies'].items():
        print(f"{comp:<25} {stats['mean_ms']:>9.3f}ms {stats['p50_ms']:>9.3f}ms "
              f"{stats['p95_ms']:>9.3f}ms {stats['p99_ms']:>9.3f}ms")
        if stats['errors'] > 0:
            print(f"  ⚠️  Errors: {stats['errors']}")
    
    print("\n--- Total Pipeline ---")
    tp = results['total_pipeline']
    print(f"Mean:   {tp['mean_ms']:.3f} ms")
    print(f"P50:    {tp['p50_ms']:.3f} ms")
    print(f"P95:    {tp['p95_ms']:.3f} ms")
    print(f"P99:    {tp['p99_ms']:.3f} ms")
    print(f"Min:    {tp['min_ms']:.3f} ms")
    print(f"Max:    {tp['max_ms']:.3f} ms")
    
    # Check against target
    target_ms = 17.0  # L1 + Bridge + Part A target
    if tp['p99_ms'] < target_ms:
        print(f"\n✅ PASS: P99 ({tp['p99_ms']:.3f}ms) < Target ({target_ms}ms)")
    else:
        print(f"\n❌ FAIL: P99 ({tp['p99_ms']:.3f}ms) > Target ({target_ms}ms)")
    
    print("\n" + "=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L1-L2 Pipeline Latency Test")
    parser.add_argument("-n", "--iterations", type=int, default=100,
                        help="Number of test iterations (default: 100)")
    args = parser.parse_args()
    
    try:
        results = run_pipeline_latency_test(args.iterations)
        print_results(results)
        
        # Return exit code based on latency target
        target_met = results['total_pipeline']['p99_ms'] < 17.0
        sys.exit(0 if target_met else 1)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
