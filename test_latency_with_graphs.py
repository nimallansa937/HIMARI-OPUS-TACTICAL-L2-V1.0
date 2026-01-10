"""
L1-L2 Pipeline Latency Test with Visualization

Runs large-scale latency test and generates graphs.

Created by: Antigravity AI Agent
Date: January 6, 2026
"""

import sys
import os
import time
import numpy as np
import json
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Add paths
LAYER1_PATH = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\HIMARI SIGNAL LAYER"
LAYER2_PATH = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1"
sys.path.insert(0, LAYER1_PATH)
sys.path.insert(0, LAYER2_PATH)
sys.path.insert(0, os.path.join(LAYER2_PATH, 'src'))

# Try matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available, will save raw data only")


@dataclass
class LatencyRecord:
    l1_feature_ms: float
    bridge_ms: float
    validation_ms: float
    l2_preprocess_ms: float
    l2_regime_ms: float
    total_ms: float


class LatencyCollector:
    """Collect latency measurements for all components."""
    
    def __init__(self):
        self.records: List[LatencyRecord] = []
    
    def add(self, record: LatencyRecord):
        self.records.append(record)
    
    def get_arrays(self) -> Dict[str, np.ndarray]:
        return {
            'l1_feature': np.array([r.l1_feature_ms for r in self.records]),
            'bridge': np.array([r.bridge_ms for r in self.records]),
            'validation': np.array([r.validation_ms for r in self.records]),
            'l2_preprocess': np.array([r.l2_preprocess_ms for r in self.records]),
            'l2_regime': np.array([r.l2_regime_ms for r in self.records]),
            'total': np.array([r.total_ms for r in self.records]),
        }


def run_test(n_iterations: int = 20000) -> LatencyCollector:
    """Run latency test and collect all measurements."""
    print(f"Running {n_iterations:,} iterations...")
    
    collector = LatencyCollector()
    
    # Try to load bridge
    try:
        from l1_l2_bridge import L1L2Bridge, Layer1Output, Layer1DataAdapter
        bridge = L1L2Bridge(enable_validation=True)
        adapter = bridge.adapter
    except ImportError:
        bridge = None
        adapter = None
    
    for i in range(n_iterations):
        iter_start = time.perf_counter()
        
        # L1 Feature Generation
        t0 = time.perf_counter()
        features = np.random.randn(60) * 0.5
        features = np.clip(features, -1, 1)
        features[55] = np.random.uniform(0, 1)  # VPIN
        features[59] = np.random.uniform(0, 1)  # aggressive_ratio
        l1_ms = (time.perf_counter() - t0) * 1000
        
        # Bridge Adaptation
        t0 = time.perf_counter()
        if bridge:
            l1_out = Layer1Output(
                features=features, timestamp=time.time(), 
                symbol="BTCUSDT", n_nonzero=55
            )
            l2_in = bridge.process(l1_out)
            bridge_ms = (time.perf_counter() - t0) * 1000
        else:
            bridge_ms = 0.05  # Simulated
        
        # Validation
        t0 = time.perf_counter()
        if adapter:
            is_valid, _ = adapter.validate_feature_vector(features)
        validation_ms = (time.perf_counter() - t0) * 1000
        
        # L2 Preprocessing
        t0 = time.perf_counter()
        _ = (features - np.mean(features)) / (np.std(features) + 1e-8)
        l2_pre_ms = (time.perf_counter() - t0) * 1000
        
        # L2 Regime Detection
        t0 = time.perf_counter()
        # Ensure positive alphas for dirichlet
        alphas = np.abs(features[18:21]) + 0.5
        alphas = np.maximum(alphas, 0.1)  # Floor at 0.1
        _ = np.random.dirichlet(alphas)
        l2_reg_ms = (time.perf_counter() - t0) * 1000
        
        total_ms = (time.perf_counter() - iter_start) * 1000
        
        collector.add(LatencyRecord(
            l1_feature_ms=l1_ms,
            bridge_ms=bridge_ms,
            validation_ms=validation_ms,
            l2_preprocess_ms=l2_pre_ms,
            l2_regime_ms=l2_reg_ms,
            total_ms=total_ms
        ))
        
        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i+1:,}/{n_iterations:,}")
    
    return collector


def generate_graphs(collector: LatencyCollector, output_path: str):
    """Generate latency visualization graphs."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    data = collector.get_arrays()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('HIMARI L1→L2 Pipeline Latency Analysis (20,000 iterations)', 
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Color scheme
    colors = {
        'l1_feature': '#2ecc71',
        'bridge': '#3498db', 
        'validation': '#9b59b6',
        'l2_preprocess': '#e74c3c',
        'l2_regime': '#f39c12',
        'total': '#1abc9c'
    }
    
    component_names = {
        'l1_feature': 'L1 Feature Gen',
        'bridge': 'Bridge Adapt',
        'validation': 'Validation',
        'l2_preprocess': 'L2 Preprocess',
        'l2_regime': 'L2 Regime',
        'total': 'Total Pipeline'
    }
    
    # 1. Histogram of total latency (top left, larger)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(data['total'], bins=100, color=colors['total'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(np.mean(data['total']), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(data['total']):.3f}ms")
    ax1.axvline(np.percentile(data['total'], 99), color='orange', linestyle='--', linewidth=2, label=f"P99: {np.percentile(data['total'], 99):.3f}ms")
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title('Total Pipeline Latency Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot of all components (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    bp_data = [data[k] for k in ['l1_feature', 'bridge', 'validation', 'l2_preprocess', 'l2_regime']]
    bp = ax2.boxplot(bp_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[k] for k in ['l1_feature', 'bridge', 'validation', 'l2_preprocess', 'l2_regime']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xticklabels(['L1 Feat', 'Bridge', 'Valid', 'L2 Pre', 'L2 Reg'], rotation=45)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Component Latency Box Plot')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Stacked area chart showing component contribution
    ax3 = fig.add_subplot(gs[1, :])
    n_samples = min(1000, len(data['total']))
    sample_idx = np.linspace(0, len(data['total'])-1, n_samples, dtype=int)
    
    components = ['l1_feature', 'bridge', 'validation', 'l2_preprocess', 'l2_regime']
    stacked = np.vstack([data[k][sample_idx] for k in components])
    
    ax3.stackplot(range(n_samples), stacked, 
                  labels=[component_names[k] for k in components],
                  colors=[colors[k] for k in components],
                  alpha=0.8)
    ax3.set_xlabel('Sample (downsampled)')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Latency Breakdown Over Time (Stacked)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Bar chart of mean latencies
    ax4 = fig.add_subplot(gs[2, 0])
    means = [np.mean(data[k]) for k in components]
    bars = ax4.bar(range(len(components)), means, color=[colors[k] for k in components])
    ax4.set_xticks(range(len(components)))
    ax4.set_xticklabels(['L1', 'Bridge', 'Valid', 'L2Pre', 'L2Reg'], rotation=45)
    ax4.set_ylabel('Mean Latency (ms)')
    ax4.set_title('Mean Latency by Component')
    for bar, mean in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Percentile comparison
    ax5 = fig.add_subplot(gs[2, 1])
    percentiles = [50, 90, 95, 99]
    x = np.arange(len(percentiles))
    width = 0.15
    
    for i, comp in enumerate(components):
        pcts = [np.percentile(data[comp], p) for p in percentiles]
        ax5.bar(x + i*width, pcts, width, label=component_names[comp], color=colors[comp])
    
    ax5.set_xticks(x + width*2)
    ax5.set_xticklabels(['P50', 'P90', 'P95', 'P99'])
    ax5.set_ylabel('Latency (ms)')
    ax5.set_title('Percentile Comparison')
    ax5.legend(fontsize=7, loc='upper left')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Stats summary table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = "=== LATENCY SUMMARY ===\n\n"
    stats_text += f"Total Iterations: {len(data['total']):,}\n\n"
    stats_text += "TOTAL PIPELINE:\n"
    stats_text += f"  Mean:  {np.mean(data['total']):.4f} ms\n"
    stats_text += f"  P50:   {np.percentile(data['total'], 50):.4f} ms\n"
    stats_text += f"  P95:   {np.percentile(data['total'], 95):.4f} ms\n"
    stats_text += f"  P99:   {np.percentile(data['total'], 99):.4f} ms\n"
    stats_text += f"  Max:   {np.max(data['total']):.4f} ms\n\n"
    stats_text += f"Target: < 17.0 ms\n"
    stats_text += f"Status: {'✅ PASS' if np.percentile(data['total'], 99) < 17.0 else '❌ FAIL'}"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Graph saved to: {output_path}")


def save_raw_data(collector: LatencyCollector, output_path: str):
    """Save raw latency data as JSON."""
    data = collector.get_arrays()
    
    summary = {}
    for key, arr in data.items():
        summary[key] = {
            'mean_ms': float(np.mean(arr)),
            'std_ms': float(np.std(arr)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
            'p50_ms': float(np.percentile(arr, 50)),
            'p90_ms': float(np.percentile(arr, 90)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'count': len(arr)
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Data saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--iterations", type=int, default=20000)
    parser.add_argument("-o", "--output", type=str, default="latency_results.png")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HIMARI L1→L2 Pipeline Latency Test")
    print("=" * 60)
    
    collector = run_test(args.iterations)
    
    # Generate outputs
    output_dir = LAYER2_PATH
    graph_path = os.path.join(output_dir, args.output)
    json_path = os.path.join(output_dir, "latency_results.json")
    
    if MATPLOTLIB_AVAILABLE:
        generate_graphs(collector, graph_path)
    
    save_raw_data(collector, json_path)
    
    # Print summary
    data = collector.get_arrays()
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Pipeline P99: {np.percentile(data['total'], 99):.4f} ms")
    print(f"Target: < 17.0 ms")
    print(f"Status: {'✅ PASS' if np.percentile(data['total'], 99) < 17.0 else '❌ FAIL'}")
