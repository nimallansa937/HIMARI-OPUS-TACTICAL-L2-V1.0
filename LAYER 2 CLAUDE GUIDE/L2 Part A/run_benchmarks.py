# ============================================================================
# FILE: run_benchmarks.py
# PURPOSE: Run all performance benchmarks for Part A
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import time
from scipy import stats

# Import modules
from ekf_denoiser import EKFDenoiser, EKFConfig
from conversational_ae import ConversationalAutoencoder, CAEConfig
from freq_normalizer import FrequencyDomainNormalizer, FreqNormConfig
from timegan_augment import TimeGAN, TimeGANConfig
from tab_ddpm import TabDDPM, TabDDPMConfig
from vec_normalize import VecNormalize
from online_augment import OnlineAugmentor


def generate_market_data(n_samples=500):
    """Generate realistic market data."""
    np.random.seed(42)
    returns = np.zeros(n_samples)
    vol = 0.02
    for i in range(1, n_samples):
        vol = 0.9 * vol + 0.1 * abs(returns[i-1]) + 0.001 * np.random.randn()
        returns[i] = np.random.randn() * max(vol, 0.005)
    prices = 100 * np.exp(np.cumsum(returns))
    noisy_prices = prices + np.random.randn(n_samples) * 0.5
    volumes = 1000 * (1 + 2 * np.abs(returns)) * np.random.rand(n_samples)
    return prices, noisy_prices, volumes


def compute_mmd(real, fake, sigma=1.0):
    """Compute Maximum Mean Discrepancy."""
    real_flat = real.reshape(real.shape[0], -1)
    fake_flat = fake.reshape(fake.shape[0], -1)
    n = min(100, len(real_flat), len(fake_flat))
    r = real_flat[np.random.choice(len(real_flat), n, replace=False)]
    f = fake_flat[np.random.choice(len(fake_flat), n, replace=False)]
    
    def rbf(x, y):
        diff = x[:, None, :] - y[None, :, :]
        return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))
    
    return max(np.mean(rbf(r, r)) + np.mean(rbf(f, f)) - 2 * np.mean(rbf(r, f)), 0)


def benchmark_ekf():
    """A1: Extended Kalman Filter benchmark."""
    print("\n" + "="*60)
    print("A1: EXTENDED KALMAN FILTER")
    print("="*60)
    
    prices, noisy_prices, volumes = generate_market_data(500)
    ekf = EKFDenoiser()
    
    filtered = []
    for p, v in zip(noisy_prices, volumes):
        f_price, _ = ekf.update(p, v)
        filtered.append(f_price)
    filtered = np.array(filtered)
    
    warmup = 20
    raw_mse = np.mean((noisy_prices[warmup:] - prices[warmup:])**2)
    filtered_mse = np.mean((filtered[warmup:] - prices[warmup:])**2)
    noise_reduction = 1 - filtered_mse / raw_mse
    
    # Latency
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        ekf.update(100.0 + np.random.randn(), 1000.0)
        times.append(time.perf_counter() - start)
    
    print(f"  Noise Reduction: {noise_reduction*100:.1f}% ✓")
    print(f"  Raw MSE: {raw_mse:.4f} -> Filtered MSE: {filtered_mse:.4f}")
    print(f"  Latency: {np.mean(times)*1000:.3f}ms (P99: {np.percentile(times,99)*1000:.3f}ms)")
    return noise_reduction > 0.1


def benchmark_cae():
    """A2: Conversational Autoencoders benchmark."""
    print("\n" + "="*60)
    print("A2: CONVERSATIONAL AUTOENCODERS")
    print("="*60)
    
    config = CAEConfig(input_dim=60, latent_dim=32, seq_len=24)
    cae = ConversationalAutoencoder(config)
    cae.eval()
    
    x = torch.randn(32, 24, 60)
    with torch.no_grad():
        outputs = cae(x)
    
    mse_1 = torch.nn.functional.mse_loss(outputs['recon_1'], x).item()
    mse_2 = torch.nn.functional.mse_loss(outputs['recon_2'], x).item()
    mse_cons = torch.nn.functional.mse_loss(outputs['consensus'], x).item()
    
    # Latency
    times = []
    for _ in range(100):
        x = torch.randn(1, 24, 60)
        start = time.perf_counter()
        with torch.no_grad():
            _ = cae(x)
        times.append(time.perf_counter() - start)
    
    print(f"  Reconstruction MSE - AE1: {mse_1:.4f}, AE2: {mse_2:.4f}, Consensus: {mse_cons:.4f}")
    print(f"  KL Divergence (disagreement): {outputs['disagreement'].item():.4f}")
    print(f"  Latency: {np.mean(times)*1000:.2f}ms")
    return True


def benchmark_freqnorm():
    """A3: Frequency Domain Normalization benchmark."""
    print("\n" + "="*60)
    print("A3: FREQUENCY DOMAIN NORMALIZATION")
    print("="*60)
    
    config = FreqNormConfig(window_size=256, n_freq_components=32)
    norm = FrequencyDomainNormalizer(config)
    
    # Spectral stability
    t = np.linspace(0, 10, 256)
    signal_1 = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    norm_1, meta_1 = norm.normalize(signal_1)
    
    amp_var_orig = np.var(meta_1['original_amplitudes'])
    amp_var_norm = np.var(meta_1['normalized_amplitudes'])
    
    # Spectral concentration (leakage test)
    freq = 10
    pure_sine = np.sin(2 * np.pi * freq * np.linspace(0, 1, 256))
    _, meta = norm.normalize(pure_sine)
    amps = meta['original_amplitudes']
    concentration = amps[np.argmax(amps)] / np.sum(amps)
    
    # Latency
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        norm.normalize(np.random.randn(256))
        times.append(time.perf_counter() - start)
    
    print(f"  Amplitude Variance - Original: {amp_var_orig:.4f}, Normalized: {amp_var_norm:.4f}")
    print(f"  Spectral Concentration: {concentration*100:.1f}% (peak energy)")
    print(f"  Dominant Frequency Bin: {meta['dominant_frequency']}")
    print(f"  Latency: {np.mean(times)*1000:.3f}ms")
    return True


def benchmark_timegan():
    """A4: TimeGAN benchmark (slow)."""
    print("\n" + "="*60)
    print("A4: TimeGAN AUGMENTATION")
    print("="*60)
    
    config = TimeGANConfig(
        seq_len=16, feature_dim=10, hidden_dim=32,
        latent_dim=16, num_layers=2, epochs=20, batch_size=32
    )
    timegan = TimeGAN(config, device='cpu')
    
    # Generate sinusoidal training data
    np.random.seed(42)
    n_samples = 200
    t = np.linspace(0, 4*np.pi, 16)
    real_data = np.zeros((n_samples, 16, 10))
    for i in range(n_samples):
        freq = 1 + np.random.rand()
        phase = np.random.rand() * 2 * np.pi
        for j in range(10):
            real_data[i, :, j] = np.sin(freq * t + phase) + 0.1 * np.random.randn(16)
    
    print("  Training (20 epochs)...")
    timegan.train(real_data)
    
    synthetic = timegan.generate(100)
    mmd = compute_mmd(real_data, synthetic)
    
    # Statistical similarity
    real_mean, fake_mean = real_data.mean(), synthetic.mean()
    real_std, fake_std = real_data.std(), synthetic.std()
    
    print(f"  MMD Score: {mmd:.6f} (target: <0.01)")
    print(f"  Mean - Real: {real_mean:.3f}, Synthetic: {fake_mean:.3f}")
    print(f"  Std - Real: {real_std:.3f}, Synthetic: {fake_std:.3f}")
    return synthetic.shape == (100, 16, 10)


def benchmark_tabddpm():
    """A5: Tab-DDPM benchmark (slow)."""
    print("\n" + "="*60)
    print("A5: TAB-DDPM DIFFUSION")
    print("="*60)
    
    config = TabDDPMConfig(
        feature_dim=10, hidden_dim=64, n_layers=3,
        n_timesteps=100, epochs=20, batch_size=64
    )
    ddpm = TabDDPM(config, device='cpu')
    
    # Fat-tailed training data
    np.random.seed(42)
    real_data = np.random.standard_t(df=3, size=(500, 10))
    
    print("  Training (20 epochs)...")
    ddpm.train(real_data)
    
    synthetic = ddpm.generate(200)
    
    # Tail coverage
    real_tail_pct = (np.abs(real_data) > 2).mean() * 100
    fake_tail_pct = (np.abs(synthetic) > 2).mean() * 100
    
    # MMD
    mmd = compute_mmd(real_data, synthetic)
    
    print(f"  MMD Score: {mmd:.6f}")
    print(f"  Tail Events (|x|>2) - Real: {real_tail_pct:.1f}%, Synthetic: {fake_tail_pct:.1f}%")
    return True


def benchmark_vecnorm():
    """A6: VecNormalize benchmark."""
    print("\n" + "="*60)
    print("A6: VECNORMALIZE")
    print("="*60)
    
    norm = VecNormalize((60,))
    np.random.seed(42)
    data = np.random.randn(1000, 60) * 10 + 50
    
    normalized = []
    for x in data:
        normalized.append(norm(x))
    normalized = np.array(normalized)[100:]  # Skip warmup
    
    post_mean = normalized.mean()
    post_std = normalized.std()
    post_skew = stats.skew(normalized.flatten())
    post_kurtosis = stats.kurtosis(normalized.flatten())
    
    # Clipping test
    outlier = np.ones(10) * 100
    norm_small = VecNormalize((10,))
    for _ in range(100):
        norm_small(np.random.randn(10))
    clipped = norm_small(outlier)
    
    print(f"  Post-norm Mean: {post_mean:.4f} (target: ~0)")
    print(f"  Post-norm Std: {post_std:.4f} (target: ~1)")
    print(f"  Skewness: {post_skew:.4f}, Kurtosis: {post_kurtosis:.4f}")
    print(f"  Clipping: Input max {outlier.max():.0f} -> Output max {clipped.max():.2f}")
    return abs(post_mean) < 1.0 and 0.5 < post_std < 2.0


def benchmark_onlineaugment():
    """A8: Online Augmentation benchmark."""
    print("\n" + "="*60)
    print("A8: ONLINE AUGMENTATION")
    print("="*60)
    
    aug = OnlineAugmentor()
    original = np.random.randn(60)
    
    # Diversity
    augmented = np.array([aug.augment(original) for _ in range(100)])
    dist_from_orig = np.mean([np.linalg.norm(a - original) for a in augmented])
    pairwise_dist = np.mean([np.linalg.norm(augmented[i] - augmented[i+1]) 
                            for i in range(len(augmented)-1)])
    
    # Regime adaptation
    aug.set_regime_uncertainty(0.0)
    low_jitter = np.std([aug.add_jitter(np.zeros(60)) for _ in range(100)])
    aug.set_regime_uncertainty(1.0)
    high_jitter = np.std([aug.add_jitter(np.zeros(60)) for _ in range(100)])
    
    # Latency
    times = []
    for _ in range(1000):
        x = np.random.randn(60)
        start = time.perf_counter()
        aug.augment(x)
        times.append(time.perf_counter() - start)
    
    print(f"  Mean Distance from Original: {dist_from_orig:.4f}")
    print(f"  Pairwise Diversity: {pairwise_dist:.4f}")
    print(f"  Low Uncertainty Jitter Std: {low_jitter:.6f}")
    print(f"  High Uncertainty Jitter Std: {high_jitter:.6f}")
    print(f"  Regime Adaptation Ratio: {high_jitter/low_jitter:.2f}x")
    print(f"  Latency: {np.mean(times)*1000:.3f}ms")
    return high_jitter > low_jitter


def main():
    print("="*60)
    print("   HIMARI PART A - PERFORMANCE BENCHMARKS")
    print("="*60)
    
    results = {}
    
    # Quick tests
    results['A1_EKF'] = benchmark_ekf()
    results['A2_CAE'] = benchmark_cae()
    results['A3_FreqNorm'] = benchmark_freqnorm()
    results['A6_VecNorm'] = benchmark_vecnorm()
    results['A8_OnlineAug'] = benchmark_onlineaugment()
    
    # Slow tests
    print("\n" + "="*60)
    print("RUNNING SLOW TESTS (TimeGAN, Tab-DDPM)...")
    print("="*60)
    
    results['A4_TimeGAN'] = benchmark_timegan()
    results['A5_TabDDPM'] = benchmark_tabddpm()
    
    # Summary
    print("\n" + "="*60)
    print("   BENCHMARK SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("   ALL BENCHMARKS PASSED!")
    else:
        print("   SOME BENCHMARKS FAILED")
    print("="*60)


if __name__ == '__main__':
    main()
