#!/usr/bin/env python3
"""
Train Student-T AHHMM Regime Detector - SUPERVISED VERSION

Key difference from unsupervised:
- Uses ground truth labels to directly compute emission parameters per regime
- Much better accuracy since we're not guessing cluster assignments

Key Lessons from PPO Training Applied:
- Variance normalization for features
- Adaptive parameters (not hardcoded)
- Test on unseen data to verify generalization
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.special import logsumexp

# =============================================================================
# Configuration
# =============================================================================

# Data paths - LOCAL FILES
DATA_DIR = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2020_2024.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2025_2026.csv")

# Output paths
OUTPUT_DIR = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1\L2V1 AHHMM FINAL"
MODEL_PATH = os.path.join(OUTPUT_DIR, "student_t_ahhmm_supervised.pkl")

# Processed features paths
TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, "btc_1h_2020_2024_features.pkl")
TEST_FEATURES_PATH = os.path.join(DATA_DIR, "btc_1h_2025_2026_features.pkl")

# Model config
N_MARKET_STATES = 4  # LOW_VOL, TRENDING, HIGH_VOL, CRISIS
DF = 5.0             # Student-t degrees of freedom (fat tails)

# =============================================================================
# Data Loading
# =============================================================================

def load_btc_data(csv_path: str) -> pd.DataFrame:
    """Load BTC data from local CSV file."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df

# =============================================================================
# Feature Preparation - ENHANCED with more discriminative features
# =============================================================================

def prepare_ahhmm_features(df: pd.DataFrame, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare ENHANCED features for AHHMM:
    [returns, volume_norm, volatility, trend_strength, vol_of_vol, volume_spike]

    More features = better regime discrimination
    """
    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values
    low = df['low'].values

    # Feature 1: Returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])

    # Feature 2: Volatility (20-period rolling std of returns)
    volatility = pd.Series(returns).rolling(20).std().values
    volatility = np.nan_to_num(volatility, nan=0.01)

    # Feature 3: Volume normalized (ratio to 20-period MA)
    vol_ma = pd.Series(volume).rolling(20).mean().values
    volume_norm = volume / (vol_ma + 1e-10)
    volume_norm = np.nan_to_num(volume_norm, nan=1.0)

    # Feature 4: Trend strength (absolute cumulative returns over 20 bars)
    trend_strength = pd.Series(returns).rolling(20).sum().abs().values
    trend_strength = np.nan_to_num(trend_strength, nan=0.0)

    # Feature 5: Volatility of volatility (regime instability indicator)
    vol_of_vol = pd.Series(volatility).rolling(20).std().values
    vol_of_vol = np.nan_to_num(vol_of_vol, nan=0.001)

    # Feature 6: Volume spike (z-score of volume)
    vol_mean = pd.Series(volume).rolling(50).mean().values
    vol_std = pd.Series(volume).rolling(50).std().values
    volume_spike = (volume - vol_mean) / (vol_std + 1e-10)
    volume_spike = np.nan_to_num(volume_spike, nan=0.0)
    volume_spike = np.clip(volume_spike, -3, 3)  # Clip outliers

    # Feature 7: True Range normalized (intrabar volatility)
    true_range = (high - low) / close
    tr_ma = pd.Series(true_range).rolling(20).mean().values
    true_range_norm = true_range / (tr_ma + 1e-10)
    true_range_norm = np.nan_to_num(true_range_norm, nan=1.0)

    # Stack features
    features = np.column_stack([
        returns,           # 0: direction
        volatility,        # 1: risk level
        volume_norm,       # 2: activity level
        trend_strength,    # 3: trending vs ranging
        vol_of_vol,        # 4: regime stability
        volume_spike,      # 5: sudden interest
        true_range_norm    # 6: intrabar volatility
    ])

    # Remove warmup period
    features = features[200:]
    returns_out = returns[200:]

    print(f"Prepared features: {features.shape} (7 features)")
    print(f"  Returns range: [{returns_out.min():.4f}, {returns_out.max():.4f}]")
    print(f"  Volatility range: [{volatility[200:].min():.4f}, {volatility[200:].max():.4f}]")
    print(f"  Trend strength range: [{trend_strength[200:].min():.4f}, {trend_strength[200:].max():.4f}]")

    # Save processed features if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'features': features,
                'returns': returns_out,
                'timestamps': df.index[200:].tolist(),
                'feature_names': ['returns', 'volatility', 'volume_norm', 'trend_strength',
                                  'vol_of_vol', 'volume_spike', 'true_range_norm'],
                'created': datetime.now().isoformat()
            }, f)
        print(f"  Saved processed features to: {save_path}")

    return features.astype(np.float32), returns_out.astype(np.float32)

# =============================================================================
# Ground Truth Regime Labels
# =============================================================================

def create_ground_truth_regimes(returns: np.ndarray) -> np.ndarray:
    """
    Create ground truth regime labels based on volatility percentiles.
    Same logic as PPO training for consistency.

    Regimes:
        0 = LOW_VOL (quiet markets)
        1 = TRENDING (strong directional moves)
        2 = HIGH_VOL (volatile but not crisis)
        3 = CRISIS (extreme volatility)
    """
    n = len(returns)
    regimes = np.zeros(n, dtype=np.int64)

    # Rolling volatility
    vol_20 = pd.Series(returns).rolling(20).std().values

    # Trend strength (absolute cumulative returns)
    cum_ret_20 = pd.Series(returns).rolling(20).sum().abs().values

    # Percentile thresholds
    vol_low = np.nanpercentile(vol_20, 33)
    vol_high = np.nanpercentile(vol_20, 67)
    vol_crisis = np.nanpercentile(vol_20, 90)
    trend_thresh = np.nanpercentile(cum_ret_20, 60)

    for i in range(n):
        vol = vol_20[i] if not np.isnan(vol_20[i]) else 0
        trend = cum_ret_20[i] if not np.isnan(cum_ret_20[i]) else 0

        if vol > vol_crisis:
            regimes[i] = 3  # CRISIS
        elif vol > vol_high:
            regimes[i] = 2  # HIGH_VOL
        elif trend > trend_thresh and vol < vol_high:
            regimes[i] = 1  # TRENDING
        else:
            regimes[i] = 0  # LOW_VOL

    return regimes

# =============================================================================
# Student-T AHHMM - SUPERVISED VERSION
# =============================================================================

class SupervisedStudentTAHHMM:
    """
    Student-t AHHMM trained with SUPERVISED learning.

    Key difference: We use ground truth labels to compute emission parameters
    directly, instead of unsupervised EM clustering.
    """

    def __init__(self, n_states: int = 4, df: float = 5.0, n_features: int = 7):
        self.n_states = n_states
        self.df = df
        self.n_features = n_features

        # Emission parameters per state: mean, scale
        self.means = np.zeros((n_states, n_features))
        self.scales = np.ones((n_states, n_features))

        # Transition matrix (learned from data)
        self.trans = np.eye(n_states) * 0.9 + 0.1 / n_states

        # Initial state probabilities
        self.pi = np.ones(n_states) / n_states

        # State tracking
        self.state_probs = np.ones(n_states) / n_states
        self._fitted = False

        # State names for reporting
        self.state_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']

        # Feature names
        self.feature_names = ['returns', 'volatility', 'volume_norm', 'trend_strength',
                              'vol_of_vol', 'volume_spike', 'true_range_norm']

    def _student_t_logpdf(self, x: np.ndarray, mean: np.ndarray,
                          scale: np.ndarray, df: float) -> float:
        """Multivariate Student-t log probability."""
        return np.sum(stats.t.logpdf(x, df=df, loc=mean, scale=scale))

    def _emission_log_prob(self, obs: np.ndarray, state: int) -> float:
        """Log probability of observation given state."""
        # Use fatter tails (df=3) for CRISIS state
        df = 3.0 if state == 3 else self.df
        return self._student_t_logpdf(obs, self.means[state], self.scales[state], df)

    def fit(self, features: np.ndarray, labels: np.ndarray, verbose: bool = True):
        """
        SUPERVISED fit: Compute emission parameters directly from labeled data.

        Args:
            features: (n_samples, n_features) array
            labels: (n_samples,) ground truth regime labels
        """
        n_samples, n_features = features.shape
        self.n_features = n_features

        if verbose:
            print(f"\nSupervised fitting on {n_samples} samples with {n_features} features")

        # Compute emission parameters for each state using labeled data
        for s in range(self.n_states):
            mask = labels == s
            count = mask.sum()

            if count > 0:
                state_features = features[mask]
                self.means[s] = np.mean(state_features, axis=0)
                self.scales[s] = np.std(state_features, axis=0) + 1e-6

                if verbose:
                    print(f"\n  {self.state_names[s]} ({count} samples, {count/n_samples*100:.1f}%):")
                    for f_idx, f_name in enumerate(self.feature_names[:n_features]):
                        print(f"    {f_name}: mean={self.means[s, f_idx]:.5f}, scale={self.scales[s, f_idx]:.5f}")
            else:
                if verbose:
                    print(f"\n  {self.state_names[s]}: NO SAMPLES - using defaults")

        # Learn transition matrix from label sequences
        trans_counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, n_samples):
            prev_state = labels[t - 1]
            curr_state = labels[t]
            trans_counts[prev_state, curr_state] += 1

        # Normalize with Laplace smoothing
        for s in range(self.n_states):
            row_sum = trans_counts[s].sum()
            if row_sum > 0:
                self.trans[s] = (trans_counts[s] + 1) / (row_sum + self.n_states)
            else:
                self.trans[s] = np.ones(self.n_states) / self.n_states

        # Learn initial state distribution
        state_counts = np.bincount(labels, minlength=self.n_states)
        self.pi = (state_counts + 1) / (n_samples + self.n_states)

        self._fitted = True

        if verbose:
            print(f"\nTransition matrix learned:")
            print(f"{'':>12}", end='')
            for name in self.state_names:
                print(f"{name:>10}", end='')
            print()
            for i, name in enumerate(self.state_names):
                print(f"{name:>12}", end='')
                for j in range(self.n_states):
                    print(f"{self.trans[i, j]:>10.3f}", end='')
                print()

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime for each observation using Viterbi-like forward pass.

        Returns:
            regimes: (n_samples,) predicted regime indices
            confidences: (n_samples,) confidence scores
        """
        n_samples = features.shape[0]

        # Compute emission log probabilities
        log_probs = np.zeros((n_samples, self.n_states))
        for t in range(n_samples):
            for s in range(self.n_states):
                log_probs[t, s] = self._emission_log_prob(features[t], s)

        # Forward pass with transitions
        state_probs = np.zeros((n_samples, self.n_states))
        state_probs[0] = self.pi

        for t in range(1, n_samples):
            # Transition
            pred = state_probs[t-1] @ self.trans
            # Emission update
            emission = np.exp(log_probs[t] - logsumexp(log_probs[t]))
            state_probs[t] = pred * emission
            state_probs[t] /= state_probs[t].sum() + 1e-10

        regimes = np.argmax(state_probs, axis=1)
        confidences = np.max(state_probs, axis=1)

        return regimes, confidences

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_states': self.n_states,
                'df': self.df,
                'n_features': self.n_features,
                'means': self.means,
                'scales': self.scales,
                'trans': self.trans,
                'pi': self.pi,
                'state_names': self.state_names,
                'feature_names': self.feature_names,
                'fitted': self._fitted
            }, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SupervisedStudentTAHHMM':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(n_states=data['n_states'], df=data['df'], n_features=data['n_features'])
        model.means = data['means']
        model.scales = data['scales']
        model.trans = data['trans']
        model.pi = data['pi']
        model.state_names = data['state_names']
        model.feature_names = data.get('feature_names', model.feature_names)
        model._fitted = data['fitted']

        return model

# =============================================================================
# Evaluation
# =============================================================================

def evaluate_regime_detection(predicted: np.ndarray, ground_truth: np.ndarray,
                              state_names: List[str]) -> Dict:
    """Evaluate regime detection accuracy."""

    n_samples = len(predicted)

    # Overall accuracy
    accuracy = (predicted == ground_truth).mean() * 100

    # Per-regime accuracy and confusion
    results = {
        'overall_accuracy': accuracy,
        'per_regime': {},
        'confusion_matrix': np.zeros((4, 4), dtype=int)
    }

    for r in range(4):
        mask = ground_truth == r
        if mask.sum() > 0:
            regime_acc = (predicted[mask] == r).mean() * 100
            results['per_regime'][state_names[r]] = {
                'accuracy': regime_acc,
                'count': int(mask.sum()),
                'predicted_dist': {
                    state_names[i]: (predicted[mask] == i).sum()
                    for i in range(4)
                }
            }

    # Build confusion matrix
    for i in range(n_samples):
        results['confusion_matrix'][ground_truth[i], predicted[i]] += 1

    # Compute F1 scores per regime
    for r in range(4):
        tp = results['confusion_matrix'][r, r]
        fp = results['confusion_matrix'][:, r].sum() - tp
        fn = results['confusion_matrix'][r, :].sum() - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if state_names[r] in results['per_regime']:
            results['per_regime'][state_names[r]]['f1'] = f1 * 100
            results['per_regime'][state_names[r]]['precision'] = precision * 100
            results['per_regime'][state_names[r]]['recall'] = recall * 100

    return results

def print_results(results: Dict, title: str):
    """Pretty print evaluation results."""
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)

    print(f"\nOverall Accuracy: {results['overall_accuracy']:.1f}%")

    print("\nPer-Regime Metrics:")
    print("-" * 60)
    print(f"{'Regime':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Count':>8}")
    print("-" * 60)

    for regime, metrics in results['per_regime'].items():
        print(f"{regime:<12} {metrics['accuracy']:>9.1f}% {metrics.get('precision', 0):>9.1f}% "
              f"{metrics.get('recall', 0):>9.1f}% {metrics.get('f1', 0):>9.1f}% {metrics['count']:>8}")

    print("\nConfusion Matrix (rows=ground truth, cols=predicted):")
    state_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']
    print(f"{'':>12}", end='')
    for name in state_names:
        print(f"{name:>10}", end='')
    print()

    for i, name in enumerate(state_names):
        print(f"{name:>12}", end='')
        for j in range(4):
            print(f"{results['confusion_matrix'][i, j]:>10}", end='')
        print()

# =============================================================================
# Main Training
# =============================================================================

def main():
    print("=" * 70)
    print("Student-T AHHMM Regime Detector - SUPERVISED TRAINING")
    print("=" * 70)
    print(f"\nKey improvements over unsupervised:")
    print("  - Direct learning from ground truth labels")
    print("  - Enhanced features (7 instead of 3)")
    print("  - Better regime discrimination")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Step 1: Load Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Training Data (2020-2024)")
    print("=" * 70)

    train_df = load_btc_data(TRAIN_DATA_PATH)
    train_features, train_returns = prepare_ahhmm_features(train_df, save_path=TRAIN_FEATURES_PATH)
    train_labels = create_ground_truth_regimes(train_returns)

    print(f"\nTraining data: {len(train_features)} samples")
    print(f"Ground truth regime distribution:")
    for r in range(4):
        pct = (train_labels == r).mean() * 100
        print(f"  {['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS'][r]}: {pct:.1f}%")

    # =========================================================================
    # Step 2: Train Model (SUPERVISED)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Training Student-T AHHMM (SUPERVISED)")
    print("=" * 70)

    model = SupervisedStudentTAHHMM(n_states=N_MARKET_STATES, df=DF, n_features=train_features.shape[1])
    model.fit(train_features, train_labels, verbose=True)

    # =========================================================================
    # Step 3: Evaluate on Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Evaluating on Training Data")
    print("=" * 70)

    train_predicted, train_confidence = model.predict(train_features)
    train_results = evaluate_regime_detection(
        train_predicted, train_labels, model.state_names
    )
    print_results(train_results, "Training Set Results (2020-2024)")

    # =========================================================================
    # Step 4: Load and Evaluate on Test Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Evaluating on Unseen Data (2025-2026)")
    print("=" * 70)

    test_df = load_btc_data(TEST_DATA_PATH)
    test_features, test_returns = prepare_ahhmm_features(test_df, save_path=TEST_FEATURES_PATH)
    test_labels = create_ground_truth_regimes(test_returns)

    print(f"\nTest data: {len(test_features)} samples")
    print(f"Ground truth regime distribution:")
    for r in range(4):
        pct = (test_labels == r).mean() * 100
        print(f"  {['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS'][r]}: {pct:.1f}%")

    test_predicted, test_confidence = model.predict(test_features)
    test_results = evaluate_regime_detection(
        test_predicted, test_labels, model.state_names
    )
    print_results(test_results, "Test Set Results (2025-2026 Unseen Data)")

    # =========================================================================
    # Step 5: Generalization Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Generalization Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Training':>12} {'Test':>12} {'Diff':>10}")
    print("-" * 54)
    print(f"{'Overall Accuracy':<20} {train_results['overall_accuracy']:>11.1f}% "
          f"{test_results['overall_accuracy']:>11.1f}% "
          f"{test_results['overall_accuracy'] - train_results['overall_accuracy']:>+9.1f}%")

    for regime in ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']:
        train_acc = train_results['per_regime'].get(regime, {}).get('accuracy', 0)
        test_acc = test_results['per_regime'].get(regime, {}).get('accuracy', 0)
        diff = test_acc - train_acc
        print(f"{regime:<20} {train_acc:>11.1f}% {test_acc:>11.1f}% {diff:>+9.1f}%")

    # =========================================================================
    # Step 6: Save Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Saving Model")
    print("=" * 70)

    model.save(MODEL_PATH)

    # Save training log
    log_path = os.path.join(OUTPUT_DIR, "training_log_supervised.txt")
    with open(log_path, 'w') as f:
        f.write("Student-T AHHMM SUPERVISED Training Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {datetime.now().isoformat()}\n")
        f.write(f"Training Data: {TRAIN_DATA_PATH}\n")
        f.write(f"Test Data: {TEST_DATA_PATH}\n")
        f.write(f"N States: {N_MARKET_STATES}\n")
        f.write(f"Student-t df: {DF}\n")
        f.write(f"N Features: {train_features.shape[1]}\n\n")
        f.write(f"Training Accuracy: {train_results['overall_accuracy']:.1f}%\n")
        f.write(f"Test Accuracy: {test_results['overall_accuracy']:.1f}%\n\n")
        f.write("Per-Regime Test Accuracy:\n")
        for regime in ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']:
            acc = test_results['per_regime'].get(regime, {}).get('accuracy', 0)
            f.write(f"  {regime}: {acc:.1f}%\n")

    print(f"Training log saved to {log_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Training accuracy: {train_results['overall_accuracy']:.1f}%")
    print(f"Test accuracy: {test_results['overall_accuracy']:.1f}%")

    # Check generalization
    acc_diff = test_results['overall_accuracy'] - train_results['overall_accuracy']
    if abs(acc_diff) < 10:
        print("\n✅ Model generalizes well (accuracy drift < 10%)")
    else:
        print(f"\n⚠️ Generalization concern: accuracy drift = {acc_diff:.1f}%")

if __name__ == "__main__":
    main()
