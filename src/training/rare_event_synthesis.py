"""
HIMARI Layer 2 - Part K8: Rare Event Synthesis
Crisis scenario generation for robust training.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RareEventConfig:
    """Configuration for rare event synthesis."""
    crash_magnitude_range: Tuple[float, float] = (0.05, 0.20)
    volatility_spike_range: Tuple[float, float] = (2.0, 5.0)
    correlation_breakdown_prob: float = 0.3
    flash_crash_duration_bars: int = 10
    recovery_duration_bars: int = 50
    cascade_probability: float = 0.2
    n_synthetic_per_real: int = 3


class CrashScenarioGenerator:
    """Generate synthetic market crash scenarios."""
    
    def __init__(self, config: RareEventConfig = None):
        self.config = config or RareEventConfig()
    
    def generate_flash_crash(self, base_returns: np.ndarray) -> np.ndarray:
        """Generate flash crash pattern."""
        n = len(base_returns)
        synthetic = base_returns.copy()
        
        # Random crash start point
        crash_start = np.random.randint(20, n - self.config.flash_crash_duration_bars - 20)
        
        # Crash magnitude
        magnitude = np.random.uniform(*self.config.crash_magnitude_range)
        
        # Sharp drop
        crash_returns = -magnitude / self.config.flash_crash_duration_bars
        crash_returns *= np.linspace(1.5, 0.5, self.config.flash_crash_duration_bars)
        
        synthetic[crash_start:crash_start + self.config.flash_crash_duration_bars] = crash_returns
        
        # Partial recovery
        recovery_start = crash_start + self.config.flash_crash_duration_bars
        recovery_end = min(recovery_start + self.config.recovery_duration_bars, n)
        recovery_magnitude = magnitude * 0.7  # Don't fully recover
        recovery_returns = recovery_magnitude / (recovery_end - recovery_start)
        synthetic[recovery_start:recovery_end] = recovery_returns * np.exp(-np.linspace(0, 2, recovery_end - recovery_start))
        
        return synthetic
    
    def generate_volatility_spike(self, base_returns: np.ndarray) -> np.ndarray:
        """Generate high volatility regime."""
        synthetic = base_returns.copy()
        
        # Random spike period
        spike_start = np.random.randint(10, len(base_returns) - 50)
        spike_duration = np.random.randint(20, 50)
        spike_end = min(spike_start + spike_duration, len(base_returns))
        
        # Scale up volatility
        vol_multiplier = np.random.uniform(*self.config.volatility_spike_range)
        synthetic[spike_start:spike_end] *= vol_multiplier
        
        return synthetic


class CorrelationBreakdownGenerator:
    """Generate correlation breakdown scenarios."""
    
    def __init__(self, config: RareEventConfig = None):
        self.config = config or RareEventConfig()
    
    def generate(self, features: np.ndarray) -> np.ndarray:
        """Break correlations temporarily."""
        synthetic = features.copy()
        n, d = synthetic.shape
        
        # Random breakdown period
        breakdown_start = np.random.randint(10, n - 30)
        breakdown_duration = np.random.randint(10, 30)
        breakdown_end = min(breakdown_start + breakdown_duration, n)
        
        # Shuffle correlations within period
        for i in range(breakdown_start, breakdown_end):
            # Randomly permute some features
            n_permute = np.random.randint(1, d // 3)
            permute_idx = np.random.choice(d, n_permute, replace=False)
            synthetic[i, permute_idx] = np.random.permutation(synthetic[i, permute_idx])
        
        return synthetic


class CascadeEventGenerator:
    """Generate liquidation cascade events."""
    
    def __init__(self, config: RareEventConfig = None):
        self.config = config or RareEventConfig()
    
    def generate(self, base_returns: np.ndarray) -> np.ndarray:
        """Generate cascading liquidation pattern."""
        synthetic = base_returns.copy()
        n = len(synthetic)
        
        # Cascade start
        cascade_start = np.random.randint(20, n - 40)
        
        # Multiple waves of selling
        n_waves = np.random.randint(2, 5)
        wave_spacing = np.random.randint(3, 8)
        
        for wave in range(n_waves):
            wave_start = cascade_start + wave * wave_spacing
            wave_duration = np.random.randint(2, 5)
            wave_magnitude = -np.random.uniform(0.01, 0.03) * (1 + wave * 0.3)
            
            for i in range(wave_duration):
                idx = wave_start + i
                if idx < n:
                    synthetic[idx] = wave_magnitude * (1 - i / wave_duration)
        
        return synthetic


class RareEventSynthesizer:
    """
    Complete rare event synthesis pipeline.
    
    Generates diverse crisis scenarios for training robustness:
    - Flash crashes
    - Volatility spikes
    - Correlation breakdowns
    - Liquidation cascades
    """
    
    def __init__(self, config: RareEventConfig = None):
        self.config = config or RareEventConfig()
        
        self.crash_gen = CrashScenarioGenerator(self.config)
        self.corr_gen = CorrelationBreakdownGenerator(self.config)
        self.cascade_gen = CascadeEventGenerator(self.config)
        
        self._synthesis_count = 0
    
    def synthesize(self, 
                  features: np.ndarray,
                  returns: np.ndarray,
                  labels: Optional[np.ndarray] = None,
                  n_synthetic: int = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate synthetic rare events.
        
        Args:
            features: Original features (n_samples, n_features)
            returns: Original returns (n_samples,)
            labels: Optional labels
            n_synthetic: Number of synthetic samples per original
            
        Returns:
            Augmented features, returns, and labels
        """
        n_synthetic = n_synthetic or self.config.n_synthetic_per_real
        
        all_features = [features]
        all_returns = [returns]
        all_labels = [labels] if labels is not None else None
        
        for i in range(n_synthetic):
            # Generate different types of rare events
            event_type = np.random.choice(['crash', 'volatility', 'correlation', 'cascade'])
            
            if event_type == 'crash':
                synth_returns = self.crash_gen.generate_flash_crash(returns)
                synth_features = self._adjust_features_for_crash(features, synth_returns, returns)
            elif event_type == 'volatility':
                synth_returns = self.crash_gen.generate_volatility_spike(returns)
                synth_features = features * np.abs(synth_returns / (np.abs(returns) + 1e-8)).reshape(-1, 1)
            elif event_type == 'correlation':
                synth_features = self.corr_gen.generate(features)
                synth_returns = returns.copy()
            else:
                synth_returns = self.cascade_gen.generate(returns)
                synth_features = self._adjust_features_for_crash(features, synth_returns, returns)
            
            all_features.append(synth_features)
            all_returns.append(synth_returns)
            
            if all_labels is not None:
                # Adjust labels for changed conditions
                adjusted_labels = self._adjust_labels(labels, synth_returns)
                all_labels.append(adjusted_labels)
        
        result_features = np.vstack(all_features)
        result_returns = np.concatenate(all_returns)
        result_labels = np.concatenate(all_labels) if all_labels else None
        
        self._synthesis_count += n_synthetic
        
        return result_features, result_returns, result_labels
    
    def _adjust_features_for_crash(self, features: np.ndarray, 
                                   synth_returns: np.ndarray,
                                   orig_returns: np.ndarray) -> np.ndarray:
        """Adjust features to match synthetic returns."""
        adj_features = features.copy()
        
        # Find where synthetic differs significantly
        diff_mask = np.abs(synth_returns - orig_returns) > 0.01
        
        # Increase volatility features in crash periods
        if diff_mask.any():
            adj_features[diff_mask] *= 1.5
        
        return adj_features
    
    def _adjust_labels(self, labels: np.ndarray, synth_returns: np.ndarray) -> np.ndarray:
        """Adjust labels based on synthetic returns."""
        adj_labels = labels.copy()
        
        # Strong negative returns -> sell signal
        adj_labels[synth_returns < -0.02] = 2  # SELL
        
        return adj_labels
    
    def get_statistics(self) -> Dict:
        return {
            'total_synthetic_events': self._synthesis_count
        }


def create_rare_event_synthesizer(**kwargs) -> RareEventSynthesizer:
    """Create rare event synthesizer."""
    config = RareEventConfig(**kwargs)
    return RareEventSynthesizer(config)
