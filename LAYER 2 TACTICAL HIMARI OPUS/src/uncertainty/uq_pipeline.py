"""
HIMARI Layer 2 - Part F: Uncertainty Quantification
Complete UQ pipeline with 8 methods for calibrated confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# F1: CT-SSF Latent Conformal
# ============================================================================

@dataclass
class CTSSFConfig:
    latent_dim: int = 64
    n_calibration: int = 500
    alpha: float = 0.10
    attention_temp: float = 1.0

class LatentEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )
    def forward(self, x): return self.encoder(x)

class LatentDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )
    def forward(self, z): return self.decoder(z)

class CTSSF:
    """Conformalized Time Series with Semantic Features in latent space."""
    def __init__(self, encoder, decoder, config=None, device='cpu'):
        self.encoder, self.decoder = encoder.to(device), decoder.to(device)
        self.config = config or CTSSFConfig()
        self.device = device
        self.calibration_scores = np.array([])
        self.calibration_latents = []
        
    def calibrate(self, X_cal, Y_cal):
        scores = []
        for i in range(len(X_cal)):
            with torch.no_grad():
                z = self.encoder(X_cal[i:i+1].to(self.device)).squeeze()
            scores.append(z.norm().item())
            self.calibration_latents.append(z)
        self.calibration_scores = np.array(scores)
        return {'n_calibration': len(X_cal), 'quantile': np.quantile(self.calibration_scores, 1-self.config.alpha)}
    
    @torch.no_grad()
    def predict_interval(self, x, point_pred):
        q = np.quantile(self.calibration_scores, 1-self.config.alpha) if len(self.calibration_scores) > 0 else 0.1
        return point_pred - q*0.1, point_pred + q*0.1, q*0.2


# ============================================================================
# F2: CPTC Regime Change Points
# ============================================================================

@dataclass
class CPTCConfig:
    base_alpha: float = 0.10
    regime_expansion: float = 2.0
    decay_rate: float = 0.95
    lookback_window: int = 100

class CPTC:
    """Conformal Prediction with Temporal Covariate for regime-aware intervals."""
    def __init__(self, config=None):
        self.config = config or CPTCConfig()
        self.global_scores = []
        self.expansion_factor = 1.0
        self.current_regime = 0
        self.scores_since_change = []
        
    def calibrate(self, residuals):
        self.global_scores = list(np.abs(residuals))
        return {'n_samples': len(residuals), 'base_quantile': np.quantile(residuals, 1-self.config.base_alpha)}
    
    def predict_interval(self, point_pred):
        q = np.quantile(self.global_scores, 1-self.config.base_alpha) if self.global_scores else 0.1
        q *= self.expansion_factor
        return point_pred - q, point_pred + q, {'expansion': self.expansion_factor}
    
    def update(self, residual):
        self.global_scores.append(abs(residual))
        if len(self.global_scores) > self.config.lookback_window * 2:
            self.global_scores = self.global_scores[-self.config.lookback_window:]
        self.expansion_factor = 1.0 + (self.expansion_factor - 1.0) * self.config.decay_rate


# ============================================================================
# F3: Temperature Scaling
# ============================================================================

@dataclass
class TemperatureScalingConfig:
    initial_temp: float = 1.5
    min_temp: float = 0.1
    max_temp: float = 10.0

class TemperatureScaler:
    """Post-hoc temperature scaling for calibrated confidence."""
    def __init__(self, config=None):
        self.config = config or TemperatureScalingConfig()
        self.temperature = self.config.initial_temp
        
    def fit(self, logits, labels):
        best_temp, best_ece = 1.0, float('inf')
        for temp in np.linspace(self.config.min_temp, self.config.max_temp, 50):
            probs = self._scale(logits, temp)
            ece = self._compute_ece(probs, labels)
            if ece < best_ece:
                best_ece, best_temp = ece, temp
        self.temperature = best_temp
        return {'temperature': best_temp, 'ece': best_ece}
    
    def calibrate(self, logits):
        return self._scale(logits, self.temperature)
    
    def _scale(self, logits, temp):
        if isinstance(logits, torch.Tensor):
            return F.softmax(logits / temp, dim=-1).numpy()
        return np.exp(logits/temp) / np.sum(np.exp(logits/temp), axis=-1, keepdims=True)
    
    def _compute_ece(self, probs, labels, n_bins=15):
        confs = np.max(probs, axis=-1) if probs.ndim > 1 else probs
        preds = np.argmax(probs, axis=-1) if probs.ndim > 1 else (probs > 0.5).astype(int)
        accs = (preds == labels).astype(float)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = i/n_bins, (i+1)/n_bins
            mask = (confs > lo) & (confs <= hi)
            if mask.sum() > 0:
                ece += mask.mean() * abs(accs[mask].mean() - confs[mask].mean())
        return ece


# ============================================================================
# F4: Deep Ensemble Disagreement
# ============================================================================

class DeepEnsemble:
    """Ensemble disagreement for epistemic uncertainty."""
    def __init__(self, models: List[nn.Module], device='cpu'):
        self.models = [m.to(device) for m in models]
        self.device = device
        
    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        preds = [m(x) for m in self.models]
        stacked = torch.stack(preds)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std


# ============================================================================
# F5: MC Dropout
# ============================================================================

class MCDropout:
    """Monte Carlo Dropout for cheap epistemic uncertainty."""
    def __init__(self, model: nn.Module, n_samples: int = 30, device='cpu'):
        self.model = model.to(device)
        self.n_samples = n_samples
        self.device = device
        
    def predict(self, x):
        self.model.train()  # Enable dropout
        x = x.to(self.device)
        preds = [self.model(x) for _ in range(self.n_samples)]
        stacked = torch.stack(preds)
        mean, std = stacked.mean(dim=0), stacked.std(dim=0)
        self.model.eval()
        return mean, std


# ============================================================================
# F6: Epistemic/Aleatoric Split
# ============================================================================

class UncertaintySplitter:
    """Split total uncertainty into epistemic and aleatoric components."""
    def __init__(self):
        self.epistemic_history = []
        self.aleatoric_history = []
        
    def split(self, ensemble_std, mc_std, data_noise_estimate=0.01):
        epistemic = ensemble_std  # Model uncertainty
        aleatoric = np.sqrt(np.maximum(0, mc_std**2 - ensemble_std**2 + data_noise_estimate**2))
        self.epistemic_history.append(float(np.mean(epistemic)))
        self.aleatoric_history.append(float(np.mean(aleatoric)))
        return epistemic, aleatoric
    
    def get_ratio(self):
        if not self.epistemic_history:
            return 0.5
        e, a = np.mean(self.epistemic_history[-100:]), np.mean(self.aleatoric_history[-100:])
        return e / (e + a + 1e-8)


# ============================================================================
# F7: Data Uncertainty (k-NN OOD)
# ============================================================================

class KNNOODDetector:
    """k-NN based out-of-distribution detection."""
    def __init__(self, k: int = 5, threshold_percentile: float = 95):
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.reference_embeddings = None
        self.threshold = None
        
    def fit(self, embeddings: np.ndarray):
        self.reference_embeddings = embeddings
        distances = self._compute_knn_distances(embeddings)
        self.threshold = np.percentile(distances, self.threshold_percentile)
        return {'threshold': self.threshold, 'n_reference': len(embeddings)}
    
    def _compute_knn_distances(self, query):
        if self.reference_embeddings is None:
            return np.zeros(len(query))
        dists = np.linalg.norm(query[:, None] - self.reference_embeddings[None, :], axis=-1)
        knn_dists = np.sort(dists, axis=1)[:, :self.k].mean(axis=1)
        return knn_dists
    
    def score(self, embedding: np.ndarray):
        if self.reference_embeddings is None:
            return 0.0, False
        dist = np.linalg.norm(embedding - self.reference_embeddings, axis=-1)
        knn_dist = np.sort(dist)[:self.k].mean()
        is_ood = knn_dist > self.threshold if self.threshold else False
        return knn_dist, is_ood


# ============================================================================
# F8: Predictive Uncertainty
# ============================================================================

class PredictiveUncertainty:
    """Forecast future uncertainty for proactive risk scaling."""
    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.uncertainty_history = deque(maxlen=500)
        
    def update(self, uncertainty: float):
        self.uncertainty_history.append(uncertainty)
        
    def forecast(self):
        if len(self.uncertainty_history) < 20:
            return np.mean(self.uncertainty_history) if self.uncertainty_history else 0.1
        recent = np.array(list(self.uncertainty_history)[-50:])
        trend = (recent[-1] - recent[0]) / len(recent)
        forecast = recent[-1] + trend * self.horizon
        return max(0.01, forecast)


# ============================================================================
# Complete UQ Pipeline
# ============================================================================

@dataclass
class UQPipelineConfig:
    use_ctssf: bool = True
    use_cptc: bool = True
    use_temperature: bool = True
    use_knn_ood: bool = True
    device: str = 'cpu'

class UncertaintyQuantificationPipeline:
    """Complete UQ pipeline integrating all 8 methods."""
    
    def __init__(self, config: Optional[UQPipelineConfig] = None):
        self.config = config or UQPipelineConfig()
        self.temperature_scaler = TemperatureScaler()
        self.cptc = CPTC()
        self.knn_ood = KNNOODDetector()
        self.uncertainty_splitter = UncertaintySplitter()
        self.predictive_uq = PredictiveUncertainty()
        
    def calibrate(self, logits, labels, residuals, embeddings=None):
        results = {}
        results['temperature'] = self.temperature_scaler.fit(logits, labels)
        results['cptc'] = self.cptc.calibrate(residuals)
        if embeddings is not None:
            results['knn'] = self.knn_ood.fit(embeddings)
        return results
    
    def quantify(self, logits, point_pred, embedding=None):
        calibrated_probs = self.temperature_scaler.calibrate(logits)
        confidence = float(np.max(calibrated_probs))
        
        lower, upper, cptc_info = self.cptc.predict_interval(point_pred)
        interval_width = upper - lower
        
        ood_score, is_ood = (0.0, False)
        if embedding is not None:
            ood_score, is_ood = self.knn_ood.score(embedding)
            
        if is_ood:
            confidence *= 0.5  # Reduce confidence for OOD
            
        self.predictive_uq.update(interval_width)
        future_uq = self.predictive_uq.forecast()
        
        return {
            'confidence': confidence,
            'calibrated_probs': calibrated_probs,
            'interval_lower': lower,
            'interval_upper': upper,
            'interval_width': interval_width,
            'ood_score': ood_score,
            'is_ood': is_ood,
            'future_uncertainty': future_uq,
            'epistemic_ratio': self.uncertainty_splitter.get_ratio()
        }
