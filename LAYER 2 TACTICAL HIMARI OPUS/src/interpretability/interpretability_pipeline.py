"""
HIMARI Layer 2 - Part N: Interpretability Framework
Model interpretability and explainability.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# N1: Feature Attribution
# ============================================================================

class FeatureAttributor:
    """Compute feature importance via gradient-based methods."""
    
    def __init__(self, model: nn.Module, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def integrated_gradients(self, inputs: torch.Tensor, target: int = None,
                           n_steps: int = 50) -> np.ndarray:
        """Compute integrated gradients for feature attribution."""
        baseline = torch.zeros_like(inputs)
        scaled_inputs = [baseline + (float(i) / n_steps) * (inputs - baseline)
                        for i in range(n_steps + 1)]
        
        grads = []
        for scaled in scaled_inputs:
            scaled.requires_grad = True
            output = self.model(scaled)
            if target is not None:
                output = output[..., target]
            output.sum().backward()
            grads.append(scaled.grad.detach())
            
        avg_grads = torch.stack(grads).mean(dim=0)
        attributions = (inputs - baseline) * avg_grads
        
        return attributions.cpu().numpy()
    
    def saliency(self, inputs: torch.Tensor, target: int = None) -> np.ndarray:
        """Simple gradient saliency maps."""
        inputs.requires_grad = True
        output = self.model(inputs)
        if target is not None:
            output = output[..., target]
        output.sum().backward()
        
        return inputs.grad.abs().cpu().numpy()


# ============================================================================
# N2: Decision Explanation
# ============================================================================

class DecisionExplainer:
    """Generate human-readable explanations for decisions."""
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or [f"feature_{i}" for i in range(100)]
        
    def explain(self, attributions: np.ndarray, threshold: float = 0.1) -> str:
        """Generate text explanation from attributions."""
        abs_attr = np.abs(attributions)
        total = abs_attr.sum()
        
        if total < 1e-8:
            return "No significant feature influences detected."
            
        # Normalize
        normalized = abs_attr / total
        
        # Find important features
        important_idx = np.where(normalized > threshold)[0]
        if len(important_idx) == 0:
            important_idx = np.argsort(abs_attr)[-3:]
            
        explanations = []
        for idx in important_idx:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            direction = "positive" if attributions[idx] > 0 else "negative"
            impact = normalized[idx] * 100
            explanations.append(f"{name} has {direction} impact ({impact:.1f}%)")
            
        return "; ".join(explanations)


# ============================================================================
# N3: Attention Visualization
# ============================================================================

class AttentionVisualizer:
    """Visualize attention patterns in transformer models."""
    
    def __init__(self):
        self.attention_maps = []
        
    def capture_attention(self, attention_weights: torch.Tensor, layer_name: str):
        """Capture attention weights from a layer."""
        self.attention_maps.append({
            'layer': layer_name,
            'weights': attention_weights.detach().cpu().numpy()
        })
        
    def get_attention_summary(self) -> Dict:
        """Get summary of attention patterns."""
        if not self.attention_maps:
            return {}
            
        summaries = {}
        for attn_map in self.attention_maps:
            weights = attn_map['weights']
            summaries[attn_map['layer']] = {
                'max_attention': float(weights.max()),
                'mean_attention': float(weights.mean()),
                'entropy': float(-np.sum(weights * np.log(weights + 1e-8)))
            }
            
        return summaries
    
    def clear(self):
        self.attention_maps = []


# ============================================================================
# N4: Counterfactual Analysis
# ============================================================================

class CounterfactualAnalyzer:
    """Generate counterfactual explanations."""
    
    def __init__(self, model: nn.Module, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def find_counterfactual(self, inputs: torch.Tensor, target_class: int,
                           max_steps: int = 100, lr: float = 0.1) -> Tuple[np.ndarray, float]:
        """Find minimal change to flip prediction."""
        cf = inputs.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([cf], lr=lr)
        
        original_pred = self.model(inputs).argmax().item()
        
        for step in range(max_steps):
            optimizer.zero_grad()
            output = self.model(cf)
            
            # Loss: classification + proximity
            class_loss = -output[..., target_class].sum()
            prox_loss = torch.norm(cf - inputs)
            loss = class_loss + 0.1 * prox_loss
            
            loss.backward()
            optimizer.step()
            
            if output.argmax().item() == target_class:
                break
                
        change = (cf - inputs).detach().cpu().numpy()
        new_pred = self.model(cf).argmax().item()
        
        return change, new_pred


# ============================================================================
# N5: Rule Extraction
# ============================================================================

class RuleExtractor:
    """Extract interpretable rules from model behavior."""
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or []
        self.rules = []
        
    def extract_threshold_rules(self, inputs: np.ndarray, predictions: np.ndarray,
                                n_rules: int = 10) -> List[str]:
        """Extract simple threshold rules via decision stump analysis."""
        rules = []
        
        for i in range(min(inputs.shape[1], len(self.feature_names))):
            feature = inputs[:, i]
            feature_name = self.feature_names[i]
            
            # Find best threshold
            thresholds = np.percentile(feature, [25, 50, 75])
            
            for thresh in thresholds:
                above_mask = feature > thresh
                if above_mask.sum() > 10 and (~above_mask).sum() > 10:
                    above_pred = predictions[above_mask].mean()
                    below_pred = predictions[~above_mask].mean()
                    
                    if abs(above_pred - below_pred) > 0.3:
                        direction = "BUY" if above_pred > 0 else "SELL"
                        rules.append(f"IF {feature_name} > {thresh:.2f} THEN {direction}")
                        
        self.rules = rules[:n_rules]
        return self.rules


# ============================================================================
# N6: Confidence Calibration Analysis
# ============================================================================

class ConfidenceAnalyzer:
    """Analyze model confidence calibration."""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_accuracies = []
        self.bin_confidences = []
        
    def compute_calibration(self, confidences: np.ndarray,
                           correct: np.ndarray) -> Dict:
        """Compute calibration metrics."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        
        ece = 0.0  # Expected Calibration Error
        mce = 0.0  # Maximum Calibration Error
        
        self.bin_accuracies = []
        self.bin_confidences = []
        
        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
                bin_size = in_bin.mean()
                
                self.bin_accuracies.append(bin_acc)
                self.bin_confidences.append(bin_conf)
                
                gap = abs(bin_acc - bin_conf)
                ece += gap * bin_size
                mce = max(mce, gap)
                
        return {
            'ece': float(ece),
            'mce': float(mce),
            'bin_accuracies': self.bin_accuracies,
            'bin_confidences': self.bin_confidences
        }


# ============================================================================
# Complete Interpretability Pipeline
# ============================================================================

@dataclass
class InterpretabilityConfig:
    n_steps_ig: int = 50
    n_rules: int = 10
    n_bins: int = 10

class InterpretabilityPipeline:
    """Complete interpretability framework."""
    
    def __init__(self, model: nn.Module = None, feature_names: List[str] = None,
                 config=None, device='cpu'):
        self.config = config or InterpretabilityConfig()
        
        if model is not None:
            self.attributor = FeatureAttributor(model, device)
            self.counterfactual = CounterfactualAnalyzer(model, device)
        else:
            self.attributor = None
            self.counterfactual = None
            
        self.explainer = DecisionExplainer(feature_names)
        self.attention_viz = AttentionVisualizer()
        self.rule_extractor = RuleExtractor(feature_names)
        self.confidence_analyzer = ConfidenceAnalyzer(n_bins=self.config.n_bins)
        
    def explain_decision(self, inputs: torch.Tensor, 
                        prediction: int = None) -> Dict:
        """Generate comprehensive explanation for a decision."""
        result = {}
        
        # Feature attribution
        if self.attributor:
            attributions = self.attributor.integrated_gradients(
                inputs, n_steps=self.config.n_steps_ig
            )
            result['attributions'] = attributions
            result['explanation'] = self.explainer.explain(attributions.flatten())
            
        # Attention summary
        result['attention_summary'] = self.attention_viz.get_attention_summary()
        
        return result
    
    def analyze_model(self, inputs: np.ndarray, predictions: np.ndarray,
                     confidences: np.ndarray, correct: np.ndarray) -> Dict:
        """Run full model analysis."""
        return {
            'rules': self.rule_extractor.extract_threshold_rules(
                inputs, predictions, n_rules=self.config.n_rules
            ),
            'calibration': self.confidence_analyzer.compute_calibration(
                confidences, correct
            )
        }
