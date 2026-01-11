"""HIMARI Layer 2 - Part N: Interpretability Tests"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestInterpretability(unittest.TestCase):
    """Tests for Part N: Interpretability Framework."""
    
    def test_shap_attribution(self):
        from interpretability.shap_attribution import SHAPAttribution
        shap = SHAPAttribution()
        result = shap.explain(np.random.randn(60))
        self.assertIn('shap_values', result)
        
    def test_integrated_gradients(self):
        from interpretability.shap_attribution import IntegratedGradients
        ig = IntegratedGradients()
        attribution = ig.attribute(np.random.randn(60))
        self.assertEqual(len(attribution), 60)
        
    def test_attention_visualization(self):
        from interpretability.shap_attribution import AttentionVisualization
        av = AttentionVisualization()
        weights = av.get_attention_weights({'attention': np.random.rand(60)})
        self.assertEqual(len(weights), 60)
        
    def test_concept_activation(self):
        from interpretability.concept_activation import ConceptActivation
        cav = ConceptActivation()
        cav.add_concept("bullish", [np.random.randn(60) for _ in range(10)])
        score = cav.score_concept(np.random.randn(60), "bullish")
        self.assertIsInstance(score, float)
        
    def test_counterfactual(self):
        from interpretability.concept_activation import CounterfactualExplanation
        cf = CounterfactualExplanation()
        result = cf.generate(np.random.randn(60), "SELL")
        self.assertIn('counterfactual', result)


if __name__ == '__main__':
    unittest.main()
