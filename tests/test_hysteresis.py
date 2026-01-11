"""
HIMARI Layer 2 - Part G: Hysteresis Tests
Unit tests for hysteresis filter subsystem.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHysteresis(unittest.TestCase):
    """Tests for Part G: Hysteresis Filter."""
    
    def test_hysteresis_pipeline_import(self):
        """Test hysteresis pipeline can be imported."""
        from hysteresis import HysteresisPipeline, HysteresisConfig
        self.assertIsNotNone(HysteresisPipeline)
        
    def test_kama_adaptive(self):
        """Test KAMA adaptive thresholds."""
        from hysteresis.kama_adaptive import KAMAAdaptive
        kama = KAMAAdaptive()
        for i in range(20):
            result = kama.update(100 + np.random.randn())
        self.assertIsNotNone(result)
        
    def test_knn_pattern(self):
        """Test KNN pattern matching."""
        from hysteresis.knn_pattern import KNNPatternMatcher
        matcher = KNNPatternMatcher()
        for i in range(10):
            matcher.add_pattern(np.random.randn(10), i % 3 == 0)
        prob, _ = matcher.predict_whipsaw_probability(np.random.randn(10))
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
        
    def test_atr_bands(self):
        """Test ATR bands calculation."""
        from hysteresis.atr_bands import ATRBands
        bands = ATRBands()
        for i in range(20):
            bands.update(100 + i, 98 + i, 99 + i)
        upper, middle, lower = bands.get_bands()
        self.assertGreater(upper, middle)
        self.assertLess(lower, middle)
        
    def test_meta_learned_k(self):
        """Test meta-learned K adaptation."""
        from hysteresis.meta_learned_k import MetaLearnedK
        mlk = MetaLearnedK()
        k = mlk.get_k(regime=2)
        self.assertGreater(k, 0)
        
    def test_loss_aversion(self):
        """Test loss aversion weighting."""
        from hysteresis.loss_aversion import LossAversionWeight
        law = LossAversionWeight()
        weight = law.calculate(-0.01)
        self.assertGreater(weight, 1.0)  # Loss should have higher weight


if __name__ == '__main__':
    unittest.main()
