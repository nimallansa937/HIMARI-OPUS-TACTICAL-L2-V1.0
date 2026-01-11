"""
HIMARI Layer 2 - Part B: Regime Detection Tests
Unit tests for regime detection subsystem.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRegimeDetection(unittest.TestCase):
    """Tests for Part B: Regime Detection."""
    
    def test_regime_pipeline_import(self):
        """Test regime pipeline can be imported."""
        from regime_detection import RegimeDetectionPipeline, RegimePipelineConfig
        self.assertIsNotNone(RegimeDetectionPipeline)
        self.assertIsNotNone(RegimePipelineConfig)
        
    def test_regime_pipeline_creation(self):
        """Test regime pipeline can be created."""
        from regime_detection import RegimeDetectionPipeline, RegimePipelineConfig
        config = RegimePipelineConfig()
        pipeline = RegimeDetectionPipeline(config)
        self.assertIsNotNone(pipeline)
        
    def test_regime_detection(self):
        """Test regime detection on sample data."""
        from regime_detection import RegimeDetectionPipeline, RegimePipelineConfig
        config = RegimePipelineConfig()
        pipeline = RegimeDetectionPipeline(config)
        
        features = np.random.randn(60)
        result = pipeline.detect(features)
        
        self.assertIn('regime', result)
        self.assertIn('confidence', result)
        self.assertIn(result['regime'], [0, 1, 2, 3])
        
    def test_hmm_detector(self):
        """Test HMM-based regime detection."""
        from regime_detection import HMMRegimeDetector
        detector = HMMRegimeDetector()
        features = np.random.randn(100, 10)
        regimes = detector.fit_predict(features)
        self.assertEqual(len(regimes), 100)
        
    def test_integration_example(self):
        """Test integration example runs."""
        from regime_detection.integration_example import integration_example
        result = integration_example()
        self.assertIn('regime', result)
        self.assertIn('strategy', result)


if __name__ == '__main__':
    unittest.main()
