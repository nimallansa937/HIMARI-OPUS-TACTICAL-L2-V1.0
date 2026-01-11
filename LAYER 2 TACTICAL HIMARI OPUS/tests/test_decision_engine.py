"""
HIMARI Layer 2 - Part D: Decision Engine Tests
Unit tests for decision engine subsystem.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDecisionEngine(unittest.TestCase):
    """Tests for Part D: Decision Engine."""
    
    def test_decision_pipeline_import(self):
        """Test decision pipeline can be imported."""
        from decision_engine import DecisionEngine, DecisionConfig
        self.assertIsNotNone(DecisionEngine)
        self.assertIsNotNone(DecisionConfig)
        
    def test_decision_pipeline_creation(self):
        """Test decision engine can be created."""
        from decision_engine import DecisionEngine, DecisionConfig
        config = DecisionConfig()
        engine = DecisionEngine(config)
        self.assertIsNotNone(engine)
        
    def test_decision_making(self):
        """Test basic decision making."""
        from decision_engine import DecisionEngine, DecisionConfig
        config = DecisionConfig()
        engine = DecisionEngine(config)
        
        features = np.random.randn(60)
        result = engine.decide(features)
        
        self.assertIn('action', result)
        self.assertIn('confidence', result)
        self.assertIn(result['action'], ['BUY', 'SELL', 'HOLD'])
        
    def test_ensemble_aggregation(self):
        """Test ensemble voting."""
        from decision_engine import EnsembleAggregator
        aggregator = EnsembleAggregator()
        
        predictions = [
            {'action': 'BUY', 'confidence': 0.8},
            {'action': 'BUY', 'confidence': 0.7},
            {'action': 'HOLD', 'confidence': 0.6}
        ]
        result = aggregator.aggregate(predictions)
        self.assertEqual(result['action'], 'BUY')
        
    def test_confidence_calibration(self):
        """Test confidence calibration."""
        from decision_engine import ConfidenceCalibrator
        calibrator = ConfidenceCalibrator()
        raw_conf = 0.9
        calibrated = calibrator.calibrate(raw_conf)
        self.assertGreaterEqual(calibrated, 0.0)
        self.assertLessEqual(calibrated, 1.0)


if __name__ == '__main__':
    unittest.main()
