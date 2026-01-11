"""
HIMARI Layer 2 - Part E: HSM State Machine Tests
Unit tests for Hierarchical State Machine subsystem.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHSM(unittest.TestCase):
    """Tests for Part E: HSM State Machine."""
    
    def test_trading_hsm_import(self):
        """Test TradingHSM can be imported."""
        from state_machine import TradingHSM, TradingHSMConfig
        self.assertIsNotNone(TradingHSM)
        self.assertIsNotNone(TradingHSMConfig)
        
    def test_trading_hsm_creation(self):
        """Test TradingHSM can be created."""
        from state_machine import TradingHSM, TradingHSMConfig
        config = TradingHSMConfig(use_learned_transitions=False)
        hsm = TradingHSM(config)
        self.assertIsNotNone(hsm)
        
    def test_hsm_initial_state(self):
        """Test HSM starts in FLAT state."""
        from state_machine import TradingHSM, TradingHSMConfig
        config = TradingHSMConfig(use_learned_transitions=False)
        hsm = TradingHSM(config)
        state = hsm.get_state()
        self.assertEqual(state['position'], 'FLAT')
        
    def test_hsm_buy_action(self):
        """Test HSM processes BUY action."""
        from state_machine import TradingHSM, TradingHSMConfig
        config = TradingHSMConfig(use_learned_transitions=False, use_oscillation_detection=False)
        hsm = TradingHSM(config)
        result = hsm.process_action('BUY')
        self.assertTrue(result['valid'])
        self.assertEqual(result['state'], 'LONG_ENTRY')
        
    def test_orthogonal_regions(self):
        """Test orthogonal regions work independently."""
        from state_machine import OrthogonalHSM, create_trading_hsm
        hsm = create_trading_hsm()
        self.assertIn('position', hsm.regions)
        self.assertIn('regime', hsm.regions)
        
    def test_oscillation_detector(self):
        """Test oscillation detection blocks rapid changes."""
        from state_machine import OscillationDetector, OscillationConfig
        config = OscillationConfig(min_transition_interval=0.01)
        detector = OscillationDetector(config)
        
        blocked, _ = detector.should_block('FLAT', 'LONG_ENTRY')
        self.assertFalse(blocked)
        
    def test_history_states(self):
        """Test history state management."""
        from state_machine import HistoryStateManager, HistoryConfig
        config = HistoryConfig(conservative_reentry=True)
        manager = HistoryStateManager(config)
        
        manager.record_exit('LONG_MODE', 'LONG_HOLD')
        restored = manager.get_restoration_state('LONG_MODE', 'LONG_ENTRY')
        self.assertEqual(restored, 'LONG_ENTRY')  # Conservative


if __name__ == '__main__':
    unittest.main()
