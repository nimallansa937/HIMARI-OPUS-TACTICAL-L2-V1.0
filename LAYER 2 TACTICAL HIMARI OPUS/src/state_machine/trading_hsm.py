# ============================================================================
# FILE: src/hsm/trading_hsm.py
# PURPOSE: Complete HSM integrating all 6 methods (E1-E6)
# LATENCY: <1ms total for all checks
# ============================================================================
"""
HIMARI Layer 2 - Complete Trading HSM
Integrates all 6 HSM methods into a unified interface.

Components:
    - E1: Orthogonal Regions (position + regime)
    - E2: Hierarchical Nesting (super-state transitions)
    - E3: History States (resume after interruption)
    - E4: Synchronized Events (cross-region coordination)
    - E5: Learned Transitions (ML-based timing)
    - E6: Oscillation Detection (anti-churn)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import logging
import time

from .orthogonal_regions import OrthogonalHSM, create_trading_hsm, PositionState, RegimeState
from .hierarchical_nesting import HierarchicalHSM, create_hierarchical_trading_fsm
from .history_states import HistoryStateManager, HistoryConfig
from .synchronized_events import SynchronizedEventBus, create_trading_event_bus, TradingEvents
from .learned_transitions import LearnedTransitionManager, LearnedTransitionConfig
from .oscillation_detection import OscillationDetector, OscillationConfig

logger = logging.getLogger(__name__)


@dataclass
class TradingHSMConfig:
    """Configuration for complete trading HSM."""
    use_learned_transitions: bool = True
    use_oscillation_detection: bool = True
    conservative_reentry: bool = True
    device: str = 'cpu'
    feature_dim: int = 64


class TradingHSM:
    """
    Complete Hierarchical State Machine for trading.
    
    Integrates:
    - E1: Orthogonal regions (position + regime)
    - E2: Hierarchical nesting (super-state transitions)
    - E3: History states (resume after interruption)
    - E4: Synchronized events (cross-region coordination)
    - E5: Learned transitions (ML-based timing)
    - E6: Oscillation detection (anti-churn)
    
    Usage:
        hsm = TradingHSM()
        result = hsm.process_action(
            proposed_action='BUY',
            features=market_features,
            regime=2,
            confidence=0.7
        )
        if result['valid']:
            execute_trade(result['action'])
    """
    
    def __init__(self, config: Optional[TradingHSMConfig] = None):
        self.config = config or TradingHSMConfig()
        
        # E1: Orthogonal regions
        self.orthogonal_hsm = create_trading_hsm()
        
        # E2: Hierarchical nesting (alternative view)
        self.hierarchical_hsm = create_hierarchical_trading_fsm()
        
        # E3: History states
        self.history_manager = HistoryStateManager(
            HistoryConfig(conservative_reentry=self.config.conservative_reentry)
        )
        
        # E4: Synchronized events
        self.event_bus = create_trading_event_bus()
        self._setup_event_handlers()
        
        # E5: Learned transitions
        self.learned_transitions = None
        if self.config.use_learned_transitions:
            self.learned_transitions = LearnedTransitionManager(
                LearnedTransitionConfig(feature_dim=self.config.feature_dim),
                device=self.config.device
            )
            
        # E6: Oscillation detection
        self.oscillation_detector = None
        if self.config.use_oscillation_detection:
            self.oscillation_detector = OscillationDetector()
            
        self._action_count = 0
        self._blocked_count = 0
        
    def _setup_event_handlers(self) -> None:
        """Register event handlers for cross-region coordination."""
        # Crisis handling
        self.event_bus.subscribe(
            TradingEvents.CRISIS_DETECTED,
            self._handle_crisis,
            priority=0  # Highest priority
        )
        
        self.event_bus.subscribe(
            TradingEvents.CRISIS_RESOLVED,
            self._handle_crisis_resolved,
            priority=0
        )
        
    def _handle_crisis(self, event: str, data: Dict) -> None:
        """Handle crisis event - force exit all positions."""
        logger.warning(f"Crisis detected: {data}")
        # Record history before forced exit
        current_pos = self.orthogonal_hsm.get_state('position')
        self.history_manager.record_exit('LONG_MODE', current_pos.name)
        self.history_manager.record_exit('SHORT_MODE', current_pos.name)
        # Force to FLAT
        self.orthogonal_hsm.process_event('crisis_exit')
        
    def _handle_crisis_resolved(self, event: str, data: Dict) -> None:
        """Handle crisis resolution - potentially resume trading."""
        logger.info(f"Crisis resolved: {data}")
        # History manager will handle re-entry logic
        
    def _map_action_to_event(self, action: str, current_state: str) -> Optional[str]:
        """Map trading action to HSM event."""
        action = action.upper()
        
        if action == 'BUY':
            if current_state == 'FLAT':
                return 'buy_signal'
            elif current_state in ['SHORT_HOLD', 'SHORT_EXIT']:
                return 'exit_signal'  # Close short first
        elif action == 'SELL':
            if current_state == 'FLAT':
                return 'sell_signal'
            elif current_state in ['LONG_HOLD', 'LONG_EXIT']:
                return 'exit_signal'  # Close long
        elif action == 'HOLD':
            return None  # No transition needed
            
        return None
    
    def process_action(
        self,
        proposed_action: str,
        features: Optional[np.ndarray] = None,
        regime: int = 2,
        confidence: float = 0.5,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a proposed trading action through the HSM.
        
        Args:
            proposed_action: 'BUY', 'SELL', or 'HOLD'
            features: Market features for learned transitions
            regime: Current market regime
            confidence: Decision engine confidence
            timestamp: Current timestamp
            
        Returns:
            Dict with:
                - valid: Whether action was validated
                - action: Final action to execute
                - state: Current HSM state
                - blocked_reason: Why blocked (if applicable)
                - transition_info: Details about any transition
        """
        timestamp = timestamp or time.time()
        self._action_count += 1
        
        current_state = self.orthogonal_hsm.get_state('position').name
        
        result = {
            'valid': False,
            'action': 'HOLD',
            'state': current_state,
            'blocked_reason': None,
            'transition_info': {}
        }
        
        # HOLD requires no state change
        if proposed_action.upper() == 'HOLD':
            result['valid'] = True
            return result
            
        # Map action to HSM event
        event = self._map_action_to_event(proposed_action, current_state)
        if event is None:
            result['blocked_reason'] = f"No valid transition for {proposed_action} from {current_state}"
            return result
            
        target_state = self._get_target_state(current_state, event)
        
        # E6: Check oscillation
        if self.oscillation_detector:
            should_block, block_reason = self.oscillation_detector.should_block(
                current_state, target_state
            )
            if should_block:
                self._blocked_count += 1
                result['blocked_reason'] = block_reason
                self.oscillation_detector.record_transition(
                    current_state, target_state, 
                    was_blocked=True, block_reason=block_reason
                )
                return result
                
        # E5: Check learned transitions (if enabled and features provided)
        if self.learned_transitions and features is not None:
            should_transition, ml_target, ml_conf, ml_info = self.learned_transitions.recommend(
                current_state, features, regime
            )
            result['transition_info']['ml_recommendation'] = {
                'should_transition': should_transition,
                'target': ml_target,
                'confidence': ml_conf
            }
            
            # If ML says don't transition, respect it (unless confidence is very high)
            if not should_transition and confidence < 0.8:
                result['blocked_reason'] = f"ML model recommends staying (conf={ml_conf:.3f})"
                return result
                
        # E1: Check orthogonal HSM validity
        if not self.orthogonal_hsm.validate_action(event, 'position'):
            result['blocked_reason'] = f"Invalid transition: {event} from {current_state}"
            return result
            
        # Execute transition
        transition_results = self.orthogonal_hsm.process_event(event, timestamp)
        
        if transition_results.get('position', False):
            new_state = self.orthogonal_hsm.get_state('position').name
            
            # Record in oscillation detector
            if self.oscillation_detector:
                self.oscillation_detector.record_transition(current_state, new_state)
                
            result['valid'] = True
            result['action'] = proposed_action.upper()
            result['state'] = new_state
            result['transition_info']['from'] = current_state
            result['transition_info']['to'] = new_state
            result['transition_info']['event'] = event
            
        return result
    
    def _get_target_state(self, current: str, event: str) -> str:
        """Get target state for a transition."""
        # Look up in orthogonal HSM transitions
        region = self.orthogonal_hsm.regions['position']
        current_enum = PositionState[current]
        key = (current_enum, event)
        if key in region.transitions:
            return region.transitions[key].name
        return current
    
    def process_regime_change(self, new_regime: int, timestamp: Optional[float] = None) -> Dict:
        """Process a regime change event."""
        timestamp = timestamp or time.time()
        
        regime_events = {
            0: 'crisis_detected',
            1: 'downtrend_detected',
            2: 'trend_end',
            3: 'trend_detected'
        }
        
        event = regime_events.get(new_regime)
        if event:
            # E4: Emit synchronized event
            results = self.event_bus.emit(event, {'regime': new_regime, 'timestamp': timestamp})
            
            # Update regime region
            self.orthogonal_hsm.process_event(event, timestamp, target_regions=['regime'])
            
            return {'event': event, 'results': results}
            
        return {'event': None}
    
    def get_state(self) -> Dict[str, str]:
        """Get current state of all regions."""
        return {
            'position': self.orthogonal_hsm.get_state('position').name,
            'regime': self.orthogonal_hsm.get_state('regime').name
        }
    
    def get_statistics(self) -> Dict:
        """Get HSM statistics."""
        stats = {
            'action_count': self._action_count,
            'blocked_count': self._blocked_count,
            'block_rate': self._blocked_count / max(1, self._action_count),
            'current_state': self.get_state()
        }
        
        if self.oscillation_detector:
            stats['oscillation'] = self.oscillation_detector.get_statistics()
            
        return stats
    
    def save(self, path: str) -> None:
        """Save HSM state."""
        import pickle
        state = {
            'orthogonal_states': self.orthogonal_hsm.get_all_states(),
            'history': self.history_manager.get_all_history(),
            'stats': self.get_statistics()
        }
        with open(f"{path}/hsm_state.pkl", 'wb') as f:
            pickle.dump(state, f)
            
        if self.learned_transitions:
            self.learned_transitions.save(f"{path}/learned_transitions.pt")
            
    def load(self, path: str) -> None:
        """Load HSM state."""
        import pickle
        with open(f"{path}/hsm_state.pkl", 'rb') as f:
            state = pickle.load(f)
            
        if self.learned_transitions:
            self.learned_transitions.load(f"{path}/learned_transitions.pt")
