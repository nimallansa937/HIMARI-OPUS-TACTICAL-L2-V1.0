"""
HIMARI Layer 2 - Synchronized Events (E4)
Cross-region event coordination for orthogonal HSM.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyncEventConfig:
    synchronous: bool = True
    default_priority: int = 100
    continue_on_error: bool = True


@dataclass
class EventSubscription:
    handler: Callable
    priority: int = 100
    filter_condition: Optional[Callable] = None


class SynchronizedEventBus:
    """Event bus for cross-region coordination in orthogonal HSM."""
    
    def __init__(self, config: Optional[SyncEventConfig] = None):
        self.config = config or SyncEventConfig()
        self.subscriptions: Dict[str, List[EventSubscription]] = {}
        self.event_chains: Dict[str, List[str]] = {}
        self.event_history: List[tuple] = []
        
    def subscribe(self, event: str, handler: Callable, priority: int = 100,
                 filter_condition: Optional[Callable] = None) -> None:
        if event not in self.subscriptions:
            self.subscriptions[event] = []
            
        sub = EventSubscription(handler=handler, priority=priority,
                               filter_condition=filter_condition)
        self.subscriptions[event].append(sub)
        self.subscriptions[event].sort(key=lambda s: s.priority)
        
    def unsubscribe(self, event: str, handler: Callable) -> bool:
        if event not in self.subscriptions:
            return False
        original_len = len(self.subscriptions[event])
        self.subscriptions[event] = [s for s in self.subscriptions[event] 
                                     if s.handler != handler]
        return len(self.subscriptions[event]) < original_len
    
    def chain_events(self, trigger: str, chained: List[str]) -> None:
        self.event_chains[trigger] = chained
        
    def emit(self, event: str, data: Optional[Dict[str, Any]] = None,
            source: Optional[str] = None) -> Dict[str, Any]:
        data = data or {}
        results = {'event': event, 'source': source, 'handlers': []}
        
        self.event_history.append((event, data, source))
        
        if event not in self.subscriptions:
            return results
            
        for sub in self.subscriptions[event]:
            if sub.filter_condition and not sub.filter_condition(data):
                continue
                
            try:
                result = sub.handler(event, data)
                results['handlers'].append({
                    'handler': sub.handler.__name__,
                    'result': result, 'error': None
                })
            except Exception as e:
                logger.error(f"Handler {sub.handler.__name__} failed: {e}")
                results['handlers'].append({
                    'handler': sub.handler.__name__,
                    'result': None, 'error': str(e)
                })
                if not self.config.continue_on_error:
                    raise
                    
        if event in self.event_chains:
            for chained_event in self.event_chains[event]:
                chained_results = self.emit(chained_event, data, source=f"{event}->chain")
                results['chained'] = results.get('chained', [])
                results['chained'].append(chained_results)
                
        return results


class TradingEvents:
    """Standard trading events for HIMARI HSM."""
    BUY_SIGNAL = 'buy_signal'
    SELL_SIGNAL = 'sell_signal'
    ENTRY_CONFIRMED = 'entry_confirmed'
    EXIT_CONFIRMED = 'exit_confirmed'
    STOP_LOSS_TRIGGERED = 'stop_loss_triggered'
    
    TREND_DETECTED = 'trend_detected'
    TREND_REVERSAL = 'trend_reversal'
    VOLATILITY_SPIKE = 'volatility_spike'
    CRISIS_DETECTED = 'crisis_detected'
    CRISIS_RESOLVED = 'crisis_resolved'
    
    FORCE_LIQUIDATE = 'force_liquidate'
    REDUCE_EXPOSURE = 'reduce_exposure'
    HALT_TRADING = 'halt_trading'
    RESUME_TRADING = 'resume_trading'


def create_trading_event_bus() -> SynchronizedEventBus:
    bus = SynchronizedEventBus()
    bus.chain_events(TradingEvents.CRISIS_DETECTED, [
        TradingEvents.FORCE_LIQUIDATE, TradingEvents.HALT_TRADING
    ])
    bus.chain_events(TradingEvents.CRISIS_RESOLVED, [TradingEvents.RESUME_TRADING])
    return bus
