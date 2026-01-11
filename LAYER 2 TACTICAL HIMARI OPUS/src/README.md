# HIMARI Layer 2 - Source Code

## Overview

HIMARI Layer 2 Tactical Decision Engine with 14 integrated subsystems implementing 96+ methods.

## Subsystems

| Part | Directory | Description | Methods |
|------|-----------|-------------|---------|
| A | `preprocessing/` | Data preprocessing, normalization | 8 |
| B | `regime_detection/` | Market regime detection | 8 |
| C | `multi_timeframe/` | Multi-timeframe signal fusion | 8 |
| D | `decision_engine/` | RL agents, ensemble voting | 10 |
| E | `state_machine/` | Hierarchical FSM | 6 |
| F | `uncertainty/` | Uncertainty quantification | 8 |
| G | `hysteresis/` | Anti-whipsaw filtering | 6 |
| H | `risk_management/` | RSS risk management | 6 |
| I | `simplex_safety/` | Safety systems | 6 |
| J | `llm_integration/` | LLM-based analysis | 6 |
| K | `training/` | Training infrastructure | 8 |
| L | `training/` | Validation framework | 6 |
| M | `adaptation/` | Online adaptation | 6 |
| N | `interpretability/` | Model explainability | 4 |

## Quick Start

```python
from src.preprocessing import TradingKalmanFilter, VecNormalize
from src.regime_detection import IntegratedRegimeDetector
from src.decision_engine import DecisionEngine
```

## Architecture

Each subsystem follows consistent patterns:

- `*_pipeline.py` - Main orchestrator
- Config dataclasses for parameters
- Factory functions for easy instantiation
- Comprehensive `__init__.py` exports
