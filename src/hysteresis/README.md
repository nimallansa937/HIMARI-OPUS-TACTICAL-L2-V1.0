# Part G: Hysteresis Filter

6 methods for anti-whipsaw filtering.

## Methods

- G1: KAMA adaptive thresholds
- G2: KNN pattern matching
- G3: ATR-scaled bands
- G4: Meta-learned k values
- G5: 2.2× Loss aversion ratio
- G6: Whipsaw learning

## Usage

```python
from src.hysteresis import HysteresisFilter, LossAversionThresholds, WhipsawLearner
```
