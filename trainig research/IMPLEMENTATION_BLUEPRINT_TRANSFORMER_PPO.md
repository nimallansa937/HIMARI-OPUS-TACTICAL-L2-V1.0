# Implementation Blueprint: Transformer + PPO for BTC 5min Trading (2020-2024)

**Based on:** Deep analysis of theanh97, TomatoFT, Traxin3 repos  
**Target:** Achieve 1.5+ Sharpe on unseen 2023-2024 data

---

## PART 1: ENVIRONMENT DESIGN

### 1.1 Observation Space

```python
class BTCTradingEnv(gym.Env):
    def __init__(self, data, lookback=200):
        """
        data: (N, 5) array of [open, high, low, close, volume]
        lookback: 200 * 5min = 1000 minutes = ~16 hours
        """
        self.lookback = lookback
        self.data = self._normalize_ohlcv(data)
        
    def _normalize_ohlcv(self, data):
        """Normalize OHLCV to [-1, 1] range (critical for Transformer stability)"""
        returns = np.diff(np.log(data[:, 3]))  # log returns
        
        # Rolling normalization (prevent look-ahead)
        normalized = np.zeros_like(data, dtype=np.float32)
        for t in range(self.lookback, len(data)):
            window = data[t-self.lookback:t]
            # Use only historical data for normalization
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            normalized[t] = (data[t] - mean) / (std + 1e-8)
        
        return normalized
    
    def _get_observation(self, step):
        """Return (lookback, 5) Transformer input"""
        return self.data[step-self.lookback:step].astype(np.float32)
```

**Why this design:**
- 200 * 5min = ~16 hours of history (captures intraday patterns)
- Normalization per-window prevents non-stationarity leaking
- No future data in normalization (strict causality)

### 1.2 Action Space

**Recommendation: Discrete (LONG/FLAT/SHORT)**

```python
self.action_space = gym.spaces.Discrete(3)
# 0: LONG (+1.0 position)
# 1: FLAT (0.0 position)
# 2: SHORT (-1.0 position)

# In environment step:
action_to_position = {0: 1.0, 1: 0.0, 2: -1.0}
self.position = action_to_position[action]
```

**Why discrete over continuous:**
- More stable during training (PPO converges faster)
- Simpler to debug (3 fixed actions vs. continuous range)
- Reduces "thrashing" (oscillating between -0.1, 0.1, 0.2, ...)
- Still allows position management via entry/exit timing

### 1.3 Reward Function (CRITICAL - This is 80% of training success)

```python
def _calculate_reward(self, step):
    """
    Based on empirically successful formula from theanh97
    """
    
    # 1. REALIZED P&L
    price_change = self.data[step, 3] - self.data[step-1, 3]
    pnl = self.position * price_change / self.data[step-1, 3]
    
    # 2. TRANSACTION COSTS (Binance realistic)
    action_changed = int(self.position != self.prev_position)
    maker_fee = 0.0002  # BTC/USDT maker on Binance
    taker_fee = 0.0004
    cost = action_changed * (maker_fee + taker_fee) * abs(self.position)
    
    # 3. VOLATILITY PENALTY (prevent large drawdowns)
    realized_vol = np.std(np.diff(np.log(self.data[step-20:step, 3])))
    vol_penalty = -0.001 * realized_vol * abs(self.position)
    
    # 4. HOLDING COST (for crypto: funding rates, ~0.01% every 8 hours)
    hold_cost = -0.00000104 * abs(self.position)
    
    # 5. COMBINE (with normalization - CRITICAL)
    reward = (pnl - cost + vol_penalty + hold_cost)
    
    return np.float32(reward)

def _normalize_rewards(self, rewards):
    """
    Per-batch normalization to prevent scaling issues
    Applied in PPO update, not here
    """
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
```

**Key design decisions:**
- PnL in percentage terms (0.001 = 0.1%) → prevents scale explosion
- Action change tracking → penalizes overtrading
- Volatility penalty scales with position size → adaptive risk
- All terms in [-0.01, 0.01] range → balanced contribution

### 1.4 Episode Structure

```python
def step(self, action):
    if self.current_step >= len(self.data) - 1:
        done = True
    else:
        done = False
    
    self.prev_position = self.position
    self.position = {0: 1.0, 1: 0.0, 2: -1.0}[action]
    
    reward = self._calculate_reward(self.current_step)
    obs = self._get_observation(self.current_step)
    
    self.trades.append({
        'step': self.current_step,
        'action': action,
        'position': self.position,
        'close': self.data[self.current_step, 3],
        'reward': reward
    })
    
    self.current_step += 1
    return obs, reward, done, {}

def reset(self):
    self.current_step = self.lookback
    self.position = 0.0
    self.prev_position = 0.0
    self.trades = []
    return self._get_observation(self.current_step)
```

---

## PART 2: TRANSFORMER POLICY NETWORK

### 2.1 Architecture

```python
import torch
import torch.nn as nn

class TransformerPolicyNet(nn.Module):
    def __init__(self, obs_size=5, lookback=200, hidden_dim=128, num_heads=4):
        super().__init__()
        self.lookback = lookback
        
        # Input embedding (project OHLCV to hidden)
        self.embedding = nn.Linear(obs_size, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Positional encoding (for sequence ordering)
        self.pos_encoding = nn.Parameter(torch.randn(1, lookback, hidden_dim) * 0.02)
        
        # Output layers for PPO
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Value estimate
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Logits for 3 actions
        )
    
    def forward(self, obs):
        """
        obs: (batch, lookback, 5) tensor
        returns: (batch, 3) logits, (batch, 1) value
        """
        batch_size = obs.shape[0]
        
        # Embed observations
        x = self.embedding(obs)  # (batch, lookback, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer pass
        x = self.transformer(x)  # (batch, lookback, hidden_dim)
        
        # Use last token as aggregated representation
        x = x[:, -1, :]  # (batch, hidden_dim)
        
        # Actor & Critic heads
        logits = self.actor_head(x)  # (batch, 3)
        value = self.critic_head(x)  # (batch, 1)
        
        return logits, value.squeeze(-1)
```

---

## PART 3: WALK-FORWARD VALIDATION

### 3.1 Data Splits (Temporal, No Leakage)

```python
# 5min bars from 2020-2024
data = load_btc_5min_data('2020-01-01', '2024-12-31')

# CRITICAL: No future data, no overlapping

train_data = data['2020-01-01':'2022-12-31']  # 2 years
val_data = data['2023-01-01':'2023-06-30']    # 6 months (unseen)
test_data = data['2023-07-01':'2024-12-31']   # 6+ months (hold-out)

print(f"Train: {len(train_data)} bars ({len(train_data)*5/1440:.1f} days)")
print(f"Val:   {len(val_data)} bars")
print(f"Test:  {len(test_data)} bars")
```

### 3.2 Metrics

```python
class BacktestMetrics:
    def __init__(self, trades):
        self.trades = trades
    
    @property
    def sharpe(self):
        """Annualized Sharpe (252 trading days, 24/7 for crypto)"""
        returns = np.array([t['reward'] for t in self.trades])
        return np.mean(returns) / np.std(returns) * np.sqrt(252*24)
    
    @property
    def max_drawdown(self):
        """Peak-to-trough decline"""
        equity = np.cumprod(1 + np.array([t['reward'] for t in self.trades]))
        drawdown = (equity - np.max(equity)) / np.max(equity)
        return np.min(drawdown)
    
    @property
    def win_rate(self):
        """% of winning trades"""
        returns = np.array([t['reward'] for t in self.trades if t['reward'] != 0])
        return np.sum(returns > 0) / len(returns)
    
    @property
    def profit_factor(self):
        """Gross wins / Gross losses"""
        returns = np.array([t['reward'] for t in self.trades])
        wins = np.sum(returns[returns > 0])
        losses = np.abs(np.sum(returns[returns < 0]))
        return wins / (losses + 1e-8)
```

---

## PART 4: PRODUCTION CHECKLIST

Before going live:

- [ ] Train Sharpe ≥ 1.5 on 2020-2022
- [ ] Validation Sharpe ≥ 1.2 on 2023 Q1-Q2 (unseen)
- [ ] No OOD detected (val Sharpe > train * 0.8)
- [ ] Test Sharpe ≥ 1.0 on 2023 Q3-2024
- [ ] Win rate between 40-60% (not skewed)
- [ ] Max drawdown < 25%
- [ ] Profit factor > 1.3
- [ ] No look-ahead bias verified (walk-forward independent)
- [ ] Transaction costs include slippage (realistic)
- [ ] Deterministic policy used for final eval

---

## QUICK START (Copy-Paste)

```bash
# 1. Install dependencies
pip install stable-baselines3 gymnasium torch numpy pandas

# 2. Prepare data
python fetch_btc_5min.py --start 2020-01-01 --end 2024-12-31

# 3. Train
python train_transformer_ppo.py --train-end 2022-12-31 --val-end 2023-06-30

# 4. Backtest
python backtest.py --model ./models/transformer_ppo.zip --test-start 2023-07-01

# 5. Analyze
python analyze_results.py --trades ./backtest_trades.json
```

---

**Status:** Ready to implement  
**Expected Performance:** 1.5+ Sharpe on test set  
**Timeline:** 2-3 weeks from data to validation
