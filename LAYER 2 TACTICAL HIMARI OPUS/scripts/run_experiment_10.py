"""
HIMARI Layer 2 - Experiment 10: PPO with Anti-Overtrading on 1H Data

Run on Vast.ai:
    python scripts/run_experiment_10.py

Key differences from Experiment 9:
1. 1-hour data instead of 5-minute (2024-2026)
2. Anti-overtrading features:
   - Trade cooldown penalty (4 hours)
   - Action persistence bonus
   - Early exit penalty (min 2 hour hold)
3. Experiment 7 entropy settings + Experiment 9 carry cost (scaled for 1H)
"""

import sys
import os
import pickle
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.models.transformer_a2c import TransformerA2C, TransformerA2CConfig
from src.environment.transformer_a2c_env import TransformerA2CEnv, TransformerEnvConfig
from src.training.sortino_anti_overtrade import SortinoAntiOvertrade, create_exp10_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PPO Config from Experiment 7 (with modifications)
# ==============================================================================

class PPOConfig:
    """PPO configuration - Experiment 7 settings."""
    
    # Rollout
    rollout_steps: int = 2048
    num_envs: int = 1
    
    # PPO specific
    clip_range: float = 0.2
    ppo_epochs: int = 10
    num_minibatches: int = 4
    target_kl: float = 0.05  # Exp7: higher than default 0.03
    
    # Normalization
    normalize_advantage: bool = True
    
    # Entropy (Exp7 settings)
    entropy_coef: float = 0.10          # Start high
    entropy_min: float = 0.02           # Don't go too low
    entropy_decay: float = 0.998        # Slow decay (0.2%/update)
    
    # Value
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95


# ==============================================================================
# Trainer
# ==============================================================================

class Experiment10Trainer:
    """
    Experiment 10 - PPO with Anti-Overtrading on 1H data.
    """
    
    def __init__(
        self,
        train_data: np.ndarray,
        train_prices: np.ndarray,
        val_data: np.ndarray,
        val_prices: np.ndarray,
        device: str = "cuda",
        output_dir: str = "./output/exp10",
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model config
        self.model_config = TransformerA2CConfig(
            input_dim=44,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            context_length=100,
        )
        
        # PPO config (Exp7 settings)
        self.ppo = PPOConfig()
        
        # Environment config
        env_config = TransformerEnvConfig(
            context_length=100,
            feature_dim=44,
        )
        
        # Create environments
        self.train_env = TransformerA2CEnv(train_data, train_prices, env_config)
        self.val_env = TransformerA2CEnv(val_data, val_prices, env_config)
        
        # Create model
        self.model = TransformerA2C(self.model_config).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.ppo.learning_rate
        )
        
        # Create reward function (anti-overtrading)
        self.reward_fn = create_exp10_reward("1h")
        
        # Training state
        self.total_steps = 0
        self.best_val_sharpe = -np.inf
        self.current_entropy_coef = self.ppo.entropy_coef
        
        logger.info(f"Experiment 10 Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Train samples: {len(train_prices)}")
        logger.info(f"  Val samples: {len(val_prices)}")
        logger.info(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def collect_rollout(self, env, steps: int):
        """Collect experience from environment."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state, _ = env.reset()
        self.reward_fn.reset()
        
        for _ in range(steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(state_tensor)
                probs = output["probs"]
                value = output["value"]
                # Temperature sampling
                logits = torch.log(probs + 1e-8) / 0.5
                probs_temp = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs_temp)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Step environment
            next_state, market_return, done, info = env.step(action.item())
            
            # Compute reward with anti-overtrading
            reward = self.reward_fn.compute(
                action=action.item(),
                market_return=market_return,
                confidence=probs[0, action.item()].item()
            )
            
            # Store experience
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            state = next_state
            
            if done:
                state, _ = env.reset()
                self.reward_fn.reset()
        
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones),
        }
    
    def compute_returns_and_advantages(self, rollout):
        """Compute GAE returns and advantages."""
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.ppo.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.ppo.gamma * self.ppo.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def update(self, rollout, returns, advantages):
        """Perform PPO update."""
        states = torch.FloatTensor(rollout["states"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        
        if self.ppo.normalize_advantage:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO epochs
        batch_size = len(states)
        minibatch_size = batch_size // self.ppo.num_minibatches
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0
        
        for epoch in range(self.ppo.ppo_epochs):
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                
                # Forward pass - use evaluate_actions method
                new_log_probs, values, entropy_batch = self.model.evaluate_actions(mb_states, mb_actions)
                entropy = entropy_batch.mean()
                

                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo.clip_range, 1 + self.ppo.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (values.squeeze() - mb_returns).pow(2).mean()
                
                # Total loss
                loss = policy_loss + self.ppo.value_coef * value_loss - self.current_entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo.max_grad_norm)
                self.optimizer.step()
                
                # Track
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
                with torch.no_grad():
                    kl = (mb_old_log_probs - new_log_probs).mean().item()
                    total_kl += kl
                
                num_updates += 1
            
            # Early KL stopping
            if abs(total_kl / num_updates) > self.ppo.target_kl:
                logger.info(f"  KL divergence {total_kl/num_updates:.4f} > target, stopping epoch {epoch}")
                break
        
        # Decay entropy
        self.current_entropy_coef = max(
            self.ppo.entropy_min,
            self.current_entropy_coef * self.ppo.entropy_decay
        )
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl": total_kl / num_updates,
        }
    
    def validate(self, use_temperature: bool = True):
        """Run validation and return metrics."""
        state, _ = self.val_env.reset()
        self.reward_fn.reset()
        
        action_counts = {0: 0, 1: 0, 2: 0}
        total_reward = 0
        steps = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(state_tensor)
                probs = output["probs"]
                
                if use_temperature:
                    logits = torch.log(probs + 1e-8) / 0.5
                    probs_temp = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs_temp)
                    action = dist.sample().item()
                else:
                    action = probs.argmax(dim=-1).item()
            
            next_state, market_return, done, _ = self.val_env.step(action)
            
            reward = self.reward_fn.compute(action, market_return)
            total_reward += reward
            action_counts[action] += 1
            steps += 1
            
            state = next_state
            
            if done:
                break
        
        stats = self.reward_fn.get_stats()
        
        return {
            "sharpe": stats["sharpe"],
            "net_return": stats["net_return"] * 100,  # Percent
            "trade_count": stats["trade_count"],
            "action_distribution": {
                "FLAT": action_counts[0] / steps * 100,
                "LONG": action_counts[1] / steps * 100,
                "SHORT": action_counts[2] / steps * 100,
            },
            "anti_overtrade_stats": {
                "cooldown_penalties": stats["cooldown_penalties"] * 100,
                "persistence_bonuses": stats["persistence_bonuses"] * 100,
                "early_exit_penalties": stats["early_exit_penalties"] * 100,
            }
        }
    
    def save_checkpoint(self, val_sharpe: float, tag: str = ""):
        """Save model checkpoint."""
        filename = f"checkpoint_{self.total_steps}"
        if tag:
            filename += f"_{tag}"
        filename += ".pt"
        
        path = self.output_dir / filename
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "val_sharpe": val_sharpe,
            "entropy_coef": self.current_entropy_coef,
        }, path)
        
        logger.info(f"Saved checkpoint: {path}")
    
    def train(self, total_steps: int = 500_000, eval_interval: int = 20_000):
        """Main training loop."""
        logger.info("="*60)
        logger.info("EXPERIMENT 10: PPO with Anti-Overtrading on 1H BTC Data")
        logger.info("="*60)
        logger.info(f"Config:")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Eval interval: {eval_interval:,}")
        logger.info(f"  Entropy: {self.ppo.entropy_coef} -> {self.ppo.entropy_min}")
        logger.info(f"  Target KL: {self.ppo.target_kl}")
        logger.info("="*60)
        
        next_eval_step = eval_interval
        next_save_step = 50_000  # Save every 50k regardless
        
        while self.total_steps < total_steps:
            # Collect rollout
            rollout = self.collect_rollout(self.train_env, self.ppo.rollout_steps)
            returns, advantages = self.compute_returns_and_advantages(rollout)
            
            # Update
            update_stats = self.update(rollout, returns, advantages)
            
            self.total_steps += self.ppo.rollout_steps
            
            # Log training
            if self.total_steps % 4096 == 0:
                logger.info(
                    f"Step {self.total_steps:>7,} | "
                    f"PL: {update_stats['policy_loss']:.4f} | "
                    f"VL: {update_stats['value_loss']:.4f} | "
                    f"Ent: {update_stats['entropy']:.4f} | "
                    f"KL: {update_stats['kl']:.4f} | "
                    f"EntCoef: {self.current_entropy_coef:.4f}"
                )
            
            # Periodic checkpoint save (every 50k steps)
            if self.total_steps >= next_save_step:
                self.save_checkpoint(0.0, f"step_{self.total_steps}")
                next_save_step += 50_000
            
            # Validate (fixed: use >= instead of %)
            if self.total_steps >= next_eval_step:
                val_stats = self.validate()
                
                logger.info("-"*60)
                logger.info(f"VALIDATION @ {self.total_steps:,} steps:")
                logger.info(f"  Sharpe: {val_stats['sharpe']:.2f}")
                logger.info(f"  Net Return: {val_stats['net_return']:.1f}%")
                logger.info(f"  Trades: {val_stats['trade_count']:,}")
                logger.info(f"  Actions: FLAT={val_stats['action_distribution']['FLAT']:.1f}% | "
                           f"LONG={val_stats['action_distribution']['LONG']:.1f}% | "
                           f"SHORT={val_stats['action_distribution']['SHORT']:.1f}%")
                logger.info(f"  Anti-OT: Cooldown={val_stats['anti_overtrade_stats']['cooldown_penalties']:.4f}% | "
                           f"Persist={val_stats['anti_overtrade_stats']['persistence_bonuses']:.4f}% | "
                           f"EarlyExit={val_stats['anti_overtrade_stats']['early_exit_penalties']:.4f}%")
                logger.info("-"*60)
                
                # Save best
                if val_stats['sharpe'] > self.best_val_sharpe:
                    self.best_val_sharpe = val_stats['sharpe']
                    self.save_checkpoint(val_stats['sharpe'], "best")
                
                next_eval_step += eval_interval
        
        # Final save
        self.save_checkpoint(0.0, "final")
        
        logger.info("="*60)
        logger.info(f"Training complete! Best Sharpe: {self.best_val_sharpe:.2f}")
        logger.info("="*60)


# ==============================================================================
# Main
# ==============================================================================

def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cpu":
        logger.warning("No GPU detected! Training will be slow.")
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "btc_1h_2024_2026.pkl"
    
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        logger.error("Please run: python scripts/download_btc_1h_2024_2026.py first")
        return
    
    logger.info(f"Loading data from: {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    features = data["features"]
    prices = data["prices"]
    
    logger.info(f"Loaded {len(prices)} samples, {features.shape[1]} features")
    
    # Split: 60% train, 20% val, 20% test
    n = len(prices)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_features = features[:train_end]
    train_prices = prices[:train_end]
    val_features = features[train_end:val_end]
    val_prices = prices[train_end:val_end]
    test_features = features[val_end:]
    test_prices = prices[val_end:]
    
    logger.info(f"Train: {len(train_prices)} | Val: {len(val_prices)} | Test: {len(test_prices)}")
    
    # Create trainer
    trainer = Experiment10Trainer(
        train_data=train_features,
        train_prices=train_prices,
        val_data=val_features,
        val_prices=val_prices,
        device=device,
        output_dir="./output/exp10"
    )
    
    # Train
    trainer.train(
        total_steps=500_000,    # 500k steps
        eval_interval=20_000    # Eval every 20k
    )


if __name__ == "__main__":
    main()
