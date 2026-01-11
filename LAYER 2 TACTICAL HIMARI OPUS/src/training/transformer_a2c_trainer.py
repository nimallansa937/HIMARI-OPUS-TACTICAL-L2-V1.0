"""
HIMARI Layer 2 - Transformer-A2C Trainer
A2C trainer with staged training and early stopping.

KEY FEATURES (from lessons learned):
1. Validation-based early stopping
2. Best checkpoint selection by validation Sharpe
3. Train/val gap monitoring for overfitting detection
4. Simple Sortino reward (not complex shaping)
"""

import os
import json
import logging
from collections import deque
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.models.transformer_a2c import TransformerA2C, TransformerA2CConfig
from src.training.sortino_reward import SimpleSortinoReward, SortinoWithTransactionCosts

# Optional: wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class TransformerA2CTrainer:
    """
    A2C trainer with staged training and early stopping.
    
    KEY FEATURES (from lessons learned):
    1. Validation-based early stopping
    2. Best checkpoint selection by validation Sharpe
    3. Train/val gap monitoring for overfitting detection
    4. Simple Sortino reward (not complex shaping)
    """
    
    def __init__(
        self,
        config: TransformerA2CConfig,
        train_env,
        val_env,
        device: str = "cuda",
        output_dir: str = "./output/transformer_a2c",
        use_wandb: bool = False,
        wandb_project: str = "himari-layer2-transformer-a2c",
        # Transaction cost parameters
        use_transaction_costs: bool = True,
        trading_fee: float = 0.001,      # 0.1% per trade (Binance taker)
        slippage: float = 0.0005,        # 0.05% slippage estimate
    ):
        self.config = config
        self.train_env = train_env
        self.val_env = val_env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = TransformerA2C(config).to(self.device)
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.AdamW(
            list(self.model.encoder.parameters()) + list(self.model.actor.parameters()),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = optim.AdamW(
            self.model.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )
        
        # Reward function - use transaction costs to discourage over-trading
        self.use_transaction_costs = use_transaction_costs
        self.trading_fee = trading_fee
        self.slippage = slippage

        if use_transaction_costs:
            self.reward_fn = SortinoWithTransactionCosts(
                target_return=0.0,
                downside_penalty=2.0,
                scale=100.0,
                trading_fee=trading_fee,
                slippage=slippage,
            )
            logger.info(
                f"Using SortinoWithTransactionCosts: "
                f"fee={trading_fee*100:.2f}%, slippage={slippage*100:.2f}%"
            )
        else:
            self.reward_fn = SimpleSortinoReward(
                target_return=0.0,
                downside_penalty=2.0,
                scale=100.0,
            )
            logger.info("Using SimpleSortinoReward (no transaction costs)")
        
        # Tracking
        self.global_step = 0
        self.best_val_sharpe = -float('inf')
        self.patience_counter = 0
        self.checkpoints = []
        
        # Metrics buffers
        self.train_sharpes = deque(maxlen=100)
        self.val_sharpes = deque(maxlen=20)
        
        # Setup logging
        self._setup_logging()
        
        # Scheduling
        self.next_val_step = config.val_frequency
        self.next_log_step = 10000
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config=asdict(self.config),
                name=f"transformer_a2c_{self.config.hidden_dim}d_{self.config.num_layers}L",
            )
    
    def collect_rollout(self, env, steps: int) -> Dict:
        """Collect experience from environment."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state, _ = env.reset()
        self.reward_fn.reset()
        
        for _ in range(steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from model
            with torch.no_grad():
                output = self.model(state_tensor, deterministic=False)
            
            action = output["action"].item()
            log_prob = output["log_prob"].item()
            value = output["value"].item()
            confidence = output["confidence"].item()
            
            # Step environment
            next_state, market_return, done, info = env.step(action)
            
            # Compute reward
            reward = self.reward_fn.compute(action, market_return, confidence)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state

            if done:
                # Calculate and store training Sharpe for completed episode
                episode_sharpe = self.reward_fn.get_episode_sharpe()
                self.train_sharpes.append(episode_sharpe)

                # Log trade count if using transaction costs
                if self.use_transaction_costs and hasattr(self.reward_fn, 'get_trade_count'):
                    trade_count = self.reward_fn.get_trade_count()
                    total_costs = self.reward_fn.get_total_costs()
                    logger.debug(
                        f"Episode complete: train_sharpe={episode_sharpe:.4f}, "
                        f"trades={trade_count}, costs={total_costs*100:.3f}%"
                    )
                else:
                    logger.debug(f"Episode complete: train_sharpe={episode_sharpe:.4f}")

                state, _ = env.reset()
                self.reward_fn.reset()

        # Get final value for bootstrapping
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            final_value = self.model(state_tensor)["value"].item()

        # If no episode completed during rollout, calculate partial Sharpe for logging
        if len(self.reward_fn._returns_buffer) > 0:
            partial_sharpe = self.reward_fn.get_episode_sharpe()
            logger.debug(
                f"Rollout partial metrics: sharpe={partial_sharpe:.4f}, "
                f"returns_in_buffer={len(self.reward_fn._returns_buffer)}"
            )
        
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones),
            "final_value": final_value,
        }
    
    def compute_returns_and_advantages(self, rollout: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE returns and advantages."""
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        final_value = rollout["final_value"]
        
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        # Append final value for bootstrapping
        values_ext = np.append(values, final_value)
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * next_non_terminal - values_ext[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return returns, advantages
    
    def update(self, rollout: Dict, returns: np.ndarray, advantages: np.ndarray) -> Dict:
        """Perform A2C update."""
        # Convert to tensors
        states = torch.FloatTensor(rollout["states"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Evaluate actions with current policy
        log_probs, values, entropy = self.model.evaluate_actions(states, actions)
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages_t.detach()).mean()
        
        # Entropy bonus for exploration with DECAY SCHEDULE
        # Decay entropy coefficient from initial value to 30% of initial over training
        # This encourages exploration early, exploitation late
        progress = min(1.0, self.global_step / self.config.max_steps)
        current_entropy_coef = self.config.entropy_coef * (1 - 0.7 * progress)
        
        entropy_loss = -entropy.mean()
        
        # Critic loss (value function) - detach from actor computation graph
        critic_loss = F.mse_loss(values.detach(), returns_t)
        
        # Combined loss for logging only
        total_loss = (
            actor_loss +
            current_entropy_coef * entropy_loss +
            self.config.value_coef * critic_loss
        )
        
        # Update actor (encoder + actor head)
        self.actor_optimizer.zero_grad()
        actor_loss_with_entropy = actor_loss + current_entropy_coef * entropy_loss
        actor_loss_with_entropy.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.encoder.parameters()) + list(self.model.actor.parameters()),
            self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        
        # Re-evaluate for critic update (fresh forward pass)
        with torch.no_grad():
            # We need fresh encoded values for critic
            pass
        
        # For critic, we need to re-compute values with fresh forward pass
        # to avoid "modified by inplace operation" error
        _, values_fresh, _ = self.model.evaluate_actions(states, actions)
        critic_loss_fresh = F.mse_loss(values_fresh, returns_t)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss_fresh.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.critic.parameters(),
            self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        
        # Log action probabilities from this batch to detect collapse early
        with torch.no_grad():
            batch_probs = F.softmax(self.model.actor(self.model.encoder(states)), dim=-1)
            mean_probs = batch_probs.mean(dim=0)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss_fresh.item(),
            "entropy": entropy.mean().item(),
            "entropy_coef": current_entropy_coef,
            "total_loss": total_loss.item(),
            "prob_flat": mean_probs[0].item(),
            "prob_long": mean_probs[1].item(),
            "prob_short": mean_probs[2].item(),
        }
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return Sharpe ratio."""
        self.model.eval()

        state, _ = self.val_env.reset()

        # Use same reward function type as training for consistency
        if self.use_transaction_costs:
            reward_fn = SortinoWithTransactionCosts(
                scale=100.0,
                trading_fee=self.trading_fee,
                slippage=self.slippage,
            )
        else:
            reward_fn = SimpleSortinoReward(scale=100.0)

        # Track action distribution to detect policy collapse
        action_counts = {0: 0, 1: 0, 2: 0}  # FLAT, LONG, SHORT

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.model(state_tensor, deterministic=True)

            action = output["action"].item()
            confidence = output["confidence"].item()
            action_counts[action] += 1

            next_state, market_return, done, _ = self.val_env.step(action)
            reward_fn.compute(action, market_return, confidence)

            state = next_state

        val_sharpe = reward_fn.get_episode_sharpe()

        # DEBUG: Log validation metrics
        returns = np.array(reward_fn._returns_buffer)
        total_actions = sum(action_counts.values())
        logger.info(
            f"Validation metrics: returns_mean={np.mean(returns):.6f}, "
            f"returns_std={np.std(returns):.6f}, n_samples={len(returns)}, "
            f"total_return={reward_fn.get_total_return():.4f}, "
            f"max_dd={reward_fn.get_max_drawdown():.4f}"
        )

        # Log transaction cost metrics if applicable
        if self.use_transaction_costs and hasattr(reward_fn, 'get_trade_count'):
            trade_count = reward_fn.get_trade_count()
            total_costs = reward_fn.get_total_costs()
            gross_return = reward_fn.get_gross_return()
            net_return = reward_fn.get_net_return()
            avg_hold = total_actions / max(trade_count, 1)  # Average bars per trade
            logger.info(
                f"Trading metrics: trades={trade_count}, "
                f"avg_hold={avg_hold:.1f} bars ({avg_hold*5:.0f} min), "
                f"gross={gross_return*100:.2f}%, net={net_return*100:.2f}%, "
                f"costs={total_costs*100:.3f}%"
            )

        # Log action distribution to detect collapse
        logger.info(
            f"Action distribution: FLAT={action_counts[0]/total_actions*100:.1f}%, "
            f"LONG={action_counts[1]/total_actions*100:.1f}%, "
            f"SHORT={action_counts[2]/total_actions*100:.1f}%"
        )

        # Warn if policy is collapsing to single action
        max_action_pct = max(action_counts.values()) / total_actions
        if max_action_pct > 0.9:
            logger.warning(
                f"⚠️ POLICY COLLAPSE DETECTED: {max_action_pct*100:.1f}% single action! "
                f"Consider increasing entropy_coef or checking reward function."
            )

        self.val_sharpes.append(val_sharpe)

        self.model.train()
        return val_sharpe
    
    def save_checkpoint(self, val_sharpe: float, tag: str = "") -> str:
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_{self.global_step}{tag}.pt"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "val_sharpe": val_sharpe,
            "config": asdict(self.config),
        }, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append({
            "path": checkpoint_path,
            "step": self.global_step,
            "val_sharpe": val_sharpe,
        })
        
        # Keep only best N checkpoints
        self.checkpoints.sort(key=lambda x: x["val_sharpe"], reverse=True)
        while len(self.checkpoints) > self.config.keep_best_n:
            removed = self.checkpoints.pop()
            if os.path.exists(removed["path"]) and removed["path"] != checkpoint_path:
                try:
                    os.remove(removed["path"])
                except OSError:
                    pass
        
        logger.info(f"💾 Saved checkpoint: {checkpoint_name} (val_sharpe={val_sharpe:.4f})")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        
        logger.info(f"📂 Loaded checkpoint from step {self.global_step}")
    
    def check_early_stopping(self, val_sharpe: float) -> bool:
        """
        Check if training should stop early.
        
        Returns True if should stop.
        """
        improved = val_sharpe > self.best_val_sharpe + self.config.min_improvement
        
        if improved:
            self.best_val_sharpe = val_sharpe
            self.patience_counter = 0
            self.save_checkpoint(val_sharpe, tag="_best")
            return False
        else:
            self.patience_counter += 1
            logger.warning(
                f"⚠️ No improvement for {self.patience_counter}/{self.config.patience} checks"
            )
            
            if self.patience_counter >= self.config.patience:
                logger.info("🛑 Early stopping triggered")
                return True
            return False
    
    def check_overfitting(self) -> bool:
        """
        Detect overfitting by comparing train/val gap.
        
        Returns True if overfitting detected.
        """
        if len(self.train_sharpes) < 10 or len(self.val_sharpes) < 3:
            return False
        
        train_sharpe = np.mean(list(self.train_sharpes)[-20:])
        val_sharpe = np.mean(list(self.val_sharpes)[-3:])
        
        # If validation is less than 70% of training, likely overfitting
        if train_sharpe > 0.1 and val_sharpe < train_sharpe * 0.7:
            logger.warning(
                f"⚠️ Overfitting detected: train={train_sharpe:.4f}, val={val_sharpe:.4f}"
            )
            return True
        
        return False
    
    def train(self) -> Optional[Dict]:
        """
        Main training loop with staged training and early stopping.
        
        Returns:
            Best checkpoint info or None
        """
        logger.info("=" * 70)
        logger.info("Starting Transformer-A2C Training")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Max steps: {self.config.max_steps:,}")
        logger.info(f"Validation frequency: {self.config.val_frequency:,}")
        logger.info(f"Early stopping patience: {self.config.patience}")
        logger.info("=" * 70)
        
        self.model.train()
        
        while self.global_step < self.config.max_steps:
            # Collect rollout
            rollout = self.collect_rollout(self.train_env, self.config.rollout_steps)
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(rollout)
            
            # Update model
            losses = self.update(rollout, returns, advantages)
            
            self.global_step += self.config.rollout_steps
            
            # Logging
            # Logging
            if self.global_step >= self.next_log_step:
                self.next_log_step += 10000
                train_sharpe = np.mean(list(self.train_sharpes)[-10:]) if self.train_sharpes else 0
                logger.info(
                    f"Step {self.global_step:,}/{self.config.max_steps:,} | "
                    f"Train Sharpe: {train_sharpe:.4f} | "
                    f"Actor Loss: {losses['actor_loss']:.4f} | "
                    f"Critic Loss: {losses['critic_loss']:.4f}"
                )
                # Log action probabilities to detect early collapse
                logger.info(
                    f"  Policy probs: FLAT={losses['prob_flat']:.3f}, "
                    f"LONG={losses['prob_long']:.3f}, SHORT={losses['prob_short']:.3f} | "
                    f"Entropy: {losses['entropy']:.4f} (coef={losses['entropy_coef']:.4f})"
                )
                
                if self.use_wandb:
                    wandb.log({
                        "global_step": self.global_step,
                        "train_sharpe": train_sharpe,
                        **losses,
                    })
            
            # Validation check
            if self.global_step >= self.next_val_step:
                self.next_val_step += self.config.val_frequency
                val_sharpe = self.validate()
                train_sharpe = np.mean(list(self.train_sharpes)[-20:]) if self.train_sharpes else 0
                
                logger.info(
                    f"📊 Validation @ {self.global_step:,}: "
                    f"val_sharpe={val_sharpe:.4f}, train_sharpe={train_sharpe:.4f}"
                )
                
                if self.use_wandb:
                    wandb.log({
                        "val_sharpe": val_sharpe,
                        "train_val_gap": train_sharpe - val_sharpe,
                    })
                
                # Check early stopping
                if self.check_early_stopping(val_sharpe):
                    break
                
                # Check overfitting
                if self.check_overfitting():
                    logger.warning("Overfitting detected, stopping training")
                    break
            
            # Regular checkpoint
            if self.global_step % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(
                    np.mean(list(self.val_sharpes)[-3:]) if self.val_sharpes else 0
                )
        
        # Training complete
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Total steps: {self.global_step:,}")
        logger.info(f"Best validation Sharpe: {self.best_val_sharpe:.4f}")
        logger.info(f"Best checkpoint: {self.checkpoints[0]['path'] if self.checkpoints else 'N/A'}")
        
        # Save final training summary
        summary = {
            "total_steps": self.global_step,
            "best_val_sharpe": self.best_val_sharpe,
            "best_checkpoint": self.checkpoints[0]["path"] if self.checkpoints else None,
            "config": asdict(self.config),
        }
        
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        if self.use_wandb:
            wandb.finish()
        
        return self.checkpoints[0] if self.checkpoints else None


# ==============================================================================
# Convenience Functions
# ==============================================================================

def train_transformer_a2c(
    train_env,
    val_env,
    config: Optional[TransformerA2CConfig] = None,
    device: str = "cuda",
    output_dir: str = "./output/transformer_a2c",
    checkpoint_path: Optional[str] = None,
    use_wandb: bool = False,
    # Transaction cost parameters
    use_transaction_costs: bool = True,
    trading_fee: float = 0.001,
    slippage: float = 0.0005,
) -> Optional[Dict]:
    """
    Train Transformer-A2C model.

    Args:
        train_env: Training environment
        val_env: Validation environment
        config: Model configuration (uses defaults if None)
        device: Device to train on
        output_dir: Directory for checkpoints
        checkpoint_path: Optional checkpoint to resume from
        use_wandb: Whether to use Weights & Biases logging
        use_transaction_costs: Whether to include transaction costs in reward
        trading_fee: Trading fee per trade (default 0.1%)
        slippage: Slippage estimate per trade (default 0.05%)

    Returns:
        Best checkpoint info
    """
    config = config or TransformerA2CConfig()

    trainer = TransformerA2CTrainer(
        config=config,
        train_env=train_env,
        val_env=val_env,
        device=device,
        output_dir=output_dir,
        use_wandb=use_wandb,
        use_transaction_costs=use_transaction_costs,
        trading_fee=trading_fee,
        slippage=slippage,
    )

    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)

    return trainer.train()
