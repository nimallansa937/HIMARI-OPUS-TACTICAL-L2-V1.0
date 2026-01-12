"""
HIMARI Layer 2 - Transformer-PPO Trainer
PPO trainer with clipped surrogate objective for stable training.

PPO ADVANTAGES OVER A2C:
1. Clipped objective prevents catastrophic policy updates
2. Multiple epochs per batch = better sample efficiency
3. More stable exploration in sparse reward settings
4. Trust region constraint via clipping

Based on:
- Schulman et al. (2017): Proximal Policy Optimization
- CleanRL PPO implementation
"""

import os
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
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


@dataclass
class PPOConfig:
    """
    PPO-specific hyperparameters.

    Key differences from A2C:
    - clip_range: Clipping parameter for surrogate objective
    - ppo_epochs: Number of epochs per rollout batch
    - num_minibatches: Split rollout into mini-batches
    - target_kl: Early stopping if KL divergence too high
    """
    # === PPO-SPECIFIC ===
    clip_range: float = 0.2           # PPO clipping parameter
    clip_range_vf: float = 0.2        # Value function clipping (None to disable)
    ppo_epochs: int = 10              # Epochs per rollout
    num_minibatches: int = 4          # Mini-batches per epoch
    target_kl: Optional[float] = 0.03 # Early stop if KL > target (None to disable)
    normalize_advantage: bool = True   # Normalize advantages

    # === ENTROPY (CRITICAL FOR EXPLORATION) ===
    entropy_coef: float = 0.05        # Start lower than A2C (PPO more stable)
    entropy_min: float = 0.01         # Minimum entropy coefficient
    entropy_decay: float = 0.995      # Decay per validation (slower than A2C)

    # === VALUE FUNCTION ===
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class TransformerPPOTrainer:
    """
    PPO trainer for Transformer-based trading agent.

    KEY FEATURES:
    1. Clipped surrogate objective for stable updates
    2. Multiple epochs per batch for sample efficiency
    3. Value function clipping to prevent large updates
    4. KL divergence monitoring with early stopping
    5. Transaction cost-aware training
    """

    def __init__(
        self,
        model_config: TransformerA2CConfig,
        ppo_config: PPOConfig,
        train_env,
        val_env,
        device: str = "cuda",
        output_dir: str = "./output/transformer_ppo",
        use_wandb: bool = False,
        wandb_project: str = "himari-layer2-transformer-ppo",
        # Transaction cost parameters
        use_transaction_costs: bool = True,
        trading_fee: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.model_config = model_config
        self.ppo_config = ppo_config
        self.train_env = train_env
        self.val_env = val_env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project

        os.makedirs(output_dir, exist_ok=True)

        # Initialize model (reuse TransformerA2C architecture)
        self.model = TransformerA2C(model_config).to(self.device)

        # Single optimizer for both actor and critic (PPO standard)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=model_config.actor_lr,
            weight_decay=model_config.weight_decay,
        )

        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=model_config.max_steps // model_config.rollout_steps,
            eta_min=model_config.actor_lr * 0.1,
        )

        # Reward function
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

        # Current entropy coefficient (will decay)
        self.current_entropy_coef = ppo_config.entropy_coef

        # Tracking
        self.global_step = 0
        self.best_val_sharpe = -float('inf')
        self.patience_counter = 0
        self.checkpoints = []

        # Metrics buffers
        self.train_sharpes = deque(maxlen=100)
        self.val_sharpes = deque(maxlen=20)
        self.kl_history = deque(maxlen=50)

        # Setup logging
        self._setup_logging()

        # Scheduling
        self.next_val_step = model_config.val_frequency
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
                config={
                    "model": asdict(self.model_config),
                    "ppo": asdict(self.ppo_config),
                },
                name=f"ppo_{self.model_config.hidden_dim}d_{self.model_config.num_layers}L",
            )

    def collect_rollout(self, env, steps: int) -> Dict:
        """Collect experience from environment."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        state, _ = env.reset()
        self.reward_fn.reset()

        for _ in range(steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(state_tensor, deterministic=False)

            action = output["action"].item()
            log_prob = output["log_prob"].item()
            value = output["value"].item()
            confidence = output["confidence"].item()

            next_state, market_return, done, info = env.step(action)
            reward = self.reward_fn.compute(action, market_return, confidence)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state

            if done:
                episode_sharpe = self.reward_fn.get_episode_sharpe()
                self.train_sharpes.append(episode_sharpe)

                if self.use_transaction_costs and hasattr(self.reward_fn, 'get_trade_count'):
                    trade_count = self.reward_fn.get_trade_count()
                    logger.debug(
                        f"Episode: sharpe={episode_sharpe:.4f}, trades={trade_count}"
                    )

                state, _ = env.reset()
                self.reward_fn.reset()

        # Get final value for bootstrapping
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            final_value = self.model(state_tensor)["value"].item()

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

        gamma = self.model_config.gamma
        gae_lambda = self.model_config.gae_lambda

        values_ext = np.append(values, final_value)
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
        """
        Perform PPO update with multiple epochs and mini-batches.

        Key differences from A2C:
        1. Clipped surrogate objective
        2. Multiple passes over data
        3. KL divergence monitoring
        4. Value function clipping
        """
        # Convert to tensors
        states = torch.FloatTensor(rollout["states"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        old_values = torch.FloatTensor(rollout["values"]).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        if self.ppo_config.normalize_advantage:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        batch_size = len(states)
        minibatch_size = batch_size // self.ppo_config.num_minibatches

        # Track metrics across epochs
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_kl_divs = []
        all_clip_fractions = []

        # Multiple epochs over the data
        for epoch in range(self.ppo_config.ppo_epochs):
            # Shuffle indices for each epoch
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_old_values = old_values[mb_indices]
                mb_returns = returns_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]

                # Evaluate current policy
                log_probs, values, entropy = self.model.evaluate_actions(mb_states, mb_actions)

                # === POLICY LOSS (Clipped Surrogate) ===
                ratio = torch.exp(log_probs - mb_old_log_probs)

                # Clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_config.clip_range,
                    1.0 + self.ppo_config.clip_range
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # === VALUE LOSS (with optional clipping) ===
                if self.ppo_config.clip_range_vf is not None:
                    # Clipped value function
                    values_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values,
                        -self.ppo_config.clip_range_vf,
                        self.ppo_config.clip_range_vf
                    )
                    value_loss_unclipped = (values - mb_returns) ** 2
                    value_loss_clipped = (values_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(values, mb_returns)

                # === ENTROPY LOSS ===
                entropy_loss = -entropy.mean()

                # === TOTAL LOSS ===
                loss = (
                    policy_loss +
                    self.ppo_config.value_coef * value_loss +
                    self.current_entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.ppo_config.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    # Approximate KL divergence
                    log_ratio = log_probs - mb_old_log_probs
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                    # Clipping fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.ppo_config.clip_range).float().mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(-entropy_loss.item())  # Store positive entropy
                all_kl_divs.append(approx_kl)
                all_clip_fractions.append(clip_fraction)

            # Early stopping based on KL divergence
            mean_kl = np.mean(all_kl_divs[-self.ppo_config.num_minibatches:])
            if self.ppo_config.target_kl is not None and mean_kl > self.ppo_config.target_kl:
                logger.info(f"Early stopping at epoch {epoch+1}/{self.ppo_config.ppo_epochs} due to KL={mean_kl:.4f}")
                break

        # Store KL for monitoring
        self.kl_history.append(np.mean(all_kl_divs))

        # Get current action probabilities for logging
        with torch.no_grad():
            batch_probs = F.softmax(self.model.actor(self.model.encoder(states[:512])), dim=-1)
            mean_probs = batch_probs.mean(dim=0)

        return {
            "policy_loss": np.mean(all_policy_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy": np.mean(all_entropy_losses),
            "entropy_coef": self.current_entropy_coef,
            "approx_kl": np.mean(all_kl_divs),
            "clip_fraction": np.mean(all_clip_fractions),
            "ppo_epochs_used": epoch + 1,
            "prob_flat": mean_probs[0].item(),
            "prob_long": mean_probs[1].item(),
            "prob_short": mean_probs[2].item(),
        }

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return Sharpe ratio."""
        self.model.eval()

        state, _ = self.val_env.reset()

        if self.use_transaction_costs:
            reward_fn = SortinoWithTransactionCosts(
                scale=100.0,
                trading_fee=self.trading_fee,
                slippage=self.slippage,
            )
        else:
            reward_fn = SimpleSortinoReward(scale=100.0)

        action_counts = {0: 0, 1: 0, 2: 0}

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

        # Log validation metrics
        returns = np.array(reward_fn._returns_buffer)
        total_actions = sum(action_counts.values())
        logger.info(
            f"Validation: returns_mean={np.mean(returns):.6f}, "
            f"returns_std={np.std(returns):.6f}, n={len(returns)}, "
            f"total_return={reward_fn.get_total_return():.4f}"
        )

        if self.use_transaction_costs and hasattr(reward_fn, 'get_trade_count'):
            trade_count = reward_fn.get_trade_count()
            total_costs = reward_fn.get_total_costs()
            gross_return = reward_fn.get_gross_return()
            net_return = reward_fn.get_net_return()
            avg_hold = total_actions / max(trade_count, 1)
            logger.info(
                f"Trading: trades={trade_count}, avg_hold={avg_hold:.1f} bars, "
                f"gross={gross_return*100:.2f}%, net={net_return*100:.2f}%, "
                f"costs={total_costs*100:.3f}%"
            )

        logger.info(
            f"Actions: FLAT={action_counts[0]/total_actions*100:.1f}%, "
            f"LONG={action_counts[1]/total_actions*100:.1f}%, "
            f"SHORT={action_counts[2]/total_actions*100:.1f}%"
        )

        # Warn on collapse
        max_action_pct = max(action_counts.values()) / total_actions
        if max_action_pct > 0.9:
            logger.warning(
                f"⚠️ POLICY COLLAPSE: {max_action_pct*100:.1f}% single action!"
            )

        self.val_sharpes.append(val_sharpe)

        # Decay entropy coefficient after validation
        self.current_entropy_coef = max(
            self.ppo_config.entropy_min,
            self.current_entropy_coef * self.ppo_config.entropy_decay
        )

        self.model.train()
        return val_sharpe

    def save_checkpoint(self, val_sharpe: float, tag: str = "") -> str:
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_{self.global_step}{tag}.pt"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)

        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "val_sharpe": val_sharpe,
            "current_entropy_coef": self.current_entropy_coef,
            "model_config": asdict(self.model_config),
            "ppo_config": asdict(self.ppo_config),
        }, checkpoint_path)

        self.checkpoints.append({
            "path": checkpoint_path,
            "step": self.global_step,
            "val_sharpe": val_sharpe,
        })

        # Keep only best N checkpoints
        self.checkpoints.sort(key=lambda x: x["val_sharpe"], reverse=True)
        while len(self.checkpoints) > self.model_config.keep_best_n:
            removed = self.checkpoints.pop()
            if os.path.exists(removed["path"]) and removed["path"] != checkpoint_path:
                try:
                    os.remove(removed["path"])
                except OSError:
                    pass

        logger.info(f"💾 Saved: {checkpoint_name} (val_sharpe={val_sharpe:.4f})")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_entropy_coef = checkpoint.get("current_entropy_coef", self.ppo_config.entropy_coef)

        logger.info(f"📂 Loaded checkpoint from step {self.global_step}")

    def check_early_stopping(self, val_sharpe: float) -> bool:
        """Check if training should stop early."""
        improved = val_sharpe > self.best_val_sharpe + self.model_config.min_improvement

        if improved:
            self.best_val_sharpe = val_sharpe
            self.patience_counter = 0
            self.save_checkpoint(val_sharpe, tag="_best")
            return False
        else:
            self.patience_counter += 1
            logger.warning(
                f"⚠️ No improvement for {self.patience_counter}/{self.model_config.patience} checks"
            )

            if self.patience_counter >= self.model_config.patience:
                logger.info("🛑 Early stopping triggered")
                return True
            return False

    def train(self) -> Optional[Dict]:
        """Main training loop with PPO updates."""
        logger.info("=" * 70)
        logger.info("Starting Transformer-PPO Training")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Max steps: {self.model_config.max_steps:,}")
        logger.info(f"PPO epochs: {self.ppo_config.ppo_epochs}")
        logger.info(f"Clip range: {self.ppo_config.clip_range}")
        logger.info(f"Initial entropy: {self.ppo_config.entropy_coef}")
        logger.info(f"Transaction costs: {self.use_transaction_costs}")
        logger.info("=" * 70)

        self.model.train()

        while self.global_step < self.model_config.max_steps:
            # Collect rollout
            rollout = self.collect_rollout(self.train_env, self.model_config.rollout_steps)

            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(rollout)

            # PPO update
            losses = self.update(rollout, returns, advantages)

            self.global_step += self.model_config.rollout_steps

            # Update learning rate
            self.lr_scheduler.step()

            # Logging
            if self.global_step >= self.next_log_step:
                self.next_log_step += 10000
                train_sharpe = np.mean(list(self.train_sharpes)[-10:]) if self.train_sharpes else 0
                logger.info(
                    f"Step {self.global_step:,}/{self.model_config.max_steps:,} | "
                    f"Train Sharpe: {train_sharpe:.4f} | "
                    f"Policy Loss: {losses['policy_loss']:.4f} | "
                    f"Value Loss: {losses['value_loss']:.4f}"
                )
                logger.info(
                    f"  Probs: FLAT={losses['prob_flat']:.3f}, "
                    f"LONG={losses['prob_long']:.3f}, SHORT={losses['prob_short']:.3f} | "
                    f"Entropy: {losses['entropy']:.4f} (coef={losses['entropy_coef']:.4f})"
                )
                logger.info(
                    f"  KL: {losses['approx_kl']:.4f} | "
                    f"Clip frac: {losses['clip_fraction']:.3f} | "
                    f"PPO epochs: {losses['ppo_epochs_used']}"
                )

                if self.use_wandb:
                    wandb.log({
                        "global_step": self.global_step,
                        "train_sharpe": train_sharpe,
                        **losses,
                    })

            # Validation
            if self.global_step >= self.next_val_step:
                self.next_val_step += self.model_config.val_frequency
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

                if self.check_early_stopping(val_sharpe):
                    break

            # Regular checkpoint
            if self.global_step % self.model_config.checkpoint_frequency == 0:
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

        # Save summary
        summary = {
            "total_steps": self.global_step,
            "best_val_sharpe": self.best_val_sharpe,
            "best_checkpoint": self.checkpoints[0]["path"] if self.checkpoints else None,
            "model_config": asdict(self.model_config),
            "ppo_config": asdict(self.ppo_config),
        }

        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_wandb:
            wandb.finish()

        return self.checkpoints[0] if self.checkpoints else None


# ==============================================================================
# Convenience Function
# ==============================================================================

def train_transformer_ppo(
    train_env,
    val_env,
    model_config: Optional[TransformerA2CConfig] = None,
    ppo_config: Optional[PPOConfig] = None,
    device: str = "cuda",
    output_dir: str = "./output/transformer_ppo",
    checkpoint_path: Optional[str] = None,
    use_wandb: bool = False,
    use_transaction_costs: bool = True,
    trading_fee: float = 0.001,
    slippage: float = 0.0005,
) -> Optional[Dict]:
    """
    Train Transformer-PPO model.

    Args:
        train_env: Training environment
        val_env: Validation environment
        model_config: Model configuration
        ppo_config: PPO-specific configuration
        device: Device to train on
        output_dir: Directory for checkpoints
        checkpoint_path: Optional checkpoint to resume from
        use_wandb: Whether to use Weights & Biases
        use_transaction_costs: Include transaction costs in reward
        trading_fee: Trading fee per trade
        slippage: Slippage estimate per trade

    Returns:
        Best checkpoint info
    """
    model_config = model_config or TransformerA2CConfig()
    ppo_config = ppo_config or PPOConfig()

    trainer = TransformerPPOTrainer(
        model_config=model_config,
        ppo_config=ppo_config,
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
