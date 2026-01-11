"""
HIMARI Layer 2 - SAC Agent
Subsystem D: Decision Engine (Method D3)

Purpose:
    Soft Actor-Critic agent with maximum entropy reinforcement learning.
    Provides exploration-focused counterbalance to exploitation-focused PPO.

Key Features:
    - Maximum entropy objective: reward + α · entropy
    - Automatic temperature tuning
    - Off-policy learning (sample efficient)
    - Excels in volatile/uncertain regimes

Expected Performance:
    - Sharpe 1.5-1.9 (lower than PPO but more robust)
    - Better performance in crisis/high-volatility regimes
    - Serves as ensemble diversity component

Reference:
    - Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
    - https://arxiv.org/abs/1801.01290
"""

from typing import Tuple, Optional
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium import spaces
from loguru import logger


class SACAgent:
    """
    Soft Actor-Critic agent for trading.

    Unlike PPO, SAC:
    - Learns off-policy (more sample efficient)
    - Maximizes entropy (more exploration)
    - Automatically tunes exploration vs exploitation

    This makes it ideal for:
    - Novel market regimes
    - High volatility periods
    - Ensemble diversity

    Example:
        >>> agent = SACAgent(state_dim=60, n_actions=3, device='cuda')
        >>> agent.learn(total_timesteps=500_000, env=trading_env)
        >>>
        >>> # Inference
        >>> action, confidence = agent.predict(observation)
    """

    def __init__(
        self,
        observation_space: Optional[spaces.Box] = None,
        action_space: Optional[spaces.Discrete] = None,
        state_dim: int = 60,
        n_actions: int = 3,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str = 'auto',  # Automatic temperature tuning
        target_entropy: str = 'auto',
        device: str = 'cuda',
        verbose: int = 1
    ):
        """
        Initialize SAC agent.

        Args:
            observation_space: Gym observation space
            action_space: Gym action space (Discrete for trading)
            state_dim: State dimension (if spaces not provided)
            n_actions: Number of actions (if spaces not provided)
            learning_rate: Learning rate for all networks
            buffer_size: Replay buffer size
            learning_starts: Steps before training starts
            batch_size: Minibatch size
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Update frequency
            gradient_steps: Gradient steps per env step
            ent_coef: Entropy coefficient (or 'auto' for automatic tuning)
            target_entropy: Target entropy (or 'auto')
            device: Device ('cuda' or 'cpu')
            verbose: Verbosity level
        """
        # Create spaces if not provided
        if observation_space is None:
            observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim,), dtype=np.float32
            )

        if action_space is None:
            action_space = spaces.Discrete(n_actions)

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        # Create SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=None,  # Will be set during training
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs={
                'net_arch': [256, 256]  # 2-layer MLP
            },
            device=device,
            verbose=verbose
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.model.policy.parameters())
        logger.info(f"SAC Agent initialized: {total_params/1e6:.2f}M parameters")

    def learn(
        self,
        total_timesteps: int,
        env=None,
        callback=None,
        log_interval: int = 10,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True
    ):
        """
        Train the SAC agent.

        Args:
            total_timesteps: Total training steps
            env: Training environment (required)
            callback: Optional callback
            log_interval: Episodes between logging
            tb_log_name: TensorBoard log name
            reset_num_timesteps: Reset counter

        Returns:
            self
        """
        if env is None:
            raise ValueError("Training environment is required")

        logger.info(f"Starting SAC training for {total_timesteps} timesteps")

        self.model.set_env(env)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps
        )

        logger.info("Training completed")
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        Predict action for observation.

        Args:
            observation: State vector
            deterministic: Use mean action vs sample from distribution

        Returns:
            action: Action index
            confidence: Action probability
        """
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)

        # Get action from model
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic
        )

        # For discrete actions, get probabilities
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).to(self.device)
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]

        action_int = int(action)
        confidence = float(probs[action_int])

        return action_int, confidence

    def get_action_distribution(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution.

        Args:
            observation: State vector

        Returns:
            probs: Action probabilities
        """
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).to(self.device)
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]

        return probs

    def get_entropy(self, observation: np.ndarray) -> float:
        """
        Get policy entropy for given observation.

        High entropy = more exploration
        Low entropy = more exploitation

        Args:
            observation: State vector

        Returns:
            entropy: Policy entropy (nats)
        """
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).to(self.device)
            distribution = self.model.policy.get_distribution(obs_tensor)
            entropy = distribution.distribution.entropy().cpu().item()

        return entropy

    def save(self, path: str):
        """Save model checkpoint"""
        self.model.save(path)
        logger.info(f"SAC model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        self.model = SAC.load(path, device=self.device)
        logger.info(f"SAC model loaded from {path}")


def create_sac_agent(
    state_dim: int = 60,
    n_actions: int = 3,
    device: str = 'cuda'
) -> SACAgent:
    """
    Factory function to create SAC agent.

    Args:
        state_dim: Observation dimension
        n_actions: Number of actions
        device: Computation device

    Returns:
        Configured SACAgent

    Example:
        >>> agent = create_sac_agent(state_dim=60, n_actions=3)
    """
    agent = SACAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device
    )

    return agent
