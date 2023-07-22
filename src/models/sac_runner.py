# sac_runner.py
from typing import Any, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from checkpoint_manager import CheckpointManager
from configuration_manager import ConfigurationManager
from data_logger import DataLogger
from replay_buffer.per import PrioritizedReplayBuffer
from replay_buffer.uer import UniformExperienceReplay
from soft_actor_critic_agent import SACAgent
from trainer import Trainer
from utils.beta_scheduler import BetaScheduler

# Setting the seed for reproducibility
torch.manual_seed(0)


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration file
    config = ConfigurationManager("train_config.yaml")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env: gym.Env[Any, Any] = gym.make(
        config.env_name,
        domain_randomize=config.domain_randomize,  # True,
        continuous=config.continuous,  # True,
        render_mode=config.render_mode,  # "human",
        max_episode_steps=config.max_episode_steps,
    )

    # We first check if state_shape is None. If it is None, we raise a ValueError.
    # Otherwise, we access the first element of state_shape using its index and
    # using the int() function.
    state_shape = env.observation_space.shape

    if state_shape is None:
        raise ValueError("Observation space shape is None.")
    state_dim = int(state_shape[0])
    state_channel = int(state_shape[2])

    # Get action spaces
    action_space = env.action_space

    if isinstance(action_space, gym.spaces.Box):
        action_high = action_space.high
        action_shape = action_space.shape
    else:
        raise ValueError("Action space is not of type Box.")
    if action_shape is None:
        raise ValueError("Action space shape is None.")

    action_dim = int(action_shape[0])
    max_action = float(action_high[0])

    # Convert from numpy to tensor
    low = torch.from_numpy(action_space.low).to(device)
    high = torch.from_numpy(action_space.high).to(device)

    # Initialize the Checkpoint, location to store training agent model checkpoint
    checkpoint = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_freq=config.checkpoint_freq,
        num_checkpoints=config.num_checkpoints,
    )

    # Initialize the replay buffer
    memory = PrioritizedReplayBuffer(
        capacity=config.replay_buffer_capacity,
        alpha=config.replay_buffer_alpha,
    )

    # Initialize Data logging
    data_logger = DataLogger()
    data_logger.initialize_writer()

    # Initialize SAC agent
    agent = SACAgent(
        state_dim=state_dim,
        state_channel=state_channel,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=config.hidden_dim,
        device=device,
        lr=config.lr,
        gamma=config.gamma,
        tau=config.tau,
        alpha=config.alpha,
        replay_buffer_capacity=config.replay_buffer_capacity,
        replay_buffer_alpha=config.replay_buffer_alpha,
    )

    # Create trainer to train agent
    trainer = Trainer(
        env=env,
        agent=agent,
        memory=memory,
        max_episodes=config.num_episodes,
        max_episode_steps=config.max_episode_steps,
        batch_size=config.batch_size,
        low=low,
        high=high,
        device=device,
        checkpoint=checkpoint,
        data_logger=data_logger,
    )

    # Train the agent and get the episode rewards and reward standard deviations
    episode_rewards, reward_std_devs = trainer.train()

    # Plot the average reward with an uncertainty band (standard deviation)
    plt.plot(episode_rewards, label="Average Reward")
    plt.fill_between(
        range(len(episode_rewards)),
        [reward - std_dev for reward, std_dev in zip(episode_rewards, reward_std_devs)],
        [reward + std_dev for reward, std_dev in zip(episode_rewards, reward_std_devs)],
        alpha=0.3,
        label="Uncertainty Band",
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward with Uncertainty Band")
    plt.legend()
    plt.show()
