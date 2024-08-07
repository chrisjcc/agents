# Importing necessary libraries
from typing import Any, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from pprint import pprint

from actor_critic_agent import ActorCriticAgent
from checkpoint_manager import CheckpointManager
from configuration_manager import ConfigurationManager
from data_logger import DataLogger
from replay_buffer.per import PrioritizedReplayBuffer
from replay_buffer.uer import UniformExperienceReplay
from trainer import Trainer
from utils.beta_scheduler import BetaScheduler

# Setting the seed for reproducibility
torch.manual_seed(0)


if __name__ == "__main__":
    """Highway-v0 Gym environment"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration file
    config_manager = ConfigurationManager("highway_config.yaml")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env: gym.Env[Any, Any] = gym.make(
        config_manager.env_name,
        render_mode=config_manager.render_mode,  # "human",
        max_episode_steps=config_manager.max_episode_steps,
    )

    action_type = "DiscreteMetaAction" #"ContinuousAction" #"DiscreteAction"  # "ContinuousAction"

    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted",
            "normalize": False
        },
        "action" :{
            "type": action_type
        },
        "duration": 20,
        "vehicles_count": 20,
        "collision_reward": -1,
        "high_speed_reward": 0.4
    }

    env.configure(config)

    pprint(env.config)
    print(list(env.action_type.actions_indexes.keys()))

    # We first check if state_shape is None. If it is None, we raise a ValueError.
    # Otherwise, we access the first element of state_shape using its index and
    # using the int() function.
    state_shape = env.observation_space.shape

    if state_shape is None:
        raise ValueError("Observation space shape is None.")
    state_dim = (15, 5) # env.observation_space.sample().shape

    # Get action spaces
    action_space = env.action_space

    action_dim = 1

    # Initialize the Checkpoint, location to store training agent model checkpoint
    checkpoint = CheckpointManager(
        checkpoint_dir=config_manager.checkpoint_dir,
        checkpoint_freq=config_manager.checkpoint_freq,
        num_checkpoints=config_manager.num_checkpoints,
    )

    # Initialize the replay buffer
    memory = PrioritizedReplayBuffer(
        capacity=config_manager.replay_buffer_capacity,
        alpha=config_manager.replay_buffer_alpha,
    )

    # Initialize Data logging
    data_logger = DataLogger()
    data_logger.initialize_writer()

    # Initialize Actor-Critic agent
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config_manager.hidden_dim,
        gamma=config_manager.gamma,
        lr=config_manager.lr,
        value_coef=config_manager.value_coef,
        entropy_coef=config_manager.entropy_coef,
        device=device,
        data_logger=data_logger,
        lr_step_size=config_manager.max_episode_steps,
        lr_gamma=config_manager.lr_gamma,
    )

    # Create trainer to train agent
    trainer = Trainer(
        env=env,
        agent=agent,
        memory=memory,
        max_episodes=config_manager.num_episodes,
        max_episode_steps=config_manager.max_episode_steps,
        batch_size=config_manager.batch_size,
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

