# Importing necessary libraries
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical, Normal

# Add the parent directory of 'agents.src' to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.neural_networks.actor_network import Actor  # type: ignore
from models.neural_networks.critic_network import Critic  # type: ignore

# Setting the seed for reproducibility
torch.manual_seed(0)


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env_name: str = "CarRacing-v2"
    num_episodes = 5  # 1000
    max_episode_steps = 600  # use less than the max to truncate episode not terminate

    env: gym.Env[Any, Any] = gym.make(
        env_name,
        domain_randomize=True,
        continuous=True,
        render_mode="human",
        max_episode_steps=max_episode_steps,
    )

    # Get state spaces
    state, info = env.reset()

    # We first check if state_shape is None. If it is None, we raise a ValueError.
    # Otherwise, we access the first element of state_shape using its index and
    # using the int() function.
    state_shape = env.observation_space.shape
    if state_shape is None:
        raise ValueError("Observation space shape is None.")
    state_dim = int(state_shape[0])

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
    max_action = int(action_high[0])
    action = env.action_space.sample()

    # Define the action space range
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -0.0, 0.0], dtype=np.float32),
        high=np.array([+1.0, +1.0, +1.0], dtype=np.float32),
        dtype=np.float32,
    )

    # Convert action_space low and high values to Tensors
    low = torch.from_numpy(action_space.low)
    high = torch.from_numpy(action_space.high)

    # Actor-Critic hyperparameters
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01

    # Initialize Actor policy and Critic networks
    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    # Initialize optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    # Training loop
    for episode in range(num_episodes):
        # Reset the environment and get the initial state
        state, info = env.reset(seed=42)

        episode_reward = 0.0
        done = False

        while not done:
            # with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # state_tensor /= 255.0  # normalize pixel values from [0, 255] to [0, 1]
            action_mean, action_std = actor(state_tensor)

            # Select action by subsampling from action space distribution
            action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore
            action = action_distribution.sample()  # type: ignore

            # Rescale the action to the range of teh action space
            rescaled_action = ((action + 1) / 2) * (
                action_space.high - action_space.low
            ) + action_space.low

            # Clip the rescaledaction to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(rescaled_action, low, high)

            # Convert from numpy to torch tensors, and send to device
            action = clipped_action.squeeze().cpu().detach().numpy()

            value = critic(state_tensor, torch.tensor([[action]], dtype=torch.float32))

            # Take a step in the environment with the chosen action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Compute TD error and update critic network
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            # next_state_tensor /= 255.0  # normalize pixel values from [0, 255] to [0, 1]

            # Select action by subsampling from action space distribution
            next_action_mean, next_action_std = actor(next_state_tensor)
            next_action_distribution = Normal(loc=next_action_mean, scale=next_action_std)  # type: ignore
            next_action = next_action_distribution.sample()  # type: ignore

            # Rescale the action to the range of teh action space
            rescaled_next_action = ((next_action + 1) / 2) * (
                action_space.high - action_space.low
            ) + action_space.low

            # Clip the rescaledaction to ensure it falls within the bounds of the action space
            clipped_next_action = torch.clamp(rescaled_next_action, low, high)

            # Convert from numpy to torch tensors, and send to device
            next_action = clipped_next_action.squeeze().cpu().detach().numpy()

            next_value = critic(
                next_state_tensor, torch.tensor([[next_action]], dtype=torch.float32)
            ).detach()

            td_error = (
                reward
                + gamma * next_value.detach().numpy()[0][0] * (1 - terminated)
                - value.detach().numpy()[0][0]
            )

            critic_loss_tensor = torch.tensor(
                td_error, dtype=torch.float32, requires_grad=True
            )
            critic_loss = critic_loss_tensor**2

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Compute advantage and update actor network
            advantage = td_error
            actor_loss = (
                -action_distribution.log_prob(torch.tensor(action, dtype=torch.float32))  # type: ignore
                * advantage
                - entropy_coef * action_distribution.entropy()  # type: ignore
            )
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss_scalar = actor_loss.mean()
            actor_optimizer.zero_grad()
            actor_loss_scalar.backward()
            actor_optimizer.step()

            episode_reward += float(reward)
            state = next_state
            print(f"Episode's cummulative reward: {episode_reward:.2f}")

            # Update if the environment is done
            if terminated or truncated:
                done = True
                break

        print("Episode %d, total reward: %f" % (episode, episode_reward))
