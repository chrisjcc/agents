# Importing necessary libraries (NOT WORKING)
import os
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal

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
    max_action = float(action_high[0])
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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        episode_reward = 0.0
        done = False

        while not done:
            #with torch.no_grad():
            # state_tensor /= 255.0  # normalize pixel values from [0, 255] to [0, 1]
            action_mean, action_std = actor(state_tensor)

            # Select action by subsampling from action space distribution
            action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore
            action_tensor = action_distribution.sample()  # type: ignore

            # Rescale, then clip the action to ensure it falls within the bounds 
            # of the action space
            clipped_action = torch.clamp(
            (((action_tensor + 1) / 2) * (high - low) + low), 
                low, high
            ).to(device)

            # Q-value(s,a) calculation
            q_value = critic(state_tensor, clipped_action)

            # Convert from numpy to torch tensors, and send to device
            action = clipped_action.squeeze().cpu().detach().numpy()

            # Take a step in the environment with the chosen action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Convert to tensor
            next_state_tensor = (
                torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            )
            #next_state_tensor /= 255.0  # normalize pixel values from [0, 255] to [0, 1]

            # Select action by subsampling from action space distribution
            next_action_mean, next_action_std = actor(next_state_tensor)

            # Select action by subsampling from action space distribution
            next_action_distribution = Normal(loc=next_action_mean, scale=next_action_std)  # type: ignore
            next_action_tensor = next_action_distribution.sample()  # type: ignore

            # Rescale, then clip the action to ensure it falls within the bounds of the action space
            clipped_next_action = torch.clamp(
                (((next_action_tensor + 1) / 2) * (high - low) + low), 
                low, high
            ).to(device)

            # Q-value(s', a') calculation
            next_q_value = critic(next_state_tensor, clipped_next_action)

            # Calculate target Q-value
            target_q_value = reward + gamma * (1 - terminated) * next_q_value

            # Calculate critic loss
            critic_loss = F.smooth_l1_loss(target_q_value, q_value)

            # Calculate advantage
            advantage = target_q_value - q_value

            # Advantage normalization can improve efficiency of gradient
            epsilon = 1e-8
            advantage_norm = (advantage - advantage.mean()) / (advantage.std() + epsilon)

            # Compute entropy
            entropy = torch.mean(action_distribution.entropy())  # type: ignore

            # Calculate actor loss
            action_log_prob = next_action_distribution.log_prob(clipped_next_action)  # type: ignore
            actor_loss = -torch.mean(action_log_prob * advantage_norm)

            # Calculate total loss
            loss = value_coef * critic_loss + actor_loss - entropy_coef * entropy

            # Zero out gradients
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()

            # Calculate backpropagation
            loss.backward()

            # Apply gradient norm clipping
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), max_norm=0.5, norm_type=2
            )
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), max_norm=0.5, norm_type=2
            )

            # Update network weights
            critic_optimizer.step()
            actor_optimizer.step()

            state = next_state
            state_tensor = next_state_tensor

            episode_reward += float(reward)
            print(f"Total reward: {episode_reward:.2f}")

            # Update if the environment is done
            done = terminated or truncated
            if done:
                break

        print("Episode %d, total reward: %f" % (episode, episode_reward))
