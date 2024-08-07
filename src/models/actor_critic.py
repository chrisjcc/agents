# Importing necessary libraries
from typing import Any, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from neural_networks.actor_network import Actor
from neural_networks.critic_network import Critic

# Setting the seed for reproducibility
torch.manual_seed(0)


# Define the ActorCritic architecture using the Actor and Critic network
class ActorCritic(nn.Module):
    """
    The ActorCritic class defines the complete actor-critic architecture.
    It consists of an Actor and a Critic neural network.
    """

    def __init__(
        self,
        state_dim: int,
        state_channel: int,
        action_dim: int,
        max_action: float,
        device: Any,
        hidden_dim: int = 256,
    ) -> None:
        super(ActorCritic, self).__init__()

        # Initialize Actor policy
        self.actor = Actor(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
            max_action=max_action,
        ).to(device)

        # Initialize Critic
        self.critic = Critic(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
        ).to(device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal, torch.Tensor]:
        """
        Return predictions from the  Actor and Critic networks, given a state tensor.
        :param state: A pytorch tensor representing the current state.
        :return: Pytorch Tensor representing the Actor network predictions and the Critic network predictions.
        """
        mean, std = self.actor(state)

        action_dist = Normal(loc=mean, scale=std)
        # Generates a sample from the distribution using the reparameterization trick.
        # Supports gradient computation through the sampling process.
        # The reparameterization trick allows gradients to flow through the sampled values, making the sampling process differentiable.
        # Used during training when needed to backpropagate through the sampling process, such as in policy gradient methods
        action = action_dist.rsample()

        q_value = self.critic(state, action)

        return action, action_dist, q_value

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """
        Performs a forward pass using the actor network.
        :param state: The current state of the agent.
        :return: A tuple containing the selected action, its distribution and its estimated value.
        """
        # We're detaching the sampled action in sample_action to prevent gradients from flowing through it.
        with torch.no_grad():
            # Sample action from actor network
            mean, std = self.actor(state)

            # Sample an action from the distribution
            action_dist = Normal(loc=mean, scale=std)  # type: ignore
            action = action_dist.rsample()  # type: ignore

        return action, action_dist

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.actor(state)

        action_dist = Normal(loc=mean, scale=std)
        log_prob = action_dist.log_prob(action).sum(dim=-1)

        q_value = self.critic(state, action)

        return log_prob, q_value, action_dist.entropy().sum(dim=-1)


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Name the environment to use
    env_name: str = "CarRacing-v2"
    max_episode_steps = 600  # default

    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
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
    state_dim = int(env.observation_space.shape[0])
    state_channel = int(env.observation_space.shape[2])

    if state_dim is None:
        raise ValueError("Observation space shape is None.")

    # Get action spaces
    action_space = env.action_space
    action_dim = int(action_space.shape[0])
    max_action = float(action_space.high[0])

    low = torch.from_numpy(action_space.low).to(device)
    high = torch.from_numpy(action_space.high).to(device)

    # Actor-Critic hyperparameters
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01

    # Initialize Actor-Critic network
    actor_critic = ActorCritic(
        state_dim=state_dim,
        state_channel=state_channel,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
    )

    # Initialize optimizer
    actor_critic_optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    # Get state spaces
    state_ndarray, info = env.reset()

    # Convert state to shape (batch_size, channel, wdith, hight)
    state = (
        torch.tensor(state_ndarray, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
        .permute(0, 3, 1, 2)
    )

    # This loop constitutes one epoch
    total_reward = 0.0
    while True:
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
            # Obtain mean and std action given state
            action_tensor, action_distribution = actor_critic.sample_action(state)

            # Rescale, then clip the action to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(
                (((action_tensor + 1) / 2) * (high - low) + low), low, high
            )

        # Evaluate Q-value of state-action pair
        _, q_value, _ = actor_critic.evaluate(state, clipped_action)
        print(f"Q-value(state,action): {q_value.item():.3f}")

        # Take one step in the environment given the agent action
        next_state_ndarray, reward_ndarray, terminated, truncated, info = env.step(
            clipped_action.squeeze().cpu().detach().numpy()
        )

        # Convert to tensor
        next_state = (
            torch.tensor(next_state_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
            .permute(0, 3, 1, 2)
        )

        reward = (
            torch.tensor(reward_ndarray, dtype=torch.float32).unsqueeze(0).to(device)
        )
        terminated = torch.tensor(terminated, dtype=torch.float32).unsqueeze(0).to(device)
        truncated = torch.tensor(truncated, dtype=torch.float32).unsqueeze(0).to(device)

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = actor_critic.sample_action(next_state)

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_next_action = torch.clamp(
            (((next_action + 1) / 2) * (high - low) + low), low, high
        )

        # Evaluate Q-value of next state-action pair
        _, next_q_value, _ = actor_critic.evaluate(next_state, clipped_next_action)
        print(f"Next Q-value(next_state,next_action): {q_value.item():.3f}")

        # Create an identity tensor with the same shape and device as state_value
        identity_tensor = torch.ones_like(reward, dtype=torch.float32).to(device)

        # Calculate target Q-value
        target_q_value = reward + gamma * (identity_tensor - terminated) * next_q_value

        # Calculate critic loss
        critic_loss = F.smooth_l1_loss(target_q_value, q_value)

        # Calculate advantage
        advantage = target_q_value - q_value

        # Compute entropy
        entropy = torch.mean(action_distribution.entropy())  # type: ignore

        # Calculate actor loss
        action_log_prob = next_action_distribution.log_prob(clipped_next_action)  # type: ignore
        actor_loss = -torch.mean(action_log_prob * advantage)

        # Calculate total loss
        loss = value_coef * critic_loss + actor_loss - entropy_coef * entropy

        # Zero out gradients
        actor_critic_optimizer.zero_grad()

        # Calculate backprogation
        loss.backward()  # type: ignore

        # Apply gradient norm clipping
        torch.nn.utils.clip_grad_norm_(
            actor_critic.parameters(), max_norm=0.5, norm_type=2
        )

        # Apply update rule to neural network weights
        actor_critic_optimizer.step()

        state = next_state

        total_reward += float(reward)
        print(f"Total reward: {total_reward:.2f}")

        # Update if the environment is done
        done = terminated or truncated
        if done:
            break
