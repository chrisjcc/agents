# Importing necessary libraries
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting the seed for reproducibility
torch.manual_seed(42)


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        state_channel: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initializes the Critic network architecture. We employ a Separate Critic Networks approach.
        In this approach, separate Critic networks are used to estimate the value function V(s) and the action-value function Q(s, a).
        Each network has its own set of parameters and is trained independently. As opposed to a Shared Network with Two Heads.
        In this approach, a shared neural network is used with two output heads, one for estimating V(s)
        and the other for estimating Q(s, a). The network shares some or all of its underlying layers,
        allowing for shared feature representations.

        :param state_dim: The number of dimensions in the state space.
        :param state_channel: The number of dimension in the state channel (e.g. RGB).
        :param action_dim: The number of dimensions in the action space.
        :param hidden_dim: The number of hidden units in the neural networks for actor and critic.
        """
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_channel, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Fully connected layers for policy approximation (1 batch)
        self.fc_input_dims = self.calculate_conv_output_dims(
            (state_channel, state_dim, state_dim)
        )
        self.fc0 = nn.Linear(self.fc_input_dims, hidden_dim)  # for state-value, V(s)
        self.fc1 = nn.Linear(self.fc_input_dims + action_dim, hidden_dim)  # for Q-value, Q(s,a)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # for both state-value and Q-value
        self.fc3 = nn.Linear(hidden_dim, 1)  # for state-value, V(s)

    def calculate_conv_output_dims(
        self,
        input_dims: Tuple[int, int, int, int],
    ) -> int:
        state = torch.zeros(*input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(torch.prod(torch.tensor(dims.size())))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the critic network and return the estimated state-value of the input state.
        Args:
        - state: A tensor of shape (batch_size, state_channel, height, width)
        Returns:
        - A tensor of shape (batch_size,) containing the Q-value of the input state.
        """

        # Scale the input state tensor to the appropriate range (e.g., [0, 1] or [-1, 1])
        state = (
            state / 255.0
        )  # Assuming the original range is [0, 255] for color channels

        # Extract features from the state's image
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the 3D features tensor to make it suitable for feed-forward layers
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc0(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the Q-value of a given state-action pair using the critic network.
        It takes the state and action as inputs and returns the Q-value estimate.
        Args:
        - state: A tensor of shape (batch_size, state_channel, height, width)
        - action: A tensor of shape (batch_size, action_dim)
        Returns:
        - A tensor of shape (batch_size,) containing the Q-value of the input state-action pair.
        """

        # Scale the input state tensor to the appropriate range (e.g., [0, 1] or [-1, 1])
        state = (
            state / 255.0
        )  # Assuming the original range is [0, 255] for color channels

        # Extract features from the state's image
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the 3D features tensor to make it suitable for feed-forward layers
        x = x.reshape(x.size(0), -1)

        x = torch.cat([x, action], dim=1)  # concatenate action and feature tensors

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the environment
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

    # Initialize Critic
    critic = Critic(
        state_dim=state_dim,
        state_channel=state_channel,
        action_dim=action_dim
    ).to(device)

    # Get state spaces
    state_ndarray, info = env.reset(seed=42)

    # Convert state numpy to tensor from shape
    # [batch_size, height, width, channels] to [batch_size, channels, height, width]
    state = (
        torch.tensor(state_ndarray, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
        .permute(0, 3, 1, 2)
    )

    # This loop constitutes one epoch
    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        print(f"Step: {step_count}")

        # Sample random action
        action_ndarray = env.action_space.sample()

        # Convert from numpy to torch tensors, and send to device
        action = torch.tensor(
            action_ndarray,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Evaluate Q-value of random state-action pair
        q_value = critic.evaluate(state, action)
        state_value = critic(state)
        print(f"\tQ-value: {q_value.item():.3f}")
        print(f"\tState-value: {state_value.item():.3f}")

        # Take a step in the environment given sampled action
        next_state_ndarray, reward, terminated, truncated, info = env.step(
            action_ndarray
        )

        next_state = (
            torch.tensor(next_state_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
            .permute(0, 3, 1, 2)
        )

        # Evaluate Q-value of random state-action pair
        next_state_value = critic(next_state)
        print(f"\tNext state-value: {next_state_value.item():.3f}")

        total_reward += float(reward)
        print(f"\tTotal reward: {total_reward:.2f}")

        state = next_state
        step_count += 1

        # Update if the environment is done
        done = terminated or truncated
