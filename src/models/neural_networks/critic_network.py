# Importing necessary libraries
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        state_channel: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initializes the Critic network architecture.
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
            (1, state_channel, state_dim, state_dim)
        )
        self.fc1 = nn.Linear(action_dim, hidden_dim)  # for critic
        self.fc2 = nn.Linear(self.fc_input_dims + hidden_dim, hidden_dim)  # for Q-value
        self.fc3 = nn.Linear(hidden_dim, 1)  # for critic

    def calculate_conv_output_dims(
        self,
        input_dims: Tuple[int, int, int, int],
    ) -> int:
        state = torch.zeros(*input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(torch.prod(torch.tensor(dims.size())))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the Critic network, given an input state and action.
        Args:
        - state: A tensor of shape (batch_size, state_channel, height, width)
        - action: A tensor of shape (batch_size, action_dim)
        Returns:
        - A tensor of shape (batch_size,) containing the Q-value of the input state-action pair.
        """
        # Extract features from the state's image
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the 3D features tensor to make it suitable for feed-forward layers
        x = x.reshape(x.size(0), -1)

        # reshape to (batch_size, action_dim)
        action = action.view(-1, action.size(-1))
        action = F.relu(self.fc1(action))  # apply linear layer to action

        x = torch.cat([x, action], dim=1)  # concatenate action and feature tensors

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Q-value of a given state-action pair.
        Args:
        - state: A tensor of shape (batch_size, state_channel, height, width)
        - action: A tensor of shape (batch_size, action_dim)
        Returns:
        - A tensor of shape (batch_size,) containing the Q-value of the input state-action pair.
        """
        return self.forward(state, action)


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env_name: str = "CarRacing-v2"
    env: gym.Env[Any, Any] = gym.make(
        env_name,
        domain_randomize=True,
        continuous=True,
        render_mode="human",
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
    max_action = int(action_high[0])

    # Initialize Critic
    critic = Critic(
        state_dim=state_dim, state_channel=state_channel, action_dim=action_dim
    ).to(device)

    # Get state spaces
    state, info = env.reset()

    # Convert state numpy to tensor from shape [batch_size, height, width, channels] to [batch_size, channels, height, width]
    state_tensor = (
        torch.tensor(state, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
        .permute(0, 3, 1, 2)
    )

    # This loop constitutes one epoch
    while True:
        # Sample random action
        action = env.action_space.sample()

        # Convert from numpy to torch tensors, and send to device
        action_tensor = (
            torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # Evaluate Q-value of random state-action pair
        q_value = critic.evaluate(state_tensor, action_tensor)
        print(f"Q-value: {q_value.item():.3f}")

        # Take a step in the environment given sampled action
        next_state, reward, terminated, truncated, info = env.step(action)

        state_tensor = (
            torch.tensor(next_state, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
            .permute(0, 3, 1, 2)
        )

        # Update if the environment is done
        done = terminated or truncated
        if done:
            break
