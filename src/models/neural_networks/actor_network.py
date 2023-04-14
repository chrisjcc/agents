# Importing necessary libraries
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

# Setting the seed for reproducibility
torch.manual_seed(0)


# Actor network
class Actor(nn.Module):
    """
    A neural network for the actor that predicts the mean and standard deviation
    of a normal distribution for selecting an action given a state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initializes the Actor network architecture.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_action (float): Maximum value of the action space.
            hidden_dim (int): Size of the hidden layers. Default is 256.
        """

        super(Actor, self).__init__()
        # Convolutional layers to extract features from the state's image
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Fully connected layers for policy approximation
        self.fc_input_dims = self.calculate_conv_output_dims(
            (state_dim, state_dim, action_dim)
        )
        self.fc1 = nn.Linear(self.fc_input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.std_fc = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def calculate_conv_output_dims(
        self,
        input_dims: Tuple[int, int, int],
    ) -> int:
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(torch.prod(torch.tensor(dims.size())))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor network.

        Args:
            state (torch.Tensor): Current state of the environment.

        Returns:
            A tuple with the predicted mean and standard deviation of the normal distribution.
        """

        # Extract features from the state's image
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the 3D features tensor to make it suitable for feed-forward layers
        x = x.reshape(x.size(0), -1)

        # Propagate through the dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Predict the mean of the normal distribution for selecting an action
        mean = self.max_action * torch.tanh(self.mean_fc(x))

        # Predict the standard deviation of the normal distribution for selecting an action
        std = F.softplus(self.std_fc(x))

        return mean, std


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

    # We first check if state_shape is None. If it is None, we raise a ValueError with an appropriate error message.
    # Otherwise, we access the first element of state_shape using its index and convert it to an integer
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

    # Initialize Actor policy
    actor = Actor(state_dim, action_dim, max_action).to(device)

    # Get state spaces
    state, info = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    total_reward = 0.0
    while True:
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
            action_mean, action_std = actor(state_tensor)

            # Select action by subsampling from action space distribution
            action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore
            action_tensor = action_distribution.sample()  # type: ignore

            # Rescale the action to the range of teh action space
            rescaled_action = ((action_tensor + 1) / 2) * (
                action_space.high - action_space.low
            ) + action_space.low

            # Clip the rescaledaction to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(
                rescaled_action,
                torch.from_numpy(action_space.low),
                torch.from_numpy(action_space.high),
            )

        # Convert from numpy to torch tensors, and send to device
        action = clipped_action.squeeze().cpu().detach().numpy()

        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state
        total_reward += float(reward)
        print(f"Total reward: {total_reward:.2f}")

        # Update if the environment is done
        if terminated or truncated:
            break
