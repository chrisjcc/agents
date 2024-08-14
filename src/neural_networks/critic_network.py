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
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initializes the Critic network architecture.

        We employ a Separate Critic Networks approach. In this approach, separate Critic networks
        are used to estimate the state-value function V(s) and the action-value Q-value function Q(s, a).
        Each network has its own set of parameters and is trained independently.
        As opposed to a Shared Network with Two Heads. In this approach, a shared neural network
        is used with two output heads, one for estimating V(s) and the other for estimating Q(s, a).
        The network shares some or all of its underlying layers,
        allowing for shared feature representations.

        :param state_dim: The number of dimensions in the state space.
        :param action_dim: The number of dimensions in the action space.
        :param hidden_dim: The number of hidden units in the neural networks for the critic.
        """
        super(Critic, self).__init__()
        # Calculate the input dimension based on the state dimension
        input_dim = state_dim[0] * state_dim[1]

        # Q-value network
        self.q_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_fc3 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.q_fc4 = nn.Linear(hidden_dim, 1)

        # State-value network
        self.v_fc1 = nn.Linear(input_dim, hidden_dim)
        self.v_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        Perform a forward pass through the Critic network for state-value estimation.
        Args:
        - state: A tensor of shape (batch_size, state_dim)
        Returns:
        - A tensor of shape (batch_size,) containing the estimated value of the input state.
        """
        if action is not None:
            return self.q_value(state, action)
        else:
            return self.state_value(state)

    def q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s,a)
        """
        x = F.relu(self.q_fc1(state))
        x = F.relu(self.q_fc2(x))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.q_fc3(x))

        return self.q_fc4(x)

    def state_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute V(s)
        """
        x = F.relu(self.v_fc1(state))
        x = F.relu(self.v_fc2(x))

        return self.v_fc3(x)


if __name__ == "__main__":
    """Highway-v0 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the environment
    env_name: str = "highway-fast-v0"
    max_episode_steps = 600  # default

    env: gym.Env[Any, Any] = gym.make(
        env_name,
        render_mode="human",
        max_episode_steps=max_episode_steps,
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

    # We first check if state_shape is None. If it is None, we raise a ValueError.
    # Otherwise, we access the first element of state_shape using its index and
    # using the int() function.
    observation_shape = env.observation_space.shape

    if observation_shape is None:
        raise ValueError("Observation space shape is None.")

    state_dim = (15, 5) #env.observation_space.sample().shape

    # Get action spaces
    action_space = env.action_space
    action_dim = 1

    # Initialize Critic
    critic = Critic(
        state_dim=state_dim,
        action_dim=action_dim
    ).to(device)

    # Get state spaces
    state_ndarray, info = env.reset(seed=42)

    # Convert state numpy to tensor from shape
    state = torch.tensor(state_ndarray, dtype=torch.float32).to(device)

    # Flatten the state tensor to match the expected input dimension
    #state = state.flatten()
    #batch_size = 1
    #state = state.unsqueeze(0).expand(batch_size, -1) # e.g. [batch_size, 75]

    # This loop constitutes one epoch
    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        print(f"Step: {step_count}")

        # Flatten the state tensor to match the expected input dimension
        state = state.flatten()
        batch_size = 1
        state = state.unsqueeze(0).expand(batch_size, -1) # e.g. [batch_size, 75]

        # Sample random action
        action_ndarray = env.action_space.sample()

        # Convert from numpy to torch tensors, and send to device
        action = torch.tensor([action_ndarray], dtype=torch.float32).to(device)
        action = action.unsqueeze(0).expand(batch_size, -1)

        # Evaluate Q-value of random state-action pair
        state_value = critic(state)
        q_value = critic(state, action)

        print(f"\tState-value: {state_value.item():.3f}")
        print(f"\tQ-value: {q_value.item():.3f}")

        # Take a step in the environment given sampled action
        next_state_ndarray, reward, terminated, truncated, info = env.step(
            action_ndarray
        )

        next_state = torch.tensor(next_state_ndarray, dtype=torch.float32).to(device)

        # Flatten the state tensor to match the expected input dimension
        next_state = next_state.flatten()
        batch_size = 1
        next_state = next_state.unsqueeze(0).expand(batch_size, -1) # e.g. [batch_size, 75]

        # Evaluate Q-value of random state-action pair
        next_state_value = critic(next_state)
        print(f"\tNext state-value: {next_state_value.item():.3f}")

        total_reward += float(reward)
        print(f"\tTotal reward: {total_reward:.2f}")

        state = next_state
        step_count += 1

        # Update if the environment is done
        done = terminated or truncated
