# Importing necessary libraries
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

# Setting the seed for reproducibility
torch.manual_seed(42)


# Actor network
class Actor(nn.Module):
    """
    A neural network for the actor that predicts the mean and standard deviation
    of a normal distribution for selecting an action given a state.
    """

    def __init__(
        self,
        state_dim: Any,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initializes the Actor network architecture.
        :param state_dim: The number of dimensions in the state space.
        :param action_dim: The number of dimensions in the action space.
        :param hidden_dim: The number of hidden units in the neural networks for actor and critic. Default is 256.
        """
        super(Actor, self).__init__()

        # Calculate the input dimension based on the state dimension
        input_dim = state_dim[0] * state_dim[1]

        # Feedforward layers to extract features from the state's image
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of the actor network.
        Args:
            state (torch.Tensor): Current state of the environment.
        Returns:
            A tuple with the predicted mean and standard deviation of the normal distribution.
        """
        # Scale the input state tensor to the appropriate range (e.g., [0, 1] or [-1, 1])
        # Propagate through the dense layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Predict the mean
        action_logits = self.mean_fc(x)

        return action_logits

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """
        Performs a forward pass using the actor network.
        :param state: The current state of the agent.
        :return: A tuple containing the selected action, its distribution and its estimated value.
        """
        # Sample action from actor network
        with torch.no_grad():
            # Sample an action from the actor network distribution
            action_logits = self.forward(state)

        # Apply softmax to convert logits to probabilities
        # Add a small constant to ensure positivity and numerical stability
        # caused by 'std' being too close to zero.
        #action_probs = F.softmax(action_logits, dim=1) + 1e-5

        # Check the row probability sum to approximiately 1.0
        #row_sums = action_probs.sum(dim=0)
        #target_sum = torch.ones_like(row_sums)

        #assert torch.allclose(row_sums, target_sum, rtol=1e-3, atol=1e-5), "Row sums are not approximately 1"

        # Create a categorical distribution over the action values
        action_distribution = Categorical(
            logits=action_logits
            #probs=action_probs
        )

        # Sample an action from the distribution
        action = action_distribution.sample()  # e.g. returns tensor(2)

        return action.unsqueeze(0), action_distribution  # TODO: why unsqueeze(0) needed here


if __name__ == "__main__":
    """Highway-v0 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Name of environment to be used
    env_name: str = "highway-fast-v0" #"highway-v0"
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

    # We first check if state_shape has a length greater than 0 using conditional statements.
    # Otherwise, we raise a ValueError with an appropriate error message.
    observation_shape = env.observation_space.shape
    if not observation_shape or len(observation_shape) == 0:
        raise ValueError("Observation space shape is not defined.")

    state_dim = (15, 5)

    # Get action spaces
    action_space = env.action_space

    # There are certain gym environments where action_shape is None. In such cases, we set
    # action_dim and max_action to None and use action_high directly.
    action_dim = int(action_space.n)

    # Initialize Actor policy
    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)

    # Get state spaces
    state_ndarray, info = env.reset(seed=42)
    state = torch.tensor(state_ndarray, dtype=torch.float32).to(device)

    state = state.flatten()
    batch_size = 1
    state = state.unsqueeze(0).expand(batch_size, -1) # e.g. [batch_size, 75]

    # This loop constitutes one epoch
    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        print(f"Step: {step_count}")

        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
             # Select action by subsampling from action space distribution
            action, action_distribution = actor.sample_action(state)

        next_state_ndarray, reward, terminated, truncated, info = env.step(
            action.squeeze().cpu().detach().numpy()
        )
        next_state = torch.tensor(next_state_ndarray, dtype=torch.float32).to(device)
        next_state = next_state.flatten()
        next_state = next_state.unsqueeze(0).expand(batch_size, -1) # e.g. [batch_size, 75]

        total_reward += float(reward)
        print(f"\tTotal reward: {total_reward:.2f}")

        state = next_state
        step_count += 1

        # Update if the environment is done
        done = terminated or truncated
