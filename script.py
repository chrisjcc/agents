
# Importing necessary libraries
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

    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 max_action: float, 
                 hidden_dim: int = 256
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
        self.state_dim = state_dim
        # Convolutional layers to extract features from the state's image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)  # , padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)  # , padding=1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # , padding=1

        # Fully connected layers for policy approximation
        self.fc1 = nn.Linear(128 * 11 * 11, hidden_dim) # (state_dim, ...)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.std_fc = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        x = x.reshape(-1, 128 * 11 * 11)

        # Propagate through the dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Predict the mean of the normal distribution for selecting an action
        # mean = self.max_action * F.softmax(self.mean_fc(x), dim=-1)
        mean = self.max_action * torch.tanh(self.mean_fc(x))

        # Predict the standard deviation of the normal distribution for selecting an action
        std = F.softplus(self.std_fc(x))

        return mean, std

if __name__ == "__main__":
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CarRacing-v2"
    # Passing continuous=True converts the environment to use continuous action space
    # The continuous action space has 3 actions: [steering, gas, brake].
    env = gym.make(
        env_name,
        domain_randomize=True,
        continuous=True,
        render_mode="human"
    )

    state, info = env.reset()

    state_dim = env.observation_space[0]
    action_dim = env.action_space[0]
    max_action = float(env.action_space.high[0])

    # Define the action space range
    action_space = gym.space.Box(
        low=np.array([-1., -0., 0.], dtype=np.float32),
        high=np.array([+1.0, +1.0, +1.0], dtype=np.float32),
        dtype=np.float32
    )

    # Actor policy
    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action).to(device)

    total_reward = 0.0
    while True:
        # Convert the list of numpy arrays to a single numpy array
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state)
        action_mean, action_std = actor(state_tensor)

        action_distribution = torch.distributions.Normal(action_mean, action_std)
        action = action_distribution.sample()

        # Rescale the action to the range of teh action space
        rescaled_action = ((action + 1) / 2) * (action_space.high - action_space.low) + action_space.low

        # Clip the rescaledaction to ensure it falls within the bounds of the action space
        clipped_action = np.clip(rescaled_action.numpy(), action_space.low, action_space.high)

        # Convert the clipped action back to a tensor
        action = torch.from_numpy(clipped_action).float()[0].detach().numpy()

        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state
        total_reward += reward
        print(f"Total reward: {total_reward:.2f}")

        # Update if the environment is done
        if terminated or turncated:
            break