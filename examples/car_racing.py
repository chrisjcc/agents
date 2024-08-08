import gymnasium as gym
import numpy as np
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

torch.manual_seed(42)  #0


class Actor(nn.Module):
    def __init__(self, state_dim: Any, action_dim: int, max_action: float, hidden_dim: int = 256) -> None:
        super(Actor, self).__init__()   
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_out_size = self._get_conv_out(state_dim)
        self.fc1 = nn.Linear(self.conv_out_size, hidden_dim)  # 1536 / 128 * 11 * 11
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.std_fc = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def _get_conv_out(self, shape: Any) -> int:
        o = self.conv1(torch.zeros(1, *shape))
        return int(np.prod(o.size())/2)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Actor Forward'''
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(-1, self.conv_out_size)  # 1536 / 128 * 11 * 11

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.max_action * torch.tanh(self.mean_fc(x))
        std = F.softplus(self.std_fc(x))

        return mean, std
    
    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        # Choose action using actor network
        mean, std = actor(state)

        # Select action by subsampling from action space distribution
        distribution = Normal(loc=mean, scale=std)  # type: ignore
        action = distribution.sample()  # type: ignore

        return action

class Critic(nn.Module):
    def __init__(self, state_dim: Any, action_dim: int, hidden_dim: int=256) -> None:
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.conv_out_size = self._get_conv_out(state_dim)
        self.fc1 = nn.Linear(self.conv_out_size + action_dim, hidden_dim)  # 1536 / 128 * 11 * 11
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def _get_conv_out(self, shape: Any) -> int:
        o = self.conv1(torch.zeros(1, *shape))
        return int(np.prod(o.size())/2)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        action = action.view(action.size(0), -1) # reshape to (1, 3)
        x = x.reshape(x.size(0), -1)  # reshape to (1, 128*12*12)

        x = torch.cat([x, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        forward is the function that computes the Q-value 
        for a given state-action pair. evaluate is simply a wrapper around 
        forward that allows the critic to be used for both forward pass 
        and evaluation.
        """
        return self.forward(state, action)
    

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define environment hyperparameters
    env_name = "CarRacing-v2"
    render_mode = "human"
    max_episodes = 5 #1000
    randomize_domain = False  # reset with colour scheme change
    max_episode_steps = 600 # use less than the max to truncate episode not terminate

    # Initialize environment and model
    env = gym.make(env_name, 
        render_mode="human", 
        continuous=True,
        domain_randomize=randomize_domain,
        max_episode_steps=max_episode_steps,
    )

    # Number of Dimensions in the Observable Space and number of Control Actions in the Environments
    print(f'Observation Space: {env.observation_space}')
    print(f'Action Space: {env.action_space}')

    print("Observation Space Param: 96x96x3 values for Red, Green and Blue pixels")
    print(f'Observation Space Highs: {np.mean(env.observation_space.high)}')
    print(f'Observation Space Lows: {np.mean(env.observation_space.low)}')

    # Check if state is part of observation space
    state, info = env.reset(seed=42)
    print(f'Checking if the state is part of the observation space: {env.observation_space.contains(state)}')

    state = env.action_space.sample() # observation, _, _ = ...
    print(f'Checking if subsequent states are too part of the observation space: {env.observation_space.contains(state)}')

    # Actor-Critic typerparameters
    num_episodes = 3 #1000
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01

    # Initialize actor and critic networks
    state_dim = env.observation_space.shape #[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Convert action space range of low and high values to Tensors
    action_space_low = torch.from_numpy(np.array([-1.0, -0.0, 0.0], dtype=np.float32))
    action_space_high = torch.from_numpy(np.array([+1.0, +1.0, +1.0], dtype=np.float32))

    # Initialize Actor and Critic networks
    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    # Initialize optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    
    # Training loop
    for episode in range(num_episodes):
        # Reset the environment and get the initial state
        state, info = env.reset(seed=42)
        episode_reward = 0
        done = False

        while not done:
            # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
            #with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Choose action using actor network
            action_mean, action_std = actor(state_tensor)

            # Select action by subsampling from action space distribution
            action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore
            action = action_distribution.sample()  # type: ignore

            # Rescale the action to the range of teh action space
            rescaled_action = ((action + 1) / 2) * (
                action_space_high - action_space_low
            ) + action_space_low

            # Clip the rescaledaction to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(rescaled_action, action_space_low, action_space_high)

            # Take a step in the environment with the chosen action
            next_state, reward, terminated, truncated, info = env.step(clipped_action.squeeze().cpu().detach().numpy())
            done =  terminated or truncated
            episode_reward += reward

            # Compute TD error and update critic network
            reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)

            # Calculate Q-value(state, action)
            q_value = critic(state_tensor, clipped_action)

            # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
            #with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            # Choose action using actor network
            next_action_mean, next_action_std = actor(next_state_tensor)

            # Select action by subsampling from action space distribution
            next_action_distribution = Normal(loc=next_action_mean, scale=next_action_std)  # type: ignore
            next_action = next_action_distribution.sample()  # type: ignore

            # Rescale the action to the range of teh action space
            next_rescaled_action = ((next_action + 1) / 2) * (
                action_space_high - action_space_low
            ) + action_space_low

            # Clip the rescaledaction to ensure it falls within the bounds of the action space
            next_clipped_action = torch.clamp(next_rescaled_action, action_space_low, action_space_high)

            next_q_value = critic(next_state_tensor, next_clipped_action)

            td_error = reward + gamma * (1 - terminated) * next_q_value - q_value
            critic_loss = td_error ** 2

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Compute advantage and update actor network
            advantage = td_error.clone().detach().requires_grad_(True)
            actor_loss = -action_distribution.log_prob(clipped_action) * advantage - entropy_coef * action_distribution.entropy()

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            actor_loss_scalar = actor_loss.mean()
            actor_optimizer.zero_grad()
            actor_loss_scalar.backward()
            actor_optimizer.step()

            state = next_state

            if done:
                break

        print('Episode %d, reward: %f' % (episode, episode_reward))