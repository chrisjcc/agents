from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

torch.manual_seed(42)  # 0


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: Any,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
    ) -> None:
        super(Actor, self).__init__()
        # self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_out_size = 1536  # self._get_conv_out(state_dim)
        self.fc1 = nn.Linear(self.conv_out_size, hidden_dim)  # 1536 / 128 * 11 * 11
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.std_fc = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def _get_conv_out(self, shape: Any) -> int:
        o = self.conv1(torch.zeros(1, *shape))
        return int(np.prod(o.size()) / 2)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actor Forward"""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        self.conv_out_size = 1536
        x = x.reshape(-1, self.conv_out_size)  # 1536 / 128 * 11 * 11

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.max_action * torch.tanh(self.mean_fc(x))
        std = F.softplus(self.std_fc(x))

        return mean, std


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: Any,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super(Critic, self).__init__()
        # self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_out_size = 1536  # self._get_conv_out(state_dim)
        self.fc1 = nn.Linear(
            self.conv_out_size + action_dim, hidden_dim
        )  # 1536 / 128 * 11 * 11
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def _get_conv_out(self, shape: Any) -> int:
        o = self.conv1(torch.zeros(1, *shape))
        return int(np.prod(o.size()) / 2)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        action = action.view(action.size(0), -1)  # reshape to (1, 3)
        x = x.reshape(x.size(0), -1)  # reshape to (1, 128*12*12)

        x = torch.cat([x, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Any:
        """
        forward is the function that computes the Q-value
        for a given state-action pair. evaluate is simply a wrapper around
        forward that allows the critic to be used for both forward pass
        and evaluation.
        """
        return self.forward(state, action)


# Define the ActorCritic architecture using the Actor and Critic network
class ActorCritic(nn.Module):
    """
    The ActorCritic class defines the complete actor-critic architecture.
    It consists of an Actor and a Critic neural network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
    ) -> None:
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return predictions from the  Actor and Critic networks, given a state tensor.

        :param state: A pytorch tensor representing the current state.
        :return: Pytorch Tensor representing the Actor network predictions and the Critic network predictions.
        """

        action, action_distribution = self.sample_action(state)

        state_value = self.critic(state, action)

        return action, action_distribution, state_value

    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the stochastic policy given a state.

        :param state: The current state of the environment.
        :return: The selected action.
        """
        # Sample action from actor network
        # with torch.no_grad():
        # Choose action using actor network
        action_mean, action_std = self.actor(state)

        # Select action by subsampling from action space distribution
        action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore
        action = action_distribution.sample()  # type: ignore

        return action, action_distribution


# Define the ActorCritic Agent
class ActorCriticAgent:
    """
    The ActorCriticAgent class defines an actor-critic reinforcement learning agent.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int,
        lr: float = 0.01,
        gamma: float = 0.9,
        seed: int = 42,
    ) -> None:
        """
        Initializes the ActorCriticAgent.

        :param state_dim: The number of dimensions in the state space.
        :param action_dim: The number of dimensions in the action space.
        :param hidden_dim: The number of hidden units in the neural networks for actor and critic.
        :param lr: The learning rate for the optimizer.
        :param gamma: The discount factor for future rewards.
        :param seed: The random seed for reproducibility.
        """
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=hidden_dim,
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Define the learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, threshold=1e-4, patience=5
        )

        self.gamma = gamma
        self.seed = seed
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.set_seed()
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def set_seed(self) -> None:
        """
        Set the seed value for generating random numbers within the environment.
        """
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Update the actor and critic neural networks given a new experience.

        :param state: The current state of the environment.
        :param action: The action taken in the current state.
        :param reward: The reward received from the environment.
        :param next_state: The next state of the environment.
        :param done: Whether the episode has ended.
        """
        # Select action by subsampling from action space distribution
        action, action_distribution, state_value = self.actor_critic(state)

        # Calculate Q-value given state
        _, _, next_state_value = self.actor_critic(next_state)

        # Compute the advantage estimates (discounted rewards - value), temporal-difference error
        discounted_rewards = reward + self.gamma * (1 - done.item()) * next_state_value
        advantage = discounted_rewards - state_value

        # Compute the value loss (A3C algorithm proposed by Mnih et al. (2016)
        # and the GAE variant of A2C proposed by Schulman et al. (2017)).
        # Huber loss is less sensitive to outliers than the mean squared
        # error loss, which can make it more robust in some cases.
        # Compute the value loss (original A2C paper by Mnih et al. (2016))
        # Calculate value loss ( F.smooth_l1_loss() which is the PyTorch equivalent of the Huber loss.)
        # critic_loss = advantage.pow(2)
        critic_loss = F.smooth_l1_loss(state_value, discounted_rewards)

        # Compute the actor policy loss
        actor_loss = -action_distribution.log_prob(action) * advantage

        # Compute entropy
        entropy = action_distribution.entropy()
        # entropy = (log_prob * torch.exp(log_prob))

        # Compute the total loss (later used to perform backpropagation)
        loss = self.value_coef * critic_loss + actor_loss - entropy_coef * entropy

        # aggregate the values across the tensor
        loss_mean = loss.mean()  # calculate the mean of the loss tensor
        # loss_scalar = torch.sum(loss)

        # Zero the gradients
        self.optimizer.zero_grad()

        # Compute the gradients
        loss_mean.backward()  # backpropagate through the network using the mean loss
        # loss_scalar.backward()
        # grads = torch.autograd.grad(
        #    loss_mean,
        #    self.actor_critic.parameters(),
        #    create_graph=True,
        #    retain_graph=True,
        # )

        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
        # torch.nn.utils.clip_grad_norm_(grads, max_norm=0.5)

        # Update the parameters
        self.optimizer.step()

        # Update the scheduler after every step
        self.lr_scheduler.step(loss_mean)

        # Check if the learning rate has been reduced
        # TODO: does not appear to be going down!
        # print("Learning rate:", self.optimizer.param_groups[0]['lr'])


if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define environment hyperparameters
    env_name = "CarRacing-v2"
    render_mode = "human"
    max_episodes = 5  # 1000
    randomize_domain = False  # reset with colour scheme change
    max_episode_steps = 600  # use less than the max to truncate episode not terminate

    # Initialize environment and model
    env = gym.make(
        env_name,
        render_mode="human",
        continuous=True,
        domain_randomize=randomize_domain,
        max_episode_steps=max_episode_steps,
    )

    # Number of Dimensions in the Observable Space and number of Control Actions in the Environments
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    print("Observation Space Param: 96x96x3 values for Red, Green and Blue pixels")
    print(f"Observation Space Highs: {np.mean(env.observation_space.high)}")
    print(f"Observation Space Lows: {np.mean(env.observation_space.low)}")

    # Check if state is part of observation space
    state, info = env.reset(seed=42)
    print(
        f"Checking if the state is part of the observation space: {env.observation_space.contains(state)}"
    )

    state = env.action_space.sample()  # observation, _, _ = ...
    print(
        f"Checking if subsequent states are too part of the observation space: {env.observation_space.contains(state)}"
    )

    # Actor-Critic typerparameters
    num_episodes = 3  # 1000
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01
    seed = 24
    hidden_dim = 256

    # Initialize actor and critic networks
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Convert action space range of low and high values to Tensors
    action_space_low = torch.from_numpy(np.array([-1.0, -0.0, 0.0], dtype=np.float32))
    action_space_high = torch.from_numpy(np.array([+1.0, +1.0, +1.0], dtype=np.float32))

    # Initialize Actor-Critic networks
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        seed=seed,
    )

    # Training loop
    for episode in range(num_episodes):
        # Reset the environment and get the initial state
        state, info = env.reset(seed=42)
        episode_reward = torch.FloatTensor([0.0]).unsqueeze(0).to(device)
        done = False

        while not done:
            # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
            # with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Choose action using actor network
            action, action_distribution = agent.actor_critic.sample_action(state_tensor)

            # Rescale the action to the range of teh action space
            rescaled_action = ((action + 1) / 2) * (
                action_space_high - action_space_low
            ) + action_space_low

            # Clip the rescaledaction to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(
                rescaled_action, action_space_low, action_space_high
            )

            # Take a step in the environment with the chosen action
            next_state, reward, terminated, truncated, info = env.step(
                clipped_action.squeeze().cpu().detach().numpy()
            )
            reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)

            # Unsqueeze to create a tensor of shape (1,1)
            done = terminated or truncated
            done = torch.tensor([done], dtype=torch.bool).unsqueeze(0).to(device)

            # Total episode reward
            episode_reward += reward

            # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
            # with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            # Compute TD error and update actor-critic network
            agent.update(state_tensor, clipped_action, reward, next_state_tensor, done)

            state = next_state

            if done:
                break

        print("Episode %d, reward: %f" % (episode, episode_reward))
