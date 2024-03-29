# Import libraries
import os
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)


# Define the Actor network
class Actor(nn.Module):
    """
    The Actor class defines a neural network that maps state to actions.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(Actor, self).__init__()
        self.actor_linear1 = nn.Linear(state_dim, hidden_dim)
        self.actor_linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return the actor network predictions, given a state tensor.

        :param state: A pytorch tensor representing the current state.
        :return: Pytorch tensor representing the actor network predictions.
        """
        x = self.actor_linear1(state)
        x = F.relu(x)
        x = self.actor_linear2(x)
        x = F.softmax(x, dim=-1)

        return x


# Define the Critic network
class Critic(nn.Module):
    """
    The Critic class defines a neural network that estimates the value function.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(Critic, self).__init__()
        self.critic_linear1 = nn.Linear(state_dim, hidden_dim)
        self.critic_linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return the critic network predictions, given a state tensor.

        :param state: A pytorch tensor representing the current state.
        :return: Pytorch tensor representing the critic network predictions.
        """
        x = self.critic_linear1(state)
        x = F.relu(x)
        x = self.critic_linear2(x)

        return x


# Define the ActorCritic architecture using the Actor and Critic network
class ActorCritic(nn.Module):
    """
    The ActorCritic class defines the complete actor-critic architecture.
    It consists of an Actor and a Critic neural network.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return predictions from the  Actor and Critic networks, given a state tensor.

        :param state: A pytorch tensor representing the current state.
        :return: Pytorch Tensor representing the Actor network predictions and the Critic network predictions.
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)

        return action_probs, state_value


# Define the ActorCritic Agent
class ActorCriticAgent:
    """
    The ActorCriticAgent class defines an actor-critic reinforcement learning agent.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
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
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
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

    def set_seed(self) -> None:
        """
        Set the seed value for generating random numbers within the environment.
        """
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    def sample_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the stochastic policy given a state.

        :param state: The current state of the environment.
        :return: The selected action.
        """
        action_probs, _ = self.actor_critic(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        return action

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
        action_probs, state_value = self.actor_critic(state)

        _, next_state_value = self.actor_critic(next_state)

        delta = reward + self.gamma * next_state_value * (1 - done) - state_value

        actor_loss = -torch.log(action_probs[0][action]) * delta
        critic_loss = delta**2
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the scheduler after every step
        self.lr_scheduler.step(loss.item())

        # Check if the learning rate has been reduced
        # TODO: does not appear to be going down!
        # print("Learning rate:", self.optimizer.param_groups[0]['lr'])


# Trainer class to train the Agent
class Trainer:
    """
    The Trainer class defines the training loop for the actor-critic agent.
    """

    def __init__(
        self,
        env: Any,
        agent: ActorCriticAgent,
        max_episodes: int,
        max_steps: int,
        checkpoint_path: str = "model_checkpoints",
        checkpoint_freq: int = 100,
    ):
        """
        Initializes the Trainer.

        :param env: The OpenAI Gym environment.
        :param agent: The actor-critic agent.
        :param max_episodes: The maximum number of episodes to train for.
        :param max_steps: The maximum number of steps per episode.
        :param checkpoint_path: The directory to save the agent's model in.
        :param checkpoint_freq: The frequency (in episodes) at which to save the agent's model.
        """
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq

    def train_step(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a single training step in the OpenAI Gym environment.

        :param state: The current state of the environment.
        :return: The next state, reward, and whether the episode is done.
        """
        action = self.agent.sample_action(state)
        next_state, reward, terminated, truncated, info = self.env.step(action.item())

        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward]).unsqueeze(0)
        done = terminated or truncated
        done = torch.FloatTensor([done]).unsqueeze(0)

        self.agent.update(state, action, reward, next_state, done)

        return next_state, reward, done

    def train(self) -> List[int]:
        """
        Run the training loop for the actor-critic agent.

        :return: A list of episode rewards.
        """
        episode_rewards = []
        for episode in range(self.max_episodes):
            state, info = self.env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)

            episode_reward = 0
            for step in range(self.max_steps):
                state, reward, done = self.train_step(state)
                episode_reward += reward.item()

                if done:
                    break

            episode_rewards.append(episode_reward)
            print(f"Episode {episode}: reward = {episode_reward}")

            # Save model checkpoint after each 10 episode
            if episode % self.checkpoint_freq == 0:
                self.save_checkpoint(self.checkpoint_path, episode, episode_reward)

            # Save last episode model
            self.save_checkpoint(
                self.checkpoint_path, self.max_episodes, episode_reward
            )

        return episode_rewards

    def save_checkpoint(
        self,
        checkpoint_path: str,
        episode_num: int,
        episode_reward: int,
    ) -> None:
        """
        Save the current state of the agent to a file.

        :param checkpoint_path: path to checkpoint directory
        :param episode_num: The current episode number.
        :param episode_reward: The current episode reward.
        """
        checkpoint = {
            "actor_critic_state_dict": self.agent.actor_critic.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "episode": episode_num,
            "reward": episode_reward,
        }
        os.makedirs(checkpoint_path, exist_ok=True)
        fpath = os.path.join(checkpoint_path, f"checkpoint_{episode_num}.pth")
        torch.save(checkpoint, fpath)


if __name__ == "__main__":
    # Loation to store training agent model checkpoint
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Source: https://gymnasium.farama.org/environments/classic_control/cart_pole/
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="human")  # "rgb_array"

    # Action Space: Discrete(2)
    # Observation Space: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38],
    # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create the agent
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=0.01,
        gamma=0.99,
        seed=seed,
    )

    # Create a trainer for the agent
    trainer = Trainer(
        env,
        agent,
        max_episodes=300,
        max_steps=200,
        checkpoint_path=checkpoint_dir,
    )

    episode_rewards = trainer.train()
