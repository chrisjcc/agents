# Importing necessary libraries
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from actor_critic import ActorCritic
from gae import GAE

# Setting the seed for reproducibility
torch.manual_seed(0)


# Define the ActorCritic Agent
class ActorCriticAgent:
    """
    The ActorCriticAgent class defines an actor-critic reinforcement learning agent.
    """

    def __init__(
        self,
        state_dim: int,
        state_channel: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int,
        device: Any,
        lr: float = 0.01,
        gamma: float = 0.99,
        seed: int = 42,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        """
        Initializes the ActorCriticAgent.
        :param state_dim: The number of dimensions in the state space.
        :param state_channel: The number of dimension in the state channel (e.g. RGB).
        :param action_dim: The number of dimensions in the action space.
        :param hidden_dim: The number of hidden units in the neural networks for actor and critic.
        :param lr: The learning rate for the optimizer.
        :param gamma: The discount factor for future rewards.
        :param seed: The random seed for reproducibility.
        :param value_coef: The magnitude of the critic loss.
        :param entropy_coef: The magnitude of the entropy regularization.
        """
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gae = GAE(gamma=0.99, tau=0.95)

        self.gamma = gamma
        self.seed = seed
        self.state_dim = state_dim
        self.state_channel = state_channel
        self.action_dim = action_dim
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

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
        next_action: torch.Tensor,
        terminated: torch.Tensor,
        action_distribution: Any,
        next_action_distribution: Any,
        use_gae: Optional[bool] = True,
    ) -> None:
        """
        Updates the ActorCriticAgent.
        :param state: The current state of the environment.
        :param action: The action taken within the environment.
        :param reward: The reward obtained for taking the action in the current state.
        :param next_state: The next state visited by taking the action in the current state.
        :param next_action: The next action taken within the environment.
        :param done: A boolean indicating whether the episode has terminated.
        """
        # Evaluate Q-value of random state-action pair
        q_value = self.actor_critic.evaluate(state, action)

        # Evaluate Q-value of next state-action pair
        next_q_value = self.actor_critic.evaluate(next_state, next_action)

        # Calculate critic loss
        ones = torch.ones_like(
            terminated
        )  # create a tensor of 1's with the same size as terminated

        # TODO: improve this step (why is it necessary?)
        q_value = torch.squeeze(q_value, dim=1)
        next_q_value = torch.squeeze(next_q_value, dim=1)

        # Discounted rewards
        if use_gae:
            target_q_value = self.gae.calculate_gae_eligibility_trace(
                reward, q_value, next_q_value, terminated, normalize=True
            )

        else:
            target_q_value = reward + self.gamma * (ones - terminated) * next_q_value

        critic_loss = F.smooth_l1_loss(target_q_value, q_value)

        # Calculate advantage (in this case specifically temporal-difference)
        advantage = target_q_value - q_value

        # Compute entropy
        entropy = torch.mean(next_action_distribution.entropy())

        # Calculate actor loss
        action_log_prob = action_distribution.log_prob(action)

        # TODO: improve this step (is it necessary??)
        action_log_prob = action_log_prob.reshape(-1, action_log_prob.size(0))

        actor_loss = -torch.mean(action_log_prob * advantage)

        # Calculate total loss
        loss = self.value_coef * critic_loss + actor_loss - self.entropy_coef * entropy

        # Zero out gradients
        self.optimizer.zero_grad()

        # Calculate backprogation
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), max_norm=0.5, norm_type=2
        )

        # Apply update rule to neural network weights
        self.optimizer.step()


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting the seed for reproducibility
    torch.manual_seed(0)

    # Name the environment to be sued
    env_name: str = "CarRacing-v2"

    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
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
    max_action = float(action_high[0])

    # Convert from numpy to tensor
    low = torch.from_numpy(action_space.low).to(device)
    high = torch.from_numpy(action_space.high).to(device)

    # Actor-Critic hyperparameters
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01
    hidden_dim = 256

    # Initialize Actor-Critic network
    agent = ActorCriticAgent(
        state_dim=state_dim,
        state_channel=state_channel,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=hidden_dim,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
    )
    agent.set_seed()

    # Get state spaces
    state, info = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device).permute(0, 3, 1, 2)

    # This loop constitutes one epoch
    total_reward = 0.0
    while True:
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
            # Obtain mean and std action given state
            action_tensor, action_distribution = agent.actor_critic.sample_action(
                state_tensor
            )

            # Rescale, then clip the action to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(
                (((action_tensor + 1) / 2) * (high - low) + low), low, high
            )

        # Convert from numpy to torch tensors, and send to device
        action = clipped_action.squeeze().cpu().detach().numpy()

        # Take one step in the environment given the agent action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Convert to tensor
        next_state_tensor = (
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        ).permute(0, 3, 1, 2)
        reward_tensor = (
            torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)
        )
        terminated_tensor = (
            torch.tensor(terminated, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # Obtain mean and std of next action given next state
        next_action_tensor, next_action_distribution = agent.actor_critic.sample_action(
            next_state_tensor
        )

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_next_action = torch.clamp(
            (((next_action_tensor + 1) / 2) * (high - low) + low), low, high
        )

        agent.update(
            state_tensor,
            clipped_action,
            reward_tensor,
            next_state_tensor,
            clipped_next_action,
            terminated_tensor,
            action_distribution,
            next_action_distribution,
        )

        state_tensor = next_state_tensor

        total_reward += float(reward)
        print(f"Total reward: {total_reward:.2f}")

        # Update if the environment is done
        done = terminated or truncated
        if done:
            break
