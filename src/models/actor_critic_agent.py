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
        self.set_seed()

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
        indices: torch.Tensor,
        weight: torch.Tensor,
        use_gae: Optional[bool] = True,
    ) -> None:
        """
        Updates the ActorCriticAgent.
        :param state: The current state of the environment.
        :param action: The action taken within the environment.
        :param reward: The reward obtained for taking the action in the current state.
        :param next_state: The next state visited by taking the action in the current state.
        :param next_action: The next action taken within the environment.
        :param terminated: A boolean indicating whether the episode has terminated.
        """

        # Assert that state is not None
        assert state is not None, "State cannot be None"

        # Assert that action is not None
        assert action is not None, "Action cannot be None"

        # Assert that reward is not None
        assert reward is not None, "Reward cannot be None"

        # Assert that next_state is not None
        assert next_state is not None, "Next state cannot be None"

        # Assert that next_action is not None
        assert next_action is not None, "Next action cannot be None"

        # Assert that terminated is not None
        assert terminated is not None, "Terminated cannot be None"

        # Assert that action_distribution is not None
        assert action_distribution is not None, "Action distribution cannot be None"

        # Assert that next_action_distribution is not None
        assert next_action_distribution is not None, "Next action distribution cannot be None"

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


        # Check indices validity
        if len(q_value) > 1 and len(next_q_value) > 1:
            # Apply indices and weights
            # To ensures that the indices are wrapped within the valid range
            q_value = q_value[indices % len(q_value)]
            next_q_value = next_q_value[indices % len(next_q_value)]
            weight = weight[indices % len(weight)]


        # Discounted rewards
        if use_gae:
            target_q_value = self.gae.calculate_gae_eligibility_trace(
                reward, q_value, next_q_value, terminated, normalize=True
            )

        else:
            target_q_value = reward + self.gamma * (ones - terminated) * next_q_value

        #critic_loss = F.smooth_l1_loss(target_q_value, q_value)
        critic_loss = torch.mean(weight * F.smooth_l1_loss(target_q_value, q_value))

        # Calculate advantage (in this case specifically temporal-difference)
        advantage = target_q_value - q_value

        # Compute entropy
        entropy = torch.mean(next_action_distribution.entropy())

        # Calculate actor loss
        action_log_prob = action_distribution.log_prob(action)

        # Check indices validity
        if len(action_log_prob) > 1:
            # To ensures that the indices are wrapped within the valid range
            action_log_prob = action_log_prob[indices % len(action_log_prob)]

        #action_log_prob = action_log_prob.reshape(-1, action_log_prob.size(0))
        #action_log_prob = action_log_prob[indices]

        # TODO: improve this step (is it necessary??)
        action_log_prob = action_log_prob.reshape(-1, action_log_prob.size(0))

        #actor_loss = -torch.mean(action_log_prob * advantage)
        actor_loss = -torch.mean(weight * action_log_prob * advantage)

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

    # Name the environment to be used
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env_name: str = "CarRacing-v2"
    max_episode_steps = 600  # default

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

    # Convert from numpy to tensor
    low = torch.from_numpy(action_space.low).to(device)
    high = torch.from_numpy(action_space.high).to(device)

    # Actor-Critic hyperparameters
    gamma = 0.99
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
        gamma=gamma,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
    )

    # Get state spaces
    state_ndarray, info = env.reset()

    # Convert next state to shape (batch_size, channe, width, height)
    state = (
        torch.tensor(state_ndarray, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
        .permute(0, 3, 1, 2)
    )

    # Set variables
    total_reward = 0.0
    done = False

    # This loop constitutes one epoch
    while not done:
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        # with torch.no_grad():
        # Pass the state through the Actor model to obtain a probability distribution over the actions
        action, action_distribution = agent.actor_critic.sample_action(state)

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_action = torch.clamp(
            (((action + 1) / 2) * (high - low) + low), low, high
        )

        # Take the action in the environment and observe the next state, reward, and done flag
        (
            next_state_ndarray,
            reward_ndarray,
            terminated_ndarray,
            truncated_ndarray,
            info,
        ) = env.step(clipped_action.squeeze().cpu().detach().numpy())

        # Convert next state to shape (batch_size, channe, width, height)
        next_state = (
            torch.tensor(next_state_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
            .permute(0, 3, 1, 2)
        )

        reward = (
            torch.tensor(reward_ndarray, dtype=torch.float32).unsqueeze(0).to(device)
        )
        terminated = (
            torch.tensor(terminated_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        truncated = (
            torch.tensor(truncated_ndarray, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = agent.actor_critic.sample_action(
            next_state
        )

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_next_action = torch.clamp(
            (((next_action + 1) / 2) * (high - low) + low), low, high
        )

        assert isinstance(state, torch.Tensor), "State is not of type torch.Tensor"
        assert isinstance(
            clipped_action, torch.Tensor
        ), "Clipped action is not of type torch.Tensor"
        assert isinstance(reward, torch.Tensor), "Reward is not of type torch.Tensor"
        assert isinstance(
            next_state, torch.Tensor
        ), "Next state is not of type torch.Tensor"
        assert isinstance(
            terminated, torch.Tensor
        ), "Terminated is not of type torch.Tensor"
        assert isinstance(
            truncated, torch.Tensor
        ), "Truncated is not of type torch.Tensor"

        agent.update(
            state,
            clipped_action,
            reward,
            next_state,
            clipped_next_action,
            terminated,
            action_distribution,
            next_action_distribution,
        )

        # Update total reward
        total_reward += float(reward.item())
        print(f"Total reward: {total_reward:.2f}")

        state = next_state

        # Update if the environment is done
        done = terminated or truncated
        if done:
            break
