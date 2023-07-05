# Importing necessary libraries
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from torch.optim.lr_scheduler import StepLR #, LambdaLR, MultiStepLR, ExponentialLR

from actor_critic import ActorCritic
from gae import GAE

from data_logger import DataLogger

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
        data_logger: Any,
        lr_step_size: int,
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

        # Define the optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Create the learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=0.5)

        # Create an instance of GAE
        self.gae = GAE(gamma=0.99, lambda_=0.95)

        self.gamma = gamma
        self.seed = seed
        self.state_dim = state_dim
        self.state_channel = state_channel
        self.action_dim = action_dim
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.data_logger = data_logger
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
        step: int,
        use_gae: Optional[bool] = False, #True,
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
        assert (
            next_action_distribution is not None
        ), "Next action distribution cannot be None"

        # Assert that indices is not None
        assert indices is not None, "Indices cannot be None"

        # Assert that weight is not None
        assert weight is not None, "Weight cannot be None"

        # Unsqueeze the following tensor to have the correct [batch_size, num_actions], e.g. [64, 3]
        reward = reward.unsqueeze(1)
        terminated = terminated.unsqueeze(1)

        # Estimate value function V(s) for the current state
        state_value = self.actor_critic.critic(state)

        # Estimate value function V(s') for the next state
        next_state_value = self.actor_critic.critic(next_state)

        # Calculate Q-value estimates for the current and next state-action pairs
        q_value = self.actor_critic.critic.evaluate(state, action)

        # Calculate Q-value estimates for next state-action pairs
        next_q_value = self.actor_critic.critic.evaluate(next_state, next_action)

        # The GAE combines the immediate advantate (one-step TD error) and the estimated future advantages
        # using the GAE parameter (lambda) and the discount factor (gamma).
        # The eligibility trace is used to assign credit (or blame) for the TD error occurring on a given step
        # to the previous steps.
        # Eligibility traces are the primary mechanisms of temporal credit assignment in TD learning
        # The GAE trace is calculated based on the one-step TD error using the following formula:
        #   GAE(s, a) = ∑[(γ * λ)^t * δt]
        # Where:
        #   γ is the discount factor, representing the agent's preference for immediate rewards over future rewards.
        #   λ is the GAE parameter, controlling the trade-off between bias and variance in the advantage estimation.
        #   t is the time step in the trajectory.
        #   δt is the one-step TD error, calculated as δt = r(t) + γ * V(s(t+1)) - V(s(t)).
        # The GAE trace is a sum of weighted one-step TD errors across the trajectory.
        # It combines the immediate advantage (one-step TD error) with the estimated future advantages
        # (through powers of γ * λ) to form a more robust estimate of the advantage function.
        if use_gae:
            # Calculate the advantages using GAE with eligibility trace
            advantage = self.gae.calculate_gae_eligibility_trace(
                reward,
                q_value,
                next_q_value,
                terminated,
                normalize=True
            )

            # The target-Q value can be calculated as follows:
            #   target_Q(t) = GAE(t) + V(s(t))
            # Where:
            #   target_Q(t) is the target-Q value for the state-action pair at time step t.
            #   GAE(t) is the GAE trace for the state-action pair at time step t.
            #   V(s(t)) is the estimated value function (state value) for the state s(t).
            # By using the GAE trace in combination with the estimated value function,
            # the target-Q value incorporates both the immediate advantage and the estimated future rewards,
            # leading to more accurate and stable updates for the critic network.
            target_q_value = advantage + state_value
        else:
            # Calculate the standard Temporal Differencce (TD) learning, TD(0),
            # target Q-value is calculated based on the next state-action pair, using the standard TD target.
            target_q_value = reward + self.gamma * (1.0 - terminated) * next_state_value

            # Calculate advantage: A(state, action) = Q(state, action) - V(state), as means for variance reduction.
            # Q(state, action) can be obtained by calling the evaluate method with the given state-action pair as input,
            # and V(state) can be obtained by calling the forward method with just the state as input.
            # Assuming next_state_value and state_value are 1-D tensors of shape [64]
            advantage = target_q_value - state_value

        # TD_error = |target Q(s', a') - Q(s, a)|,
        # where taget Q(s', a') = r + γ * V(s'), used in PER.
        td_error = abs(target_q_value - q_value)

        # Calculate critic loss, weighted by importance sampling factor
        critic_loss = torch.mean(weight * F.smooth_l1_loss(target_q_value, q_value))

        # Assuming weight is of shape [64]
        # Reshape weight to [64, 1] to make it broadcastable to action_log_prob
        weight = weight.view(-1, 1)

        # Now, repeat the weight values along dimension 1 to match the number of actions (3)
        # This will make weight have shape [64, 3], with the same weight applied to each action in the batch
        num_actions = action.shape[1]
        weight = weight.repeat(1, num_actions)

        # Compute entropy
        entropy = next_action_distribution.entropy().mean()

        # Calculate actor log-probability
        action_log_prob = action_distribution.log_prob(action)

        # Make sure the shape of weight matches the shape of action_log_prob
        assert weight.shape == action_log_prob.shape, "Weight and action_log_prob shape mismatch."

        # Compute the actor loss, taking into account the importance sampling factor for weighting
        actor_loss = -torch.mean(weight * action_log_prob * advantage)

        # Calculate total loss
        loss = self.value_coef * critic_loss + actor_loss - self.entropy_coef * entropy

        # Update episode total loss, actor loss, and crtic loss
        self.data_logger.update_episode_cumulative_total_loss(loss.item())
        self.data_logger.update_episode_cumulative_actor_loss(actor_loss.item())
        self.data_logger.update_episode_cumulative_critic_loss(critic_loss.item())
        self.data_logger.update_episode_cumulative_entropy(entropy.item())

        # Zero out gradients
        self.optimizer.zero_grad()

        # Calculate backpropagation
        loss.backward()

        # Apply gradient clipping separately for actor and critic networks
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.actor.parameters(), max_norm=0.5, norm_type=2
        )
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.critic.parameters(), max_norm=0.5, norm_type=2
        )

        # Apply update rule to neural network weights
        self.optimizer.step()

        # Step the learning rate scheduler
        self.scheduler.step()

        # Log learning rate
        self.data_logger.log_learning_rate(self.optimizer)

        # Increment the step attribute
        self.data_logger.increment_step()

        return indices, td_error


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

        # Create a tensor for indices with the same dimensions and structure as the reward tensor
        indices = (
            torch.ones_like(
                torch.tensor(reward_ndarray, dtype=torch.float32), dtype=torch.long
            )
            .unsqueeze(0)
            .to(device)
        )

        # Create a tensor for weight with the same dimensions and structure as the reward tensor
        weight = (
            torch.ones_like(
                torch.tensor(reward_ndarray, dtype=torch.float32), dtype=torch.float32
            )
            .unsqueeze(0)
            .to(device)
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
            state=state,
            action=clipped_action,
            reward=reward,
            next_state=next_state,
            next_action=clipped_next_action,
            terminated=terminated,
            action_distribution=action_distribution,
            next_action_distribution=next_action_distribution,
            indices=indices,
            weight=weight
        )

        # Update total reward
        total_reward += float(reward.item())
        print(f"Total reward: {total_reward:.2f}")

        state = next_state

        # Update if the environment is done
        done = terminated or truncated
        if done:
            break
