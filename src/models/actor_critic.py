# Importing necessary libraries
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_networks.actor_network import Actor
from neural_networks.critic_network import Critic
from torch.distributions import Categorical, Normal

# Setting the seed for reproducibility
torch.manual_seed(42)


# Define the ActorCritic architecture using the Actor and Critic network
class ActorCriticModel(nn.Module):
    """
    The ActorCritic class defines the complete actor-critic architecture.
    It consists of an Actor and a Critic neural network.
    """

    def __init__(
        self,
        state_dim: int,
        state_channel: int,
        action_dim: int,
        max_action: float,
        device: Any,
        hidden_dim: int = 256,
    ) -> None:
        super(ActorCriticModel, self).__init__()

        # Initialize Actor policy
        self.actor = Actor(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
            max_action=max_action,
        ).to(device)

        # Initialize Critic
        self.critic = Critic(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
        ).to(device)

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """
        Performs a forward pass using the actor network.
        :param state: The current state of the agent.
        :return: A tuple containing the selected action, its distribution and its estimated value.
        """
        # Sample action from actor network
        #with torch.no_grad():
        # Sample an action from the actor network distribution
        action, action_distribution = self.actor.sample_action(state)

        return action, action_distribution

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass using the critic network
        """
        #with torch.no_grad():
        q_value = self.critic.evaluate(state, action)

        return q_value


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Name the environment to use
    env_name: str = "CarRacing-v2"
    max_episode_steps = 600  # default

    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
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
    state_dim = int(env.observation_space.shape[0])
    state_channel = int(env.observation_space.shape[2])

    if state_dim is None:
        raise ValueError("Observation space shape is None.")

    # Get action spaces
    action_space = env.action_space
    action_dim = int(action_space.shape[0])
    max_action = float(action_space.high[0])

    low = torch.from_numpy(action_space.low).to(device)
    high = torch.from_numpy(action_space.high).to(device)

    # Actor-Critic hyperparameters
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01

    # Initialize Actor-Critic network
    actor_critic = ActorCriticModel(
        state_dim=state_dim,
        state_channel=state_channel,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
    )

    # Initialize optimizer
    actor_critic_optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    # Get state spaces
    state_ndarray, info = env.reset(seed=42)

    # Convert state to shape (batch_size, channel, wdith, hight)
    state = (
        torch.tensor(state_ndarray, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
        .permute(0, 3, 1, 2)
    )

    # This loop constitutes one epoch
    total_reward = 0.0
    step_count = 0
    done = False
    while not done:
        print(f"Step: {step_count}")

        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
            # Obtain mean and std action given state
            action_tensor, action_distribution = actor_critic.sample_action(state)

            # Rescale, then clip the action to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(
                (((action_tensor + 1) / 2) * (high - low) + low), low, high
            )

        # Evaluate Q-value of state-action pair
        q_value = actor_critic.evaluate(state, clipped_action)

        # Take one step in the environment given the agent action
        next_state_ndarray, reward_ndarray, terminated, truncated, info = env.step(
            clipped_action.squeeze().cpu().detach().numpy()
        )

        # Convert to tensor
        next_state = (
            torch.tensor(next_state_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
            .permute(0, 3, 1, 2)
        )

        reward = (
            torch.tensor(reward_ndarray, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = actor_critic.sample_action(next_state)

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_next_action = torch.clamp(
            (((next_action + 1) / 2) * (high - low) + low), low, high
        )

        # Evaluate Q-value of next state-action pair
        next_q_value = actor_critic.evaluate(next_state, clipped_next_action)

        # Calculate target Q-value
        target_q_value = reward + gamma * (1 - terminated) * next_q_value
        critic_loss = F.smooth_l1_loss(target_q_value, q_value)

        # Calculate advantage
        advantage = target_q_value - q_value

        # Compute entropy
        entropy = torch.mean(next_action_distribution.entropy())  # type: ignore

        # Calculate actor loss
        action_log_prob = action_distribution.log_prob(clipped_action)  # type: ignore
        actor_loss = -torch.mean(action_log_prob * advantage)

        # Calculate total loss
        loss = value_coef * critic_loss + actor_loss - entropy_coef * entropy

        # Zero out gradients
        actor_critic_optimizer.zero_grad()

        # Calculate backprogation
        loss.backward()  # type: ignore

        # Apply gradient norm clipping
        torch.nn.utils.clip_grad_norm_(
            actor_critic.parameters(), max_norm=0.5, norm_type=2
        )

        # Apply update rule to neural network weights
        actor_critic_optimizer.step()

        total_reward += float(reward)
        step_count += 1

        # Print Reward and Q-values
        print(f"\tTotal reward: {total_reward:.2f}")
        print(f"\tQ-value(state,action): {q_value.item():.3f}")
        print(f"\tNext Q-value(next_state,next_action): {next_q_value.item():.3f}")

        state = next_state

        # Update if the environment is done
        done = terminated or truncated
