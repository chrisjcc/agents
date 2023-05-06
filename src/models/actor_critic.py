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
torch.manual_seed(0)


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
        device: Any,
        hidden_dim: int = 256,
    ) -> None:
        super(ActorCritic, self).__init__()
        # Initialize Actor policy
        self.actor = Actor(state_dim, action_dim, max_action).to(device)

        # Initialize Critic
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim).to(device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal, torch.Tensor]:
        """
        Return predictions from the  Actor and Critic networks, given a state tensor.
        :param state: A pytorch tensor representing the current state.
        :return: Pytorch Tensor representing the Actor network predictions and the Critic network predictions.
        """

        action, action_distribution = self.sample_action(state)
        state_value = self.critic(state, action)

        return action, action_distribution, state_value

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """
        Performs a forward pass using the actor network.
        :param state: The current state of the agent.
        :return: A tuple containing the selected action, its distribution and its estimated value.
        """
        # Sample action from actor network
        # with torch.no_grad():
        # Choose action using actor network
        action_mean, action_std = self.actor(state)

        # Sample an action from the distribution
        action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore
        action = action_distribution.sample()  # type: ignore

        return action, action_distribution

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Any:
        """
        Perform a forward pass using critic network to calculate Q-value(s,a).
        :param state: The current state of the agent.
        :param action: The action take by the agent.
        :return: A Q-value tuple evaluating the state-action value
        """
        q_value = self.critic(state, action)
        return q_value


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Name the environment to use
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
    state_dim = env.observation_space.shape[0]

    if state_dim is None:
        raise ValueError("Observation space shape is None.")

    # Get action spaces
    action_space = env.action_space
    max_action = float(action_space.high[0])
    action_dim = action_space.shape[0]

    low = torch.from_numpy(action_space.low).to(device)
    high = torch.from_numpy(action_space.high).to(device)

    # Actor-Critic hyperparameters
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01

    # Initialize Actor-Critic network
    actor_critic = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
    )

    # Initialize optimizer
    actor_critic_optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    # Get state spaces
    state, info = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # This loop constitutes one epoch
    total_reward = 0.0
    while True:
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
            # Obtain mean and std action given state
            action_tensor, action_distribution = actor_critic.sample_action(
                state_tensor
            )

            # Rescale, then clip the action to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(
                (((action_tensor + 1) / 2) * (high - low) + low), low, high
            )

        # Evaluate Q-value of state-action pair
        q_value = actor_critic.evaluate(state_tensor, clipped_action)
        print(f"Q-value(state,action): {q_value.item():.3f}")

        # Convert from numpy to torch tensors, and send to device
        action = clipped_action.squeeze().cpu().detach().numpy()

        # Take one step in the environment given the agent action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Convert to tensor
        next_state_tensor = (
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        )
        reward_tensor = (
            torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # Obtain mean and std of next action given next state
        next_action_tensor, next_action_distribution = actor_critic.sample_action(
            next_state_tensor
        )

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_next_action = torch.clamp(
            (((next_action_tensor + 1) / 2) * (high - low) + low), low, high
        )

        # Evaluate Q-value of next state-action pair
        next_q_value = actor_critic.evaluate(next_state_tensor, clipped_next_action)
        print(f"Next Q-value(next_state,next_action): {q_value.item():.3f}")

        # Calculate target Q-value
        target_q_value = reward_tensor + gamma * (1 - terminated) * next_q_value
        critic_loss = F.smooth_l1_loss(target_q_value, q_value)

        # Calculate advantage
        advantage = target_q_value - q_value

        # Compute entropy
        entropy = torch.mean(next_action_distribution.entropy())  # type: ignore

        # Calculate actor loss
        action_log_prob = next_action_distribution.log_prob(clipped_next_action)  # type: ignore
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

        state_tensor = next_state_tensor

        total_reward += float(reward)
        print(f"Q-value: {q_value.item():.2f}, Total reward: {total_reward:.2f}")

        # Update if the environment is done
        done = terminated or truncated
        if done:
            break
