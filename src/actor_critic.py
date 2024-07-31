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
        state_dim: Any,
        action_dim: int,
        device: Any,
        hidden_dim: int = 256,
    ) -> None:
        super(ActorCriticModel, self).__init__()

        # Initialize Actor policy
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)

        # Initialize Critic
        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """
        Performs a forward pass using the actor network.
        :param state: The current state of the agent.
        :return: A tuple containing the selected action, its distribution and its estimated value.
        """
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
            action_logits = self.actor(state)

            # Create a categorical distribution over the action values
            action_distribution = Categorical(
                logits=action_logits
            )

            # Sample an action from the distribution
            action = action_distribution.sample()

        return action, action_distribution

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass using the critic network
        """
        q_value = self.critic(state, action)

        return q_value


if __name__ == "__main__":
    """Highway-v0 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Name the environment to use
    env_name: str = "highway-fast-v0"
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

    # We first check if state_shape is None. If it is None, we raise a ValueError.
    # Otherwise, we access the first element of state_shape using its index and
    # using the int() function.
    state_dim = (15, 5) #int(env.observation_space.shape[0])

    if state_dim is None:
        raise ValueError("Observation space shape is None.")

    # Get action spaces
    action_space = env.action_space
    action_dim = 1

    # Actor-Critic hyperparameters
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01

    # Initialize Actor-Critic network
    actor_critic = ActorCriticModel(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    # Initialize optimizer
    actor_critic_optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    # Get state spaces
    state_ndarray, info = env.reset(seed=42)

    # Convert state to shape (batch_size, channel, wdith, hight)
    state = torch.tensor(state_ndarray, dtype=torch.float32).to(device)
    state = state.flatten()
    batch_size = 1
    state = state.unsqueeze(0).expand(batch_size, -1)

    # This loop constitutes one epoch
    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        print(f"Step: {step_count}")

        # Obtain mean and std action given state
        action, action_distribution = actor_critic.sample_action(state)

        # Evaluate Q-value of state-action pair
        action = action.unsqueeze(1)  # Now action has shape [1, 1]
        q_value = actor_critic.evaluate(state, action)

        # Take one step in the environment given the agent action
        next_state_ndarray, reward_ndarray, terminated, truncated, info = env.step(
            action.squeeze().cpu().detach().numpy()
        )

        # Convert to tensor
        next_state = torch.tensor(next_state_ndarray, dtype=torch.float32).to(device)
        next_state = next_state.flatten()
        next_state = next_state.unsqueeze(0).expand(batch_size, -1)

        reward = torch.tensor([reward_ndarray],
            dtype=torch.float32
        ).to(device)

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = actor_critic.sample_action(next_state)

        # Evaluate Q-value of next state-action pair
        next_action = next_action.unsqueeze(1)  # Now action has shape [1, 1]
        next_q_value = actor_critic.evaluate(next_state, next_action)

        # Calculate target Q-value
        target_q_value = reward + gamma * (1 - terminated) * next_q_value
        critic_loss = F.smooth_l1_loss(target_q_value, q_value)

        # Calculate advantage
        advantage = target_q_value - q_value

        # Compute entropy
        entropy = torch.mean(action_distribution.entropy())  # type: ignore

        # Calculate actor loss
        action_log_prob = action_distribution.log_prob(action)  # type: ignore
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
