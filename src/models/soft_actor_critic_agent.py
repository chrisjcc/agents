# soft_actor_critic_agent.py
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from neural_networks.actor_network import Actor
from neural_networks.critic_network import Critic
from replay_buffer.per import PrioritizedReplayBuffer

# Setting the seed for reproducibility
torch.manual_seed(0)


class ActorCritic(nn.Module):
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
        hidden_dim: int = 256,
    ) -> None:
        super(ActorCritic, self).__init__()

        # Initialize Actor policy
        self.actor = Actor(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
            max_action=max_action,
        )

        # Initialize Critic networks
        self.critic_1 = Critic(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
        )
        self.critic_2 = Critic(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return predictions from the Actor, Critic1, Critic2 networks, given a state tensor.
        :param state: A PyTorch tensor representing the current state.
        :return: Tuples representing the predicted action, action distribution, critic1 value, and critic2 value.
        """
        action, action_distribution = self.sample_action(state)
        critic_1_value = self.critic_1.evaluate(state, action)
        critic_2_value = self.critic_2.evaluate(state, action)

        return action, action_distribution, critic_1_value, critic_2_value

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """
        Performs a forward pass using the actor network to sample an action.
        :param state: The current state of the agent.
        :return: A tuple containing the sampled action and its distribution.
        """
        # Sample action from actor network
        action_mean, action_std = self.actor(state)
        action_distribution = Normal(loc=action_mean, scale=action_std)
        action = action_distribution.sample()

        return action, action_distribution


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        state_channel: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int,
        device: torch.device,
        lr: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        replay_buffer_capacity: int = 1024,
        replay_buffer_alpha: float = 0.99,
    ) -> None:
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Initialize the actor-critic networks
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=hidden_dim,
        ).to(device)

        # Initialize target networks
        self.target_actor_critic = ActorCritic(
            state_dim=state_dim,
            state_channel=state_channel,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=hidden_dim,
        ).to(device)

        self.target_actor_critic.load_state_dict(self.actor_critic.state_dict())

        # Set up the optimizers
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(
            self.actor_critic.critic_1.parameters(), lr=lr
        )
        self.critic_2_optimizer = optim.Adam(
            self.actor_critic.critic_2.parameters(), lr=lr
        )

        # Initialize the replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_buffer_capacity, alpha=replay_buffer_alpha
        )

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminated: torch.Tensor,
        next_action: torch.Tensor,
        action_distribution: Any,
        next_action_distribution: Any,
        indices: torch.Tensor,
        weight: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the weights of the SAC model.

        :param state (torch.Tensor): The current state tensor.
        :param action (torch.Tensor): The action tensor.
        :param reward (torch.Tensor): The reward tensor.
        :param next_state (torch.Tensor): The next state tensor.
        :param terminated (torch.Tensor): The termination tensor.
        :param next_action (torch.Tensor): The next action tensor.
        :param action_distribution (Any): The action distribution.
        :param next_action_distribution (Any): The next action distribution.
        :param indices (torch.Tensor): The indices tensor.
        :param weight (torch.Tensor): The weight tensor.
        :param step (int): The current step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the indices and TD errors.
        """
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        terminated = terminated.to(self.device)

        indices = indices.to(self.device)
        weight = weight.to(self.device)

        # Unsqueeze the following tensor to have the correct [batch_size, num_actions], e.g. [64, 3]
        reward = reward.unsqueeze(1)
        terminated = terminated.unsqueeze(1)

        # Compute the target Q-values using the target networks
        with torch.no_grad():
            next_action, next_action_distribution, _, _ = self.target_actor_critic(
                next_state
            )
            target_q_value_1 = self.target_actor_critic.critic_1.evaluate(
                next_state, next_action
            )
            target_q_value_2 = self.target_actor_critic.critic_2.evaluate(
                next_state, next_action
            )
            target_q_value = torch.min(target_q_value_1, target_q_value_2)
            target_q_value = reward + (1 - terminated) * self.gamma * (
                target_q_value
                - self.alpha * next_action_distribution.log_prob(next_action)
            )

        # Update the critics
        q_value_1 = self.actor_critic.critic_1.evaluate(state, action)
        q_value_2 = self.actor_critic.critic_2.evaluate(state, action)

        critic_1_loss = F.mse_loss(q_value_1, target_q_value, reduction="none")
        critic_2_loss = F.mse_loss(q_value_2, target_q_value, reduction="none")

        # Update the priorities in the replay buffer
        td_errors = torch.max(critic_1_loss, critic_2_loss).mean(
            dim=1, keepdim=True
        )

        critic_1_loss = (weight.unsqueeze(1) * critic_1_loss).mean()
        critic_2_loss = (weight.unsqueeze(1) * critic_2_loss).mean()

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update the actor
        action, action_distribution, log_prob, _ = self.actor_critic(state)
        q_value = torch.min(
            self.actor_critic.critic_1.evaluate(state, action),
            self.actor_critic.critic_2.evaluate(state, action),
        )
        actor_loss = (self.alpha * log_prob - q_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        for target_param, param in zip(
            self.target_actor_critic.parameters(), self.actor_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return indices, td_errors

    def state_dict(self):
        info = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict()
        }

        return info
