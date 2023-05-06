import torch
from typing import Any


class GAE:
    def __init__(self, gamma: float = 0.99, tau: float = 0.95) -> None:
        self.gamma = gamma
        self.tau = tau

    def calculate_gae_eligibility_trace(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
        masks: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Calculates the Generalized Advantage Estimate (GAE) using an eligibility trace.

        Args:
            rewards (torch.Tensor): Tensor of shape torch.Size([64]) containing the rewards for each timestep.
            values (torch.Tensor): Tensor of shape torch.Size([64]) containing the predicted values for each timestep.
            next_value (torch.Tensor): Tensor of shape torch.Size([64]) containing the predicted value for the next timestep.
            masks (torch.Tensor): Tensor of shape torch.Size([64]) containing the masks for each timestep.
            normalize (bool): Whether to normalize the GAE.

        Returns:
            gae (torch.Tensor): Tensor of shape torch.Size([64]) containing the GAE for each timestep.
        """
        gae = torch.zeros_like(rewards)
        advantage = torch.zeros_like(rewards)

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value[i] * masks[i] - values[i]
            advantage = delta + self.gamma * self.tau * advantage * masks[i]
            gae[i] = advantage[i]
            next_value[i] = values[i]

        if normalize:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        return gae




    def generalized_advantage_estimate(
        self,
        value_old_state: torch.Tensor,
        value_new_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
        lamda: float = 0.95,
        normalize: bool = True,
    ) -> Any:
        """
        Get generalized advantage estimate of a trajectory
        gamma: trajectory discount (scalar)
        lamda: exponential mean discount (scalar)
        value_old_state: value function result with old_state input
        value_new_state: value function result with new_state input
        reward: agent reward of taking actions in the environment
        done: flag for end of episode
        """

        batch_size = done.shape[0]

        advantage = torch.zeros(batch_size + 1)

        for t in reversed(range(batch_size)):
            delta = reward[t] + (gamma * value_new_state[t] * done[t]) - value_old_state[t]
            advantage[t] = delta + (gamma * lamda * advantage[t + 1] * done[t])

        #value_target = advantage[:batch_size] + value_old_state.squeeze()

        if normalize:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        return advantage[:batch_size] #, value_target


    def calculate_gae(self, rewards, values, next_value, masks, gamma=0.99, tau=0.95, normalize=True):
        """Calculate the Online Generalized Advantage Estimator (GAE).

        Args:
            rewards (Tensor): Rewards of shape (batch_size, num_steps).
            values (Tensor): Estimated values of shape (batch_size, num_steps + 1).
            next_value (Tensor): Estimated value of the last state of shape (batch_size,).
            masks (Tensor): Masks of shape (batch_size, num_steps).
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): GAE parameter. Defaults to 0.95.

        Returns:
            Tensor: GAE of shape (batch_size, num_steps).
        """
        batch_size = rewards.size()[0]
        gae = torch.zeros_like(rewards)
        advantage = torch.zeros_like(rewards)
        next_advantage = next_value

        for t in reversed(range(batch_size)):
            delta = rewards[t] + gamma * next_advantage[t] * masks[t] - values[t]
            advantage = delta + gamma * tau * masks[t] * advantage
            gae[t] = advantage[t]
            next_advantage = advantage + values[t]

        if normalize:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        return gae

    def gae_eligibility_trace(self, rewards, values, next_value, masks, gamma=0.99, tau=0.95, normalize=True,):
        """
        Calculates the Online Generalized Advantage Estimation (GAE) using eligibility traces.

        Args:
            rewards (torch.Tensor): Tensor of shape (batch_size, num_steps) containing the rewards.
            values (torch.Tensor): Tensor of shape (batch_size, num_steps) containing the estimated values.
            next_value (torch.Tensor): Tensor of shape (batch_size,) containing the estimated value of the next state.
            masks (torch.Tensor): Tensor of shape (batch_size, num_steps) containing the masks (0 if the episode ended, 1 otherwise).
            gamma (float): Discount factor.
            tau (float): GAE parameter.

        Returns:
            advantages (torch.Tensor): Tensor of shape (batch_size, num_steps) containing the advantages.
        """
        batch_size = rewards.size()[0]

        # Compute TD errors
        deltas = rewards + gamma * masks * next_value - values

        # Initialize eligibility trace
        eligibility_trace = torch.zeros(batch_size)
        eligibility_trace = torch.zeros(batch_size)

        # Initialize advantages
        gae = torch.zeros(batch_size)

        # Compute advantages using eligibility traces
        for t in reversed(range(batch_size)):
            eligibility_trace[t] = gamma * tau * eligibility_trace[t] + masks[t]

            gae[t] = eligibility_trace[t] * deltas[t] + gamma * tau * (1 - masks[t]) * gae[t]

        if normalize:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        return gae

    def calculate_gae_eligibility_trace(self, rewards, values, next_value, masks, gamma=0.99, tau=0.95, normalize=True,):
        """
        Calculates the Generalized Advantage Estimate (GAE) using an eligibility trace.

        Args:
            rewards (torch.Tensor): Tensor of shape torch.Size([64]) containing the rewards for each timestep.
            values (torch.Tensor): Tensor of shape torch.Size([64]) containing the predicted values for each timestep.
            next_value (float): The predicted value for the next timestep.
            masks (torch.Tensor): Tensor of shape torch.Size([64]) containing the masks for each timestep.
            gamma (float): The discount factor.
            tau (float): The GAE parameter.

        Returns:
            gae (torch.Tensor): Tensor of shape torch.Size([64]) containing the GAE for each timestep.
        """
        gae = torch.zeros_like(rewards)
        advantage = 0

        for i in reversed(range(len(rewards)-1)):
            delta = rewards[i] + gamma * next_value * masks[i] - values[i]
            advantage = delta + gamma * tau * advantage * masks[i]
            gae[i] = advantage[i]
            next_value = values[i+1]

        if normalize:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        return gae

    def generalized_advantage_estimate(
        value_old_state: torch.Tensor,
        value_new_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
        lamda: float = 0.95,
    ) -> Any:
        """
        Get generalized advantage estimate of a trajectory
        gamma: trajectory discount (scalar)
        lamda: exponential mean discount (scalar)
        value_old_state: value function result with old_state input
        value_new_state: value function result with new_state input
        reward: agent reward of taking actions in the environment
        done: flag for end of episode
        """
        # batch_size = done.shape[0]

        advantage = np.zeros(self.batch_size + 1)

        for t in reversed(range(self.batch_size)):
            delta = (
                reward[t] + (gamma * value_new_state[t] * done[t]) - value_old_state[t]
            )
            advantage[t] = delta + (gamma * lamda * advantage[t + 1] * done[t])

        value_target = advantage[:batch_size] + np.squeeze(value_old_state)

        return advantage[:batch_size], value_target


    def calculate_returns_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation"""
        returns = []
        advantages = []
        episode_return = 0
        prev_value = 0

        # Loop through rewards in reverse order
        for i in reversed(range(len(rewards))):
            # Update episode return and advantage
            episode_return = rewards[i] + gamma * episode_return
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            advantage = delta + gamma * lam * prev_value

            # Append to returns and advantages
            returns.insert(0, episode_return)
            advantages.insert(0, advantage)

            # Update prev_value
            prev_value = values[i]

        # Convert to tensor and normalize advantages
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages
