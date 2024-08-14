# Importing necessary libraries
import torch

class GAE:
    """
    Generalized Advantage Estimation (GAE) with eligibility trace to compute the advantages.
    GAE allows you to control the trade-off between bias and variance in advantage estimation.
    (source: https://arxiv.org/abs/1506.02438).
    """

    def __init__(self, gamma: float, lambda_: float):
        """
        Initializes the GAE object.
        :param gamma: The discount factor for future rewards.
        :param lambda_: The GAE parameter controls the trade-off between bias and variance in advantage estimation.
        """
        self.gamma = gamma
        self.lambda_ = lambda_

    def calculate_gae_eligibility_trace(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the Generalized Advantage Estimation (GAE) with eligibility trace.

        :param rewards: Tensor of shape [batch_size] containing rewards.
        :param values: Tensor of shape [batch_size] containing state values at time t.
        :param next_values: Tensor of shape [batch_size] containing state values at time t+1.
        :param dones: Tensor of shape [batch_size] indicating whether the episode is terminated.
        :param normalize: Whether to normalize the advantage values (optional).
        :return: Tensor of shape [batch_size] containing the advantages.

        Example:
            >>> gae = GAE(gamma=0.99, lambda_=0.95)
            >>> rewards = torch.tensor([0.0, 2.0, 1.0, 0.0])
            >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0])
            >>> next_values = torch.tensor([5.0, 6.0, 7.0, 8.0])
            >>> dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
            >>> advantages = gae.calculate_gae_eligibility_trace(rewards, values, next_values, dones)
        """
        # Caculate the temporal difference error, δ.
        td_errors = self.calculate_td_errors(rewards, values, next_values, dones)

        if td_errors.shape != dones.shape:
            raise ValueError("td_errors and dones must have the same shape")

        advantages = torch.zeros_like(td_errors)
        gae = 0.0

        for t in reversed(range(rewards.shape[0])):
            # Formula: gae = δ + γλ(1−dones[t])gae
            gae = td_errors[t] + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def calculate_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the Returns based on the rewards.

        :param rewards: Tensor of shape [batch_size] containing rewards.
        :param dones: Tensor of shape [batch_size] indicating whether the episode continues (0) or terminates (1).
        :return: Tensor of shape [batch_size] containing the returns.

        Example:
            >>> gae = GAE(gamma=0.99, lambda_=0.95)
            >>> rewards = torch.tensor([0.0, 2.0, 1.0, 0.0]))
            >>> dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
            >>> returns = gae.calculate_returns(rewards, dones)
        """
        if rewards.shape != dones.shape:
            raise ValueError("rewards and dones must have the same shape")

        returns = torch.zeros_like(rewards)
        running_return = 0.0

        for t in reversed(range(rewards.shape[0])):
            running_return = rewards[t] + self.gamma * (1 - dones[t]) * running_return
            returns[t] = running_return

        return returns

    def calculate_td_errors(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the Temporal Difference (TD) errors.

        :param rewards: Tensor of shape [batch_size] containing rewards.
        :param values: Tensor of shape [batch_size] containing state values at time t.
        :param next_values: Tensor of shape [batch_size] containing state values at time t+1.
        :param dones: Tensor of shape [batch_size] indicating whether the episode is terminated.
        :return: Tensor of shape [batch_size] containing TD-errors.
        
        Example:
            >>> gae = GAE(gamma=0.99, lambda_=0.95)
            >>> rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
            >>> values = torch.tensor([0.5, 1.5, 2.5, 3.5])
            >>> next_values = torch.tensor([1.5, 2.5, 3.5, 0.0])
            >>> dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
            >>> td_errors = gae.calculate_td_errors(rewards, values, next_values, dones)
        """
        if not all(tensor.shape == rewards.shape for tensor in [values, next_values, dones]):
            raise ValueError("All input tensors must have the same shape")

        td_errors = rewards + self.gamma * next_values * (1 - dones) - values

        return td_errors
