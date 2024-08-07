# Importing necessary libraries
import torch
import torch.nn.functional as F

class GAE:
    """
    Generalized Advantage Estimation (GAE) with eligibility trace.
    """

    def __init__(self, gamma: float, lambda_: float):
        """
        Initializes the GAE object.
        :param gamma: The discount factor for future rewards.
        :param lambda_: The GAE parameter, controlling the trade-off between bias and variance in advantage estimation.
        """
        self.gamma = gamma
        self.lambda_ = lambda_

    def calculate_gae_eligibility_trace(
        self,
        td_errors: torch.Tensor,
        dones: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the Generalized Advantage Estimation (GAE) with eligibility trace.

        :param td_errors: Tensor of shape [batch_size] containing TD-errors.
        :param dones: Tensor of shape [batch_size] indicating whether the episode is terminated.
        :param normalize: Whether to normalize the advantage values (optional).
        :return: Tensor of shape [batch_size] containing the advantages.
        """
        advantages = torch.zeros_like(td_errors)

        for t in reversed(range(td_errors.shape[0])):
            #advantages[t] = td_errors[t] + self.gamma * self.lambda_ * advantages[t+1]
            advantages[t] = td_errors[t] + self.gamma * self.lambda_ * advantages[t]

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
        :return: Tensor of shape [batch_size] containing the rewards.
        """
        returns = torch.zeros_like(rewards)

        for t in reversed(range(rewards.shape[0])):
            #returns[t] = rewards[t] + self.gamma * (1 - dones[t]) * returns[t+1]
            returns[t] = rewards[t] + self.gamma * (1 - dones[t]) * returns[t]

        return returns

