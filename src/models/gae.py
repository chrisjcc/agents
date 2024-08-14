import torch
import torch.nn.functional as F

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
        :param lambda_: The GAE parameter, controls the trade-off between bias and variance in advantage estimation.
        """
        self.gamma = gamma
        self.lambda_ = lambda_

    def calculate_gae_eligibility_trace(
        self,
        rewards: torch.Tensor,
        value_preds: torch.Tensor,
        masks: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the Generalized Advantage Estimation (GAE) with eligibility trace.

        :param rewards: Tensor of shape [batch_size] containing rewards.
        :param value_preds: Tensor of shape [batch_size] containing value function predictions.
        :param masks: Tensor of shape [batch_size] indicating whether the episode continues (1) or terminates (0).
        :param normalize: Whether to normalize the advantage values (optional).
        :return: Tensor of shape [batch_size] containing the advantages.
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(rewards.shape[0] - 1)):
            delta = rewards[t] + self.gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            last_gae = delta + self.gamma * self.lambda_ * masks[t] * last_gae
            advantages[t] = last_gae

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def calculate_returns(
        self,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the Returns based on the rewards.

        :param rewards: Tensor of shape [batch_size] containing rewards.
        :param masks: Tensor of shape [batch_size] indicating whether the episode continues (1) or terminates (0).
        :return: Tensor of shape [batch_size] containing the returns.
        """
        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(rewards.shape[0])):
            running_return = rewards[t] + self.gamma * masks[t] * running_return
            returns[t] = running_return

        return returns
