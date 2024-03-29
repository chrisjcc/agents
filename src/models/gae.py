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
        rewards: torch.Tensor,
        q_values: torch.Tensor,
        next_q_values: torch.Tensor,
        dones: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the Generalized Advantage Estimation (GAE) with eligibility trace.
        :param rewards: Tensor of shape [batch_size] containing rewards.
        :param q_values: Tensor of shape [batch_size] containing state q_values at time t.
        :param next_q_values: Tensor of shape [batch_size] containing state values at time t+1.
        :param dones: Tensor of shape [batch_size] indicating whether the episode is terminated.
        :param normalize: Whether to normalize the advantage values (optional).
        :return: Tensor of shape [batch_size] containing the advantages.
        """
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(rewards.shape[0])):
            delta = (
                rewards[t]
                + self.gamma * next_q_values[t] * (1 - dones[t])
                - q_values[t]
            )
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
