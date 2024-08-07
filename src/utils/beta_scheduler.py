# Importing necessary libraries
class BetaScheduler:
    """
    Prioritized Experience Replay (PER) importance sampling: Linearly anneals the importance-sampling correction factor β
    from an initial value β_0 to 1 over time, allowing for gradual adjustment of the correction factor during learning.
    """

    def __init__(self, initial_beta: float, total_steps: int):
        """
        Initialize BetaScheduler
        :param initial_beta: initial value of beta.
        :param total_steps: total number of training steps.
        """
        self.initial_beta = initial_beta
        self.final_beta = 1.0
        self.total_steps = total_steps

    def get_beta(self, current_step: int) -> float:  # compute_beta(self, step: int)
        """
        Compute the beta factor for importance sampling weight annealing.

        :param step: The current training step.
        :return: The beta factor for this training step.
        """
        # Define a schedule for beta annealing, e.g., linear annealing from 0.4 to 1.0
        annealing_fraction = min(current_step / self.total_steps, 1.0)
        beta = self.initial_beta + annealing_fraction * (
            self.final_beta - self.initial_beta
        )

        return beta

