# filename: data_logger.py
import os

import torch
from torch.utils.tensorboard import SummaryWriter


class DataLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.writer = None
        self.episode_cumulative_total_loss = 0.0
        self.episode_cumulative_actor_loss = 0.0
        self.episode_cumulative_critic_loss = 0.0
        self.episode_cumulative_entropy = 0.0
        self.episode_num = 0
        self.step_count = 0
        self.global_step_count = 0

    def initialize_writer(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step_count):
        self.writer.add_scalar(tag, value, step_count)

    def log_episode_average_total_loss(self):
        episode_loss_avg = (
            self.episode_cumulative_total_loss / self.step_count
        )  # Calculate average episode loss

        if self.writer is not None:
            self.writer.add_scalar(
                "Episode Average Loss", episode_loss_avg, self.episode_num
            )

    def log_learning_rate(self, optimizer):
        lr = self._get_learning_rate(optimizer)
        self.writer.add_scalar("Learning Rate", lr, self.global_step_count)
        self.global_step_count += 1

    def _get_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def log_actor_critic(self):  # Cumulative losses
        episode_actor_loss_avg = (
            self.episode_cumulative_actor_loss / self.step_count
        )  # Calculate average episode loss
        episode_critic_loss_avg = (
            self.episode_cumulative_critic_loss / self.step_count
        )  # Calculate average episode loss

        if self.writer is not None:
            self.writer.add_scalar(
                "Actor Average Loss", episode_actor_loss_avg, self.episode_num
            )
            self.writer.add_scalar(
                "Critic Average Loss", episode_critic_loss_avg, self.episode_num
            )

    def log_entropy(self):
        episode_entropy_avg = (
            self.episode_cumulative_entropy / self.step_count
        )  # Calculate average episode loss

        if self.writer is not None:
            self.writer.add_scalar(
                "Average Entropy", episode_entropy_avg, self.episode_num
            )

    def increment_step(self):
        self.step_count += 1

    def update_episode_cumulative_total_loss(self, step_loss):
        self.episode_cumulative_total_loss += step_loss

    def update_episode_cumulative_actor_loss(self, step_loss):
        self.episode_cumulative_actor_loss += step_loss

    def update_episode_cumulative_critic_loss(self, step_loss):
        self.episode_cumulative_critic_loss += step_loss

    def update_episode_cumulative_entropy(self, entropy):
        self.episode_cumulative_entropy += entropy

    def update_episode_num(self):
        self.episode_num += 1

    def reset(self):
        self.episode_cumulative_total_loss = 0.0
        self.episode_cumulative_actor_loss = 0.0
        self.episode_cumulative_critic_loss = 0.0
        self.episode_cumulative_entropy = 0.0
        self.step_count = 0

    def close(self):
        if self.writer is not None:
            self.writer.close()


# Example Usage:
# Assuming you have already defined and initialized your actor-critic model and optimizer
# actor_critic = ActorCritic()
# optimizer = torch.optim.Adam(actor_critic.parameters(), lr=0.001)
# data_logger = DataLogger()
# data_logger.initialize_writer()

# Inside your training loop:
# for episode in range(num_episodes):
#     episode_reward = 0
#     for step in range(max_steps_per_episode):
#         # Perform one step of the training process here and update the model and optimizer
#         # ...
#         data_logger.log_actor_critic(actor_critic)
#         data_logger.log_reward(episode_reward)
#         data_logger.log_learning_rate(optimizer)
#         data_logger.increment_step()

# After training is complete, close the data logger
# data_logger.close()
