# Importing necessary libraries
from collections import deque
from typing import Tuple

import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action, rewar, next_state, done):
        """
        This code initializes self.priorities to all ones by default,
        and then updates it with the new priority whenever a new experience
        is added. It also makes sure that the length of self.priorities
        is always the same as the length of self.buffer.
        """
        experience = (state, action, rewar, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1.0))

        if len(self.priorities) > len(self.buffer):
            self.priorities.popleft()

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor]:
        if len(self.buffer) < batch_size:
            return None

        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Check that probs has the same size as self.buffer
        if len(probs) != len(self.buffer):
            raise ValueError("probs and buffer have different sizes")

        indices = torch.multinomial(probs, batch_size, replacement=True)
        samples = [self.buffer[idx] for idx in indices]

        # We apply a common priority update rule is the proportional prioritization,
        # where the priority of an experience ii is set proportional to a value derived from its TD error or loss.
        # Experiences with higher TD errors are given higher priorities, making them more likely to be sampled
        # in future training iterations.
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.clone().detach()

        # Note: The importance weight sampling is used to correct for the bias introduced by using priorities to sample experiences.
        # When we use prioritized sampling, experiences with higher priorities are sampled more frequently.
        # This can introduce bias in the learning process because we are not sampling experiences uniformly at random
        # from the replay buffer.
        # To correct for this bias, we use the importance weights when calculating the loss during training.
        # The importance weight for each experience is computed as the inverse of its probability of being sampled.
        # By including the importance weights in the loss calculation, we give less weight to experiences that were oversampled
        # due to higher priorities, and more weight to experiences that were undersampled. This correction helps to mitigate the bias
        # and improve the stability of the learning process.

        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.stack(states).squeeze(dim=1)
        actions = torch.stack(actions).squeeze(dim=1)
        rewards = torch.stack(rewards).squeeze(dim=1)
        next_states = torch.stack(next_states).squeeze(dim=1)
        dones = torch.stack(dones).squeeze(dim=1)

        # Wrap indices within the valid range
        indices = indices % len(self.buffer)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(
        self,
        indices: torch.Tensor,
        td_errors: torch.Tensor,
        alpha: float = 0.7,
        epsilon: float = 1e-6,
    )-> None:
        # Ensure indices and td_errors have the same length
        assert len(indices) == len(
            td_errors
        ), "indices and td_errors must have the same length."

        # Update the priority scores in PER with IS
        priorities = torch.pow(torch.abs(td_errors) + epsilon, alpha)

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


    def ready(self, capacity: int) -> bool:
        return len(self.buffer) >= self.buffer.maxlen

    def __len__(self) -> int:
        return len(self.buffer)
