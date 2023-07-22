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

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Flatten the probs array to make it one-dimensional
        probs = probs.flatten()

        # Check that probs has the same size as self.buffer
        if len(probs) != len(self.buffer):
            raise ValueError("probs and buffer have different sizes")

        # Check that probs contains at least one positive value
        if np.any(probs <= 0):
            raise ValueError("probs must contain at least one positive value")

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        # We apply a common priority update rule is the proportional prioritization,
        # where the priority of an experience ii is set proportional to a value derived from its TD error or loss.
        # Experiences with higher TD errors are given higher priorities, making them more likely to be sampled
        # in future training iterations.
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
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

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        # Wrap indices within the valid range
        indices = torch.tensor(indices) % len(self.buffer) % len(self.buffer)
        indices = indices.unsqueeze(dim=1)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(dim=1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        # Ensure indices and td_errors have the same length
        assert len(indices) == len(
            td_errors
        ), "indices and td_errors must have the same length."

        # Update the priorities
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority

    def clear_priorities(self):
        self.priorities = deque(maxlen=self.capacity)

    def ready(self, capacity: int) -> bool:
        return len(self.buffer) >= self.buffer.maxlen

    def __len__(self) -> int:
        return len(self.buffer)
