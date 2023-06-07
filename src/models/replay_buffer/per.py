from collections import deque
from typing import List, Tuple

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
        self.priorities.append(max(self.priorities, default=1))

        if len(self.priorities) > len(self.buffer):
            self.priorities.popleft()


    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        if len(self.buffer) < batch_size:
            return None

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Check that probs has the same size as self.buffer
        if len(probs) != len(self.buffer):
            raise ValueError("probs and buffer have different sizes")

        # Check that probs contains at least one positive value
        if np.any(probs <= 0):
            raise ValueError("probs must contain at least one positive value")

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

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

    def update_priorities(self, indices: List, td_errors: List):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def clear_priorities(self):
        self.priorities = deque(maxlen=self.capacity)

    def ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)
