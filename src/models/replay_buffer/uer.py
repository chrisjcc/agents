# Importing necessary libraries
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch

# Setting the seed for reproducibility
torch.manual_seed(0)


class UniformExperienceReplay(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, buffer_size: int) -> None:
        """
        Initializes the UniformExperienceReplay.

        :param buffer_size: The maximum memory size for the replay buffer.
        """
        self.buffer_size = buffer_size
        self.buffer: Deque[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Adds the state transition into the memory replay buffer.

        :param state: The current state of the environment.
        :param action: The action taken in the current state.
        :param reward: The reward received from the environment.
        :param next_state: The next state of the environment.
        :param done: Whether the episode has ended.
        """
        assert isinstance(state, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        assert isinstance(reward, torch.Tensor)
        assert isinstance(next_state, torch.Tensor)
        assert isinstance(done, torch.Tensor)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects a mini-batch of samples from the replay memory buffer.

        :param batch_size: The number of samples to include in a mini-batch.
        :return: A tuple of (state, action, reward, next_state, done) for the current episode.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(
            *[self.buffer[i] for i in indices]
        )

        return (
            torch.stack(state),
            torch.stack(action),
            torch.stack(reward),
            torch.stack(next_state),
            torch.stack(done),
        )
