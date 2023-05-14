import unittest

import gymnasium as gym
import torch

from actor_critic_agent import ActorCriticAgent
from replay_buffer.replay_buffer import ReplayBuffer
from trainer_with_memory import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.env = gym.make(
            "CarRacing-v2",
            max_episode_steps=10,
            render_mode="rgb_array",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = self.env.action_space
        self.low = torch.from_numpy(self.action_space.low).to(self.device)
        self.high = torch.from_numpy(self.action_space.high).to(self.device)
        self.agent = ActorCriticAgent(
            state_dim=96,
            state_channel=3,
            action_dim=3,
            max_action=1.2,
            hidden_dim=256,
            gamma=0.99,
            lr=0.0001,
            value_coef=0.5,
            entropy_coef=0.01,
            device=self.device,
        )
        self.memory = ReplayBuffer(buffer_size=1024)
        self.trainer = Trainer(
            env=self.env,
            agent=self.agent,
            memory=self.memory,
            max_episodes=1,
            checkpoint_path="model_checkpoints",
            checkpoint_freq=1,
            batch_size=32,
            low=self.low,
            high=self.high,
            device=self.device,
        )
        state, info = self.env.reset()

    def test_env_step(self):
        action = (
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        state, reward, terminated, truncated = self.trainer.env_step(action)
        self.assertEqual(state.shape, torch.Size([1, 96, 96, 3]))
        self.assertIsInstance(reward, torch.Tensor)
        self.assertIsInstance(terminated, torch.Tensor)
        self.assertIsInstance(truncated, torch.Tensor)

    def test_train_step(self):
        for i in range(1024):
            self.memory.add(
                torch.randn(1, 3, 96, 96),  # state
                torch.randn(1, 3),  # action
                torch.randn(1),  # reward
                torch.randn(1, 3, 96, 96),  # next state
                torch.tensor([False] if i % 2 == 0 else [True]),  # done
            )
        self.trainer.train_step()
        self.assertEqual(len(self.memory), 1024)

    def test_train(self):
        self.trainer.train()
        self.assertGreater(len(self.memory), 0)


if __name__ == "__main__":
    unittest.main()
