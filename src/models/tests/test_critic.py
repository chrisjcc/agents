import os
import sys
import unittest

import gymnasium as gym
import torch

# Add the directory containing actor_critic_agent.py to the Python module search path
module_dir = os.path.dirname(os.path.abspath(__file__))
actor_critic_agent_dir = os.path.join(
    module_dir, ".."
)  # Adjust the relative path if needed
sys.path.append(actor_critic_agent_dir)

from neural_networks.critic_network import Critic


class TestCritic(unittest.TestCase):
    def setUp(self) -> None:
        self.state_dim = 96
        self.state_channel = 3
        self.action_dim = 3
        self.hidden_dim = 64
        self.critic = Critic(
            self.state_dim,
            self.state_channel,
            self.action_dim,
            self.hidden_dim,
        )

    def test_calculate_conv_output_dims(self) -> None:
        input_dims = (1, self.state_channel, self.state_dim, self.state_dim)
        output_dims = self.critic.calculate_conv_output_dims(input_dims)
        expected_output_dims = 18432
        self.assertEqual(output_dims, expected_output_dims)

    def test_forward(self) -> None:
        state = torch.randn(1, self.state_channel, self.state_dim, self.state_dim)
        action = torch.randn(1, self.action_dim)
        q_value = self.critic.forward(state, action)
        self.assertIsInstance(q_value, torch.Tensor)
        self.assertEqual(q_value.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
