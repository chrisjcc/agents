import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

# Add the directory to the Python module search path
module_dir = os.path.dirname(os.path.abspath(__file__))
actor_critic_agent_dir = os.path.join(
    module_dir, ".."
)  # Adjust the relative path if needed
sys.path.append(actor_critic_agent_dir)

from actor_critic import ActorCritic


class TestActorCritic(unittest.TestCase):
    def setUp(self):
        self.state_dim = 96
        self.state_channel = 3
        self.action_dim = 3
        self.max_action = 1.0
        self.device = torch.device("cpu")
        self.actor_critic = ActorCritic(
            state_dim=self.state_dim,
            state_channel=self.state_channel,
            action_dim=self.action_dim,
            max_action=self.max_action,
            device=self.device,
        )

    def test_forward(self):
        state = torch.randn(1, self.state_channel, self.state_dim, self.state_dim)
        action, action_distribution, state_value = self.actor_critic(state)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertIsInstance(action_distribution, torch.distributions.Normal)
        self.assertEqual(state_value.shape, (1, 1))

    def test_sample_action(self):
        state = torch.randn(1, self.state_channel, self.state_dim, self.state_dim)
        action, action_distribution = self.actor_critic.sample_action(state)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertIsInstance(action_distribution, torch.distributions.Normal)

    def test_evaluate(self):
        state = torch.randn(1, self.state_channel, self.state_dim, self.state_dim)
        action = torch.randn(1, self.action_dim)
        q_value = self.actor_critic.evaluate(state, action)
        self.assertEqual(q_value.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
