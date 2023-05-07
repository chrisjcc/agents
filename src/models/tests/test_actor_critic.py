import unittest

import torch

from actor_critic import ActorCritic


class TestActorCritic(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.max_action = 1.0
        self.device = torch.device("cpu")
        self.actor_critic = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            device=self.device,
        )

    def test_forward(self):
        state = torch.randn(1, self.state_dim)
        action, action_distribution, state_value = self.actor_critic(state)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertIsInstance(action_distribution, torch.distributions.Normal)
        self.assertEqual(state_value.shape, (1, 1))

    def test_sample_action(self):
        state = torch.randn(1, self.state_dim)
        action, action_distribution = self.actor_critic.sample_action(state)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertIsInstance(action_distribution, torch.distributions.Normal)

    def test_evaluate(self):
        state = torch.randn(1, self.state_dim)
        action = torch.randn(1, self.action_dim)
        q_value = self.actor_critic.evaluate(state, action)
        self.assertEqual(q_value.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
