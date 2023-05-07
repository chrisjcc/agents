import unittest

import torch

from neural_networks.actor_network import Actor


class TestActor(unittest.TestCase):
    def setUp(self) -> None:
        self.state_dim = 3
        self.action_dim = 2
        self.max_action = 1.0
        self.hidden_dim = 64
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
            self.max_action,
            self.hidden_dim,
        )

    def test_calculate_conv_output_dims(self) -> None:
        input_dims = (self.state_dim, self.state_dim, self.action_dim)
        output_dims = self.actor.calculate_conv_output_dims(input_dims)
        expected_output_dims = 128
        self.assertEqual(output_dims, expected_output_dims)

    def test_forward(self) -> None:
        state = torch.randn(1, self.state_dim, self.state_dim, self.action_dim)
        mean, std = self.actor.forward(state)
        self.assertIsInstance(mean, torch.Tensor)
        self.assertIsInstance(std, torch.Tensor)
        self.assertEqual(mean.shape, (1, self.action_dim))
        self.assertEqual(std.shape, (1, self.action_dim))


if __name__ == "__main__":
    unittest.main()
