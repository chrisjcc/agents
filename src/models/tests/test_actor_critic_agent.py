import unittest
import torch
from torch.distributions import Categorical, Normal

from neural_networks.actor_network import Actor
from neural_networks.critic_network import Critic
from actor_critic_agent import ActorCriticAgent

class TestActorCriticAgent(unittest.TestCase):
    def setUp(self):
        # Set up the test environment
        state_dim = 96
        state_channel = 3
        action_dim = 3
        max_action = 1.0
        hidden_dim = 64
        device = torch.device("cpu")
        self.agent = ActorCriticAgent(
            state_dim,
            state_channel,
            action_dim,
            max_action,
            hidden_dim,
            device
        )

    def test_update(self):
        # Test the update method of the agent
        state = torch.randn(1, 3, 96, 96)
        action = torch.randn(1, 3)
        reward = torch.tensor([1.0])
        next_state = torch.randn(1, 3, 96, 96)
        next_action = torch.randn(1, 3)
        terminated = torch.tensor([0.0])

        # Selection action using actor network for state
        action_mean, action_std = self.agent.actor_critic.actor(state)

        # Sample an action from the distribution
        action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore

        # Select action using actor network next state
        next_action_mean, next_action_std = self.agent.actor_critic.actor(next_state)

        # Sample an action from the distribution
        next_action_distribution = Normal(loc=next_action_mean, scale=next_action_std)  # type: ignore

        self.agent.update(
            state,
            action,
            reward,
            next_state,
            next_action,
            terminated,
            action_distribution,
            next_action_distribution
        )

        # Assert that the update method runs without any errors
        self.assertTrue(True)

    def test_assertions(self):
        # Test the assertions in the update method
        state = torch.randn(1, 3, 96, 96)
        action = torch.randn(1, 3)
        reward = torch.tensor([1.0])
        next_state = torch.randn(1, 3, 96, 96)
        next_action = torch.randn(1, 3)
        terminated = torch.tensor([0.0])

        # Selection action using actor network for state
        action_mean, action_std = self.agent.actor_critic.actor(state)

        # Sample an action from the distribution
        action_distribution = Normal(loc=action_mean, scale=action_std)  # type: ignore

        # Select action using actor network next state
        next_action_mean, next_action_std = self.agent.actor_critic.actor(next_state)

        # Sample an action from the distribution
        next_action_distribution = Normal(loc=next_action_mean, scale=next_action_std)  # type: ignore

        # Test assertion for state
        with self.assertRaises(AssertionError):
            self.agent.update(
                None,
                action,
                reward,
                next_state,
                next_action,
                terminated,
                action_distribution,
                next_action_distribution
            )

        # Test assertion for action
        with self.assertRaises(AssertionError):
            self.agent.update(
                state,
                None,
                reward,
                next_state,
                next_action,
                terminated,
                action_distribution,
                next_action_distribution
            )

        # Test assertion for reward
        with self.assertRaises(AssertionError):
            self.agent.update(
                state,
                action,
                None,
                next_state,
                next_action,
                terminated,
                action_distribution,
                next_action_distribution
            )

        # Test assertion for next_state
        with self.assertRaises(AssertionError):
            self.agent.update(
                state,
                action,
                reward,
                None,
                next_action,
                terminated,
                action_distribution,
                next_action_distribution
            )

        # Test assertion for terminated
        with self.assertRaises(AssertionError):
            self.agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                None,
                action_distribution,
                next_action_distribution
            )

        # Test assertion for action_distribution
        with self.assertRaises(AssertionError):
            self.agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                terminated,
                None,
                next_action_distribution
            )

        # Test assertion for next_action_distribution
        with self.assertRaises(AssertionError):
            self.agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                terminated,
                action_distribution,
                None
            )
