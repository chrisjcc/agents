# Importing necessary libraries
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from torch.optim.lr_scheduler import StepLR #, LambdaLR, MultiStepLR, ExponentialLR

from actor_critic import ActorCriticModel
from gae import GAE

from data_logger import DataLogger

from utils.categorical_masked import CategoricalMasked

# Setting the seed for reproducibility
torch.manual_seed(42)


# Define the ActorCritic Agent
class ActorCriticAgent:
    """
    The ActorCriticAgent class defines an actor-critic reinforcement learning agent.
    """

    def __init__(
        self,
        state_dim: Any,
        action_dim: int,
        hidden_dim: int,
        device: Any,
        data_logger: Any,
        lr_step_size: int, # e.g. max_steps_per_episode
        lr: float = 0.01,
        lr_gamma: float = 0.1,
        gamma: float = 0.99,
        seed: int = 42,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        """
        Initializes the ActorCriticAgent.
        :param state_dim: The number of dimensions in the state space.
        :param action_dim: The number of dimensions in the action space.
        :param hidden_dim: The number of hidden units in the neural networks for actor and critic.
        :param lr: The learning rate for the optimizer.
        :param gamma: The discount factor for future rewards.
        :param seed: The random seed for reproducibility.
        :param value_coef: The magnitude of the critic loss.
        :param entropy_coef: The magnitude of the entropy regularization.
        """
        self.actor_critic = ActorCriticModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        # Define the optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Create the learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_gamma)

        # Create an instance of GAE
        self.gae = GAE(gamma=0.99, lambda_=0.95)

        self.gamma = gamma
        self.seed = seed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.data_logger = data_logger
        self.device = device
        self.set_seed()

    def set_seed(self) -> None:
        """
        Set the seed value for generating random numbers within the environment.
        """
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        terminated: torch.Tensor,
        action_distribution: Any,
        next_action_distribution: Any,
        indices: torch.Tensor,
        weight: torch.Tensor,
        use_gae: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the ActorCriticAgent.
        :param state: The current state of the environment.
        :param action: The action taken within the environment.
        :param reward: The reward obtained for taking the action in the current state.
        :param next_state: The next state visited by taking the action in the current state.
        :param next_action: The next action taken within the environment.
        :param terminated: A boolean indicating whether the episode has terminated.
        """

        # Assert that state is not None
        assert state is not None, "State cannot be None"

        # Assert that action is not None
        assert action is not None, "Action cannot be None"

        # Assert that reward is not None
        assert reward is not None, "Reward cannot be None"

        # Assert that next_state is not None
        assert next_state is not None, "Next state cannot be None"

        # Assert that next_action is not None
        assert next_action is not None, "Next action cannot be None"

        # Assert that terminated is not None
        assert terminated is not None, "Terminated cannot be None"

        # Assert that action_distribution is not None
        assert action_distribution is not None, "Action distribution cannot be None"

        # Assert that next_action_distribution is not None
        assert (
            next_action_distribution is not None
        ), "Next action distribution cannot be None"

        # Assert that indices is not None
        assert indices is not None, "Indices cannot be None"

        # Assert that weight is not None
        assert weight is not None, "Weight cannot be None"

        # The concept of "returns" is used to estimate the cumulative future rewards
        # that an agent can expect to receive from a particular state-action sequence.
        # It provides a measure of the expected long-term value of taking a specific action in a given state.
        # Using returns instead of just the immediate reward value allows the agent to consider the future consequences of its actions.
        # Returns are calculated by summing up the discounted rewards from the current time step to the end of the episode.
        # The discount factor, typically denoted as gamma (γ), is a value between 0 and 1 that determines
        # the importance of immediate rewards compared to future rewards.
        # A higher gamma value places more emphasis on long-term rewards, while a lower gamma value prioritizes immediate rewards.
        #returns = self.gae.calculate_returns(reward, terminated)

        # Estimate value function V(s) for the current state
        state_value = self.actor_critic.critic(state)

        state_value = state_value.view(-1) # TODO: look into critic-network forward function to improve this

        # Estimate value function V(s') for the next state
        next_state_value = self.actor_critic.critic(next_state)

        next_state_value = next_state_value.view(-1) # TODO: look into critic-network forward function to improve this

        # Calculate Q-value estimates for the current and next state-action pairs
        q_value = self.actor_critic.critic.evaluate(state, action)

        q_value = q_value.view(-1) # TODO: look into critic-network forward function to improve this

        #head_masked = CategoricalMasked(logits=logits_or_qvalues, mask=mask)
        #print(head_masked.probs) # Impossible action are  masked
        #print(head_masked.entropy())

        # Calculate Q-value estimates for next state-action pairs
        next_q_value = self.actor_critic.critic.evaluate(next_state, next_action)  ## TODO: even used???

        next_q_value = next_q_value.view(-1) # TODO: look into critic-network forward function to improve this

        # Use a corrected masking of terminal states Q(s',a) values
        #next_q_value = torch.where(
        #    terminated > 0.0,
        #    torch.zeros_like(next_q_value),
        #    next_q_value,
        #)

        # Calculate the standard Temporal Difference (TD) learning, TD(0),
        # target Q-value is calculated based on the next state-action pair, using the standard TD target.
        target_q_value = reward + self.gamma * (1.0 - terminated) * next_state_value
        #target_q_value = returns + self.gamma * (1.0 - terminated) * next_state_value

        #batch_norm = nn.BatchNorm1d(num_features=1, track_running_stats=False).to(self.device)  # track_running_stats=True (default)
        #target_q_value = batch_norm(target_q_value.unsqueeze(1)).squeeze(1)  # Add an extra dimension to match the expected input shape of `nn.BatchNorm1d()` and then remove the extra dimension to revert back to the original shape of the target Q-value

        # TD_error = |target Q(s', a') - Q(s, a)|,
        # where taget Q(s', a') = r + γ * V(s'), used in PER.
        td_error = torch.abs(target_q_value - q_value)

        # The GAE combines the immediate advantate (one-step TD error) and the estimated future advantages
        # using the GAE parameter (lambda) and the discount factor (gamma).
        # The eligibility trace is used to assign credit (or blame) for the TD error occurring on a given step
        # to the previous steps.
        # Eligibility traces are the primary mechanisms of temporal credit assignment in TD learning
        # The GAE trace is calculated based on the one-step TD error using the following formula:
        #   GAE(s, a) = ∑[(γ * λ)^t * δt]
        # Where:
        #   γ is the discount factor, representing the agent's preference for immediate rewards over future rewards.
        #   λ is the GAE parameter, controlling the trade-off between bias and variance in the advantage estimation.
        #   t is the time step in the trajectory.
        #   δt is the one-step TD error, calculated as δt = r(t) + γ * V(s(t+1)) - V(s(t)).
        # The GAE trace is a sum of weighted one-step TD errors across the trajectory.
        # It combines the immediate advantage (one-step TD error) with the estimated future advantages
        # (through powers of γ * λ) to form a more robust estimate of the advantage function.
        if use_gae:
            # GAE(t) = target_Q(t) - V(s(t))
            # Where:
            #   GAE(t) is the GAE trace for the state-action pair at time step t.
            #   target_Q(t) is the target-Q value for the state-action pair at time step t.
            #   V(s(t)) is the estimated value function (state value) for the state s(t).
            # Calculate the advantages using GAE with eligibility trace
            advantage = self.gae.calculate_gae_eligibility_trace(
                td_error,
                terminated,
                normalize=True
            )
        else:
            # Calculate advantage: A(state, action) = Q(state, action) - V(state), as means for variance reduction.
            # Q(state, action) can be obtained by calling the evaluate method with the given state-action pair as input,
            # and V(state) can be obtained by calling the forward method with just the state as input.
            # Assuming next_state_value and state_value are 1-D tensors of shape [64]
            advantage = target_q_value - state_value

        # Calculate critic loss, weighted by importance sampling factor
        # target size (torch.Size([128, 15, 1])) that is different to the input size (torch.Size([128, 15, 128]))
        critic_loss = torch.mean(weight * F.smooth_l1_loss(target_q_value, q_value))

        # Calculate the entropy based on next state-action pair
        entropy = next_action_distribution.entropy().mean()

        # Calculate actor log-probability
        action_log_prob = action_distribution.log_prob(action)

        # Make sure the shape of weight matches the shape of action_log_prob
        action_log_prob = action_log_prob.squeeze(dim=0)

        assert weight.shape == action_log_prob.shape, "Weight and action_log_prob shape mismatch."

        # Make sure the shape of action_log_prob and advantage match
        assert action_log_prob.shape == advantage.shape, "action_log_prob and advantage shape mismatch."

        # Compute the actor loss, taking into account the importance sampling factor for weighting
        actor_loss = -torch.mean(weight * action_log_prob * advantage) #advantage.squeeze(dim=2)

        # Calculate total loss
        loss = self.value_coef * critic_loss + actor_loss - self.entropy_coef * entropy

        # Update episode total loss, actor loss, and crtic loss
        self.data_logger.update_episode_cumulative_total_loss(loss.item())
        self.data_logger.update_episode_cumulative_actor_loss(actor_loss.item())
        self.data_logger.update_episode_cumulative_critic_loss(critic_loss.item())
        self.data_logger.update_episode_cumulative_entropy(entropy.item())

        # Zero out gradients
        self.optimizer.zero_grad()

        # Calculate backpropagation
        loss.backward()

        for name, parameter in self.actor_critic.named_parameters():
            if parameter.grad is not None:
                # Access the gradient values
                gradients = parameter.grad.data

                # Log the gradients
                #print(f"Gradients for {name}: {gradients.mean()}")
                self.data_logger.update_episode_cumulative_gradients(gradients.mean().item())

        # Apply gradient clipping separately for actor and critic networks
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.actor.parameters(), max_norm=0.5, norm_type=2
        )
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.critic.parameters(), max_norm=0.5, norm_type=2
        )

        # Apply update rule to neural network weights
        self.optimizer.step()

        # Step the learning rate scheduler
        self.scheduler.step()

        # Log learning rate
        self.data_logger.log_learning_rate(self.optimizer)

        # Increment the step attribute
        self.data_logger.increment_step()

        return indices, td_error

    def state_dict(self):
        info = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "actor_policy": self.actor_critic.actor.state_dict(),
        }
        return info

    def update_critic(state: torch.Tensor, action: torch.Tensor, target_q_value: torch.Tensor) ->None:
        """
        This method updates the parameters of the critic network based on the TD-error or loss between
        the predicted Q-value  and the target Q-value. It involves computing the gradients
        and performing backpropagation.
        """
        pass

    def update_actor(state: torch.Tensor) -> None:
        """
        This method updates the parameters of the actor network using the policy gradient or advantage-based methods.
        It involves computing the gradients of the actor network's parameters with respect to the action log-probabilities
        and advantages, and performing backpropagation.
        """
        pass

if __name__ == "__main__":
    """Highway-v0 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Name the environment to be used
    env_name: str = "highway-fast-v0"
    max_episode_steps = 600  # default

    env: gym.Env[Any, Any] = gym.make(
        env_name,
        render_mode="human",
        max_episode_steps=max_episode_steps,
    )

    action_type = "DiscreteMetaAction" #"ContinuousAction" #"DiscreteAction"  # "ContinuousAction"

    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted",
            "normalize": False
        },
        "action" :{
            "type": action_type
        },
        "duration": 20,
        "vehicles_count": 20,
        "collision_reward": -1,
        "high_speed_reward": 0.4
    }

    env.configure(config)

    # We first check if state_shape is None. If it is None, we raise a ValueError.
    # Otherwise, we access the first element of state_shape using its index and
    # using the int() function.
    state_shape = env.observation_space.shape

    if state_shape is None:
        raise ValueError("Observation space shape is None.")

    state_dim = (15, 5) #int(state_shape[0])

    # Get action spaces
    action_space = env.action_space

    action_dim = int(action_space.n)

    # Initialize Data logging
    data_logger = DataLogger()
    data_logger.initialize_writer()

    # Actor-Critic hyperparameters
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01
    hidden_dim = 256
    lr_gamma = 0.1

    # Initialize Actor-Critic network
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
        data_logger=data_logger,
        lr_step_size=max_episode_steps,
        lr_gamma=lr_gamma,
    )

    # Get state spaces
    state_ndarray, info = env.reset(seed=42)

    # Convert next state to shape (batch_size, channe, width, height)
    state = torch.tensor(state_ndarray, dtype=torch.float32).to(device)
    state = state.flatten()
    batch_size = 1
    state = state.unsqueeze(0).expand(batch_size, -1)

    # Set variables
    total_reward = 0.0
    step_count = 0
    done = False

    # This loop constitutes one epoch
    while not done:
        print(f"Step: {step_count}")
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        # with torch.no_grad():
        # Pass the state through the Actor model to obtain a probability distribution over the actions
        action, action_distribution = agent.actor_critic.sample_action(state)

        # Take the action in the environment and observe the next state, reward, and done flag
        (
            next_state_ndarray,
            reward_ndarray,
            terminated_ndarray,
            truncated_ndarray,
            info,
        ) = env.step(action.squeeze().cpu().detach().numpy())

        # Convert next state to shape (batch_size, channe, width, height)
        next_state = torch.tensor(next_state_ndarray, dtype=torch.float32).to(device)
        next_state = next_state.flatten()
        next_state = next_state.unsqueeze(0).expand(batch_size, -1)
        reward = torch.tensor([reward_ndarray], dtype=torch.float32).to(device)
        terminated = torch.tensor([terminated_ndarray], dtype=torch.float32).to(device)
        truncated = torch.tensor([truncated_ndarray], dtype=torch.float32).to(device)

        # Create a tensor for indices with the same dimensions and structure as the reward tensor
        indices = torch.ones_like(torch.as_tensor(reward, dtype=torch.float32), dtype=torch.long).to(device)

        # Create a tensor for weight with the same dimensions and structure as the reward tensor
        weight = torch.ones_like(torch.as_tensor(reward, dtype=torch.float32), dtype=torch.long).to(device)

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = agent.actor_critic.sample_action(
            next_state
        )

        # Update actor-critic network
        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            terminated=terminated,
            action_distribution=action_distribution,
            next_action_distribution=next_action_distribution,
            indices=indices,
            weight=weight
        )

        # Update total reward
        total_reward += float(reward.item())
        print(f"\tTotal reward: {total_reward:.2f}")
        step_count += 1

        state = next_state

        # Update if the environment is done
        done = terminated or truncated
