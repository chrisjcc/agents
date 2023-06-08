# Importing necessary libraries
import os
from typing import Any, Tuple

import numpy as np
import torch

from actor_critic_agent import ActorCriticAgent

# from replay_buffer.replay_buffer import ReplayBuffer
from replay_buffer.per import PrioritizedReplayBuffer

# Setting the seed for reproducibility
torch.manual_seed(0)


# Trainer class to train the Agent
class Trainer:  # responsible for running over the steps and collecting all the data
    """
    The Trainer class defines the training loop for the actor-critic agent.
    """

    def __init__(
        self,
        env: Any,
        agent: ActorCriticAgent,
        # memory: ReplayBuffer,
        memory: PrioritizedReplayBuffer,
        max_episodes: int,
        low: Any,
        high: Any,
        device: Any,
        checkpoint_path: str = "model_checkpoints",
        checkpoint_freq: int = 100,
        batch_size: int = 1024,
    ):
        """
        Initializes the Trainer.

        :param env: The OpenAI Gym environment.
        :param agent: The actor-critic agent.
        :param max_episodes: The maximum number of episodes to train for.
        :param checkpoint_path: The directory to save the agent's model in.
        :param checkpoint_freq: The frequency (in episodes) at which to save the agent's model.
        """
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        self.memory = memory
        self.batch_size = batch_size
        self.low = low
        self.high = high
        self.device = device

    def env_step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns state, reward and done flag given an action."""
        # Convert action from torch.Tensor to np.ndarray
        action = action.squeeze().cpu().detach().numpy()
        # action = action.squeeze(0).cpu().detach().numpy()

        # Take one step in the environment given the agent action
        state, reward, terminated, truncated, info = self.env.step(action)

        # Convert to tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)
        terminated = (
            torch.tensor(terminated, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        truncated = (
            torch.tensor(truncated, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        return state, reward, terminated, truncated

    def train_step(self):
        """
        Run a single training step in the OpenAI Gym environment.

        :param batch: The batch of transitions of state, action, reward, next_state, and done to learn from.
        :return: The next state, reward, and whether the episode is done.
        """
        # Sample a batch from the replay buffer
        # state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done, indices, weight = self.memory.sample(
            self.batch_size
        )

        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        # with torch.no_grad():
        _, action_distribution = self.agent.actor_critic.sample_action(
            state.squeeze(1)
        )

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = self.agent.actor_critic.sample_action(
            next_state.squeeze(1)
        )

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_next_action = torch.clamp(
            (((next_action + 1) / 2) * (self.high - self.low) + self.low),
            self.low,
            self.high,
        )

        # TODO: improve this step (couldn't this be handled earlier)
        state = torch.squeeze(state, dim=1)
        action = torch.squeeze(action, dim=1)
        reward = torch.squeeze(reward, dim=1)
        next_state = torch.squeeze(next_state, dim=1)
        clipped_next_action = torch.squeeze(clipped_next_action, dim=1)
        done = torch.squeeze(done, dim=1)

        indices = torch.squeeze(indices, dim=1)
        weight = torch.squeeze(weight, dim=1)

        # Update the neural networks
        self.agent.update(
            state,
            action,
            reward,
            next_state,
            clipped_next_action,
            done,
            action_distribution,
            next_action_distribution,
            indices,
            weight,
        )

    def train(self) -> None:
        """
        Run the training loop for the actor-critic agent.

        :param batch_size: The size of the batch of transitions
            of state, action, reward, next_state, and done to learn from.
        :return: A list of episode rewards.
        """
        print("Collecting trajectory samples based on random actions.")
        # Fill the replay buffer before starting training
        state_ndarray, info = self.env.reset()

        # Convert the state to a PyTorch tensor with shape (batch_size, channel, width, height)
        state = (
            torch.tensor(state_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )

        while len(self.memory) < self.memory.buffer.maxlen:
            done = False

            while not done:
                # Get an action from the policy network
                with torch.no_grad():
                    action, action_distribution = self.agent.actor_critic.sample_action(
                        state.squeeze(1)
                    )

                # Rescale the action to the range of the action space
                rescaled_action = ((action + 1) / 2) * (self.high - self.low) + self.low

                # Clip the rescaledaction to ensure it falls within the bounds of the action space
                clipped_action = torch.clamp(rescaled_action, self.low, self.high)

                # Take a step in the environment with the chosen action
                next_state, reward, terminated, truncated = self.env_step(action)

                # Convert next state to shape (batch_size, channe, width, height)
                next_state = next_state.permute(0, 3, 1, 2)

                done = terminated or truncated

                # Collect experience trajectory in replay buffer
                self.memory.add(state, clipped_action, reward, next_state, done)

                # Update the current state
                state = next_state

        for episode in range(self.max_episodes):
            print(f"Episode: {episode}")

            # Get state spaces
            state_ndarray, info = self.env.reset()

            # Convert next state to shape (batch_size, channe, width, height)
            state = (
                torch.tensor(state_ndarray, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
                .permute(0, 3, 1, 2)
            )

            # Set variables
            episode_reward = 0.0
            done = False
            step = 0

            while not done:
                # Update model parameters using TD error
                self.train_step()

                # Pass the state through the Actor model to obtain a probability distribution over the actions
                action, action_probs = self.agent.actor_critic.sample_action(state)

                # Rescale the action to the range of the action space
                rescaled_action = ((action + 1) / 2) * (self.high - self.low) + self.low

                # Clip the rescaledaction to ensure it falls within the bounds of the action space
                clipped_action = torch.clamp(rescaled_action, self.low, self.high)

                # Take the action in the environment and observe the next state, reward, and done flag
                next_state, reward, terminated, truncated = self.env_step(
                    clipped_action
                )
                # Convert next state to shape (batch_size, channe, width, height)
                next_state = next_state.permute(0, 3, 1, 2)

                done = terminated or truncated

                self.memory.add(state, clipped_action, reward, next_state, done)

                # Update episode reward
                episode_reward += float(torch.mean(reward))
                print(
                    f"Episode {episode}: Step {step}: Total reward = {episode_reward:.2f}"
                )

                # Save model checkpoint after each 10 episode
                if episode % self.checkpoint_freq == 0:
                    self.save_checkpoint(self.checkpoint_path, episode, episode_reward)

                state = next_state
                step += 1

            # Save last episode model
            self.save_checkpoint(
                self.checkpoint_path, self.max_episodes, episode_reward
            )

    def save_checkpoint(
        self,
        checkpoint_path: str,
        episode_num: int,
        episode_reward: float,
    ) -> None:
        """
        Save the current state of the agent to a file.

        :param checkpoint_path: path to checkpoint directory
        :param episode_num: The current episode number.
        :param episode_reward: The current episode reward.
        """
        checkpoint = {
            "actor_critic_state_dict": self.agent.actor_critic.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "episode": episode_num,
            "reward": episode_reward,
        }
        os.makedirs(checkpoint_path, exist_ok=True)
        fpath = os.path.join(checkpoint_path, f"checkpoint_{episode_num}.pth")
        torch.save(checkpoint, fpath)


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env_name: str = "CarRacing-v2"
    max_episode_steps = 600  # default
    num_episodes = 10

    env: gym.Env[Any, Any] = gym.make(
        env_name,
        domain_randomize=True,
        continuous=True,
        render_mode="human",
        max_episode_steps=max_episode_steps,
    )

    # We first check if state_shape is None. If it is None, we raise a ValueError.
    # Otherwise, we access the first element of state_shape using its index and
    # using the int() function.
    state_shape = env.observation_space.shape

    if state_shape is None:
        raise ValueError("Observation space shape is None.")
    state_dim = int(state_shape[0])
    state_channel = int(state_shape[2])

    # Get action spaces
    action_space = env.action_space

    if isinstance(action_space, gym.spaces.Box):
        action_high = action_space.high
        action_shape = action_space.shape
    else:
        raise ValueError("Action space is not of type Box.")
    if action_shape is None:
        raise ValueError("Action space shape is None.")

    action_dim = int(action_shape[0])
    max_action = int(action_high[0])

    # Convert from nupy to tensor
    low = torch.from_numpy(action_space.low).to(device)
    high = torch.from_numpy(action_space.high).to(device)

    # Actor-Critic hyperparameters
    gamma = 0.99
    lr = 0.0001
    value_coef = 0.5
    entropy_coef = 0.01
    hidden_dim = 256
    batch_size = 64

    # Location to store training agent model checkpoint
    checkpoint_dir = "model_checkpoints"

    # Initialize Actor-Critic network
    agent = ActorCriticAgent(
        state_dim=state_dim,
        state_channel=state_channel,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=hidden_dim,
        gamma=gamma,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
    )

    # Initialize the replay buffer
    # memory = ReplayBuffer(buffer_size=1024)
    memory = PrioritizedReplayBuffer(capacity=1024, alpha=0.99)

    # Create trainer to train agent
    trainer = Trainer(
        env=env,
        agent=agent,
        memory=memory,
        max_episodes=num_episodes,
        checkpoint_path=checkpoint_dir,
        batch_size=batch_size,
        low=low,
        high=high,
        device=device,
    )

    trainer.train()
    # add this line to close the environment after training
    env.close()  # type: ignore
