# Importing necessary libraries
from typing import Any, List, Tuple

import numpy as np
import torch

from actor_critic_agent import ActorCriticAgent
from checkpoint_manager import CheckpointManager
from configuration_manager import ConfigurationManager
from data_logger import DataLogger
from replay_buffer.per import PrioritizedReplayBuffer
from replay_buffer.replay_buffer import ReplayBuffer
from replay_buffer.uer import UniformExperienceReplay
from utils.beta_scheduler import BetaScheduler

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
        memory: Any,
        max_episodes: int,
        low: Any,
        high: Any,
        device: Any,
        checkpoint: Any,
        data_logger: Any,
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
        self.memory = memory
        self.batch_size = batch_size
        self.low = low
        self.high = high
        self.device = device
        self.checkpoint = checkpoint
        max_episode_steps = 600  # default
        num_episodes = 10
        total_steps = max_episode_steps * num_episodes

        self.beta_scheduler = BetaScheduler(initial_beta=0.0, total_steps=total_steps)

        self.data_logger = data_logger

    def env_step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns state, reward and terminated flag given an action."""
        # Convert action from torch.Tensor to np.ndarray
        action = action.squeeze().cpu().detach().numpy()

        # Take one step in the environment given the agent action
        state, reward, terminated, truncated, info = self.env.step(action)

        # Convert to tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).view(-1, 1).to(self.device)

        terminated = torch.tensor(terminated, dtype=torch.float32).view(-1, 1).to(self.device)

        truncated = torch.tensor(truncated, dtype=torch.float32).view(-1, 1).to(self.device)

        return state, reward, terminated, truncated

    def collect_experiences(self, buffer_size) -> None:
        """
        Run the training loop for the actor-critic agent.

        :param batch_size: The size of the batch of transitions
            of state, action, reward, next_state, and terminated to learn from.
        :return: A list of episode rewards.
        """
        print("Collect experiences and add them to the replay buffer...")

        # Fill the replay buffer before starting training
        state_ndarray, info = self.env.reset()

        # Convert the state to a PyTorch tensor with shape (batch_size, channel, width, height)
        state = (
            torch.tensor(state_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
            .permute(0, 3, 1, 2)
        )

        # Check if the replay buffer has enough samples to start training
        while not self.memory.ready(capacity=buffer_size):
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

    def train_step(self, beta: float = 0.0, step: int = 0):
        """
        Run a single training step in the OpenAI Gym environment.

        :param batch: The batch of transitions of state, action, reward, next_state, and done to learn from.
        :return: The next state, reward, and whether the episode is done.
        """
        # Sample a batch from the replay buffer
        # state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        (
            state,
            action,
            reward,
            next_state,
            terminated,
            indices,
            weight,
        ) = self.memory.sample(
            self.batch_size,
            beta,
        )

        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        # with torch.no_grad():
        _, action_distribution = self.agent.actor_critic.sample_action(state.squeeze(1))

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
        terminated = torch.squeeze(terminated, dim=1)
        indices = torch.squeeze(indices, dim=1)
        weight = torch.squeeze(weight, dim=1)

        # Update the neural networks
        indices, td_errors = self.agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            indices=indices.to(self.device),
            weight=weight.to(self.device),
            step=step,
        )

        # Update prority scores of experiences
        self.memory.update_priorities(
            indices.cpu().detach().numpy(), td_errors.cpu().detach().numpy()
        )

    def train(self) -> List[float]:
        """
        Run the training loop for the actor-critic agent.

        :param batch_size: The size of the batch of transitions
            of state, action, reward, next_state, and terminated to learn from.
        :return: A list of episode rewards.
        """
        # Collect experiences and add them to the replay buffer
        buffer_size = self.memory.buffer.maxlen
        self.collect_experiences(buffer_size=buffer_size)

        # Create lists to store episode rewards and reward standard deviations
        episode_rewards = []
        reward_std_devs = []

        # Training loop
        for episode in range(self.max_episodes):
            print(f"Episode: {episode}")

            # Initialize lists to store rewards for this episode
            rewards_in_episode = []

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
            episode_cumulative_reward = 0.0
            done = False
            self.step = 0

            while not done:
                # Update model parameters using TD error
                beta = self.beta_scheduler.get_beta(self.step)
                self.train_step(beta, self.step)

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

                # Accumulate reward for the current episode
                episode_cumulative_reward += float(reward.item())

                # Accumulate episode rewards (reward per step)
                rewards_in_episode.append(reward)

                # Increment the step attribute
                self.data_logger.increment_step()

                print(
                    f"Episode {episode}: Step {self.step}: Total reward = {episode_cumulative_reward:.2f}"
                )

                # Save model checkpoint after each checkpoint_freq episode
                if episode % self.checkpoint.checkpoint_freq == 0 and episode > 0:
                    self.checkpoint.save_checkpoint(
                        self.agent.actor_critic.state_dict(),
                        self.agent.optimizer.state_dict(),
                        episode,
                        episode_cumulative_reward,
                    )

                state = next_state
                self.step += 1

            # Save the agent's model checkpoint at the end of each episode
            self.checkpoint.save_checkpoint(
                self.agent.actor_critic.state_dict(),
                self.agent.optimizer.state_dict(),
                episode,
                episode_cumulative_reward,
            )

            # Calculate and log episode average reward
            self.data_logger.log_scalar(
                tag="Episode Average Reward",
                value=episode_cumulative_reward / self.step,
                step_count=episode,
            )

            # Calculate and log the episode total, actor, and critic loss after each episode
            self.data_logger.log_episode_average_total_loss()
            self.data_logger.log_actor_critic()
            self.data_logger.log_entropy()

            # Update the episode number
            self.data_logger.update_episode_num()

            # Store the episode reward and reward standard deviation in the lists
            episode_rewards.append(episode_cumulative_reward)

            # Calculate the reward standard deviation for this episode
            reward_std_dev = torch.std(torch.cat(rewards_in_episode))
            reward_std_devs.append(float(reward_std_dev))

            # Reset the data_logger for the next episode
            self.data_logger.reset()

        # After collecting experiences for a few episodes, clear the priorities
        self.memory.clear_priorities()

        return episode_rewards, reward_std_devs


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    import gymnasium as gym
    import matplotlib.pyplot as plt

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration file
    config = ConfigurationManager("train_config.yaml")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env: gym.Env[Any, Any] = gym.make(
        config.env_name,
        domain_randomize=config.domain_randomize,  # True,
        continuous=config.continuous,  # True,
        render_mode=config.render_mode,  # "human",
        max_episode_steps=config.max_episode_steps,
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

    # Initialize Data logging
    data_logger = DataLogger()
    data_logger.initialize_writer()

    # Initialize the replay buffer
    # memory = ReplayBuffer(buffer_size=1024)
    memory = PrioritizedReplayBuffer(capacity=1024, alpha=0.99)

    # Initialize the Checkpoint object
    checkpoint = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_freq=config.checkpoint_freq,
        num_checkpoints=config.num_checkpoints,
    )

    # Initialize Actor-Critic network
    agent = ActorCriticAgent(
        state_dim=state_dim,
        state_channel=state_channel,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=config.hidden_dim,
        gamma=config.gamma,
        lr=config.lr,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        device=device,
        data_logger=data_logger,
        lr_step_size=config.max_episode_steps,
    )

    # Create trainer to train agent
    trainer = Trainer(
        env=env,
        agent=agent,
        memory=memory,
        max_episodes=config.num_episodes,
        batch_size=config.batch_size,
        low=low,
        high=high,
        device=device,
        checkpoint=checkpoint,
        data_logger=data_logger,
    )

    # trainer.train()
    # add this line to close the environment after training
    # env.close()  # type: ignore

    # Train the agent and get the episode rewards and reward standard deviations
    episode_rewards, reward_std_devs = trainer.train()

    # Plot the average reward with an uncertainty band (standard deviation)
    plt.plot(episode_rewards, label="Average Reward")
    plt.fill_between(
        range(len(episode_rewards)),
        [reward - std_dev for reward, std_dev in zip(episode_rewards, reward_std_devs)],
        [reward + std_dev for reward, std_dev in zip(episode_rewards, reward_std_devs)],
        alpha=0.3,
        label="Uncertainty Band",
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward with Uncertainty Band")
    plt.legend()
    plt.show()
