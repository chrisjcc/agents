# Importing necessary libraries
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from configuration_manager import ConfigurationManager
from checkpoint_manager import CheckpointManager

from actor_critic_agent import ActorCriticAgent
from replay_buffer.per import PrioritizedReplayBuffer
from replay_buffer.uer import UniformExperienceReplay

from utils.beta_scheduler import BetaScheduler
from data_logger import DataLogger


# Setting the seed for reproducibility
torch.manual_seed(42)


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
        max_episode_steps: int,
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
        self.device = device
        self.checkpoint = checkpoint
        self.max_episode_steps = max_episode_steps
        total_steps = max_episode_steps * max_episodes

        self.beta_scheduler = BetaScheduler(initial_beta=0.0, total_steps=total_steps)

        self.data_logger = data_logger

    def env_step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns state, reward and terminated flag given an action.
        """
        # Convert action from torch.Tensor to np.ndarray
        action = action.squeeze().cpu().detach().numpy()

        # Take one step in the environment given the agent action
        state, reward, terminated, truncated, info = self.env.step(action)

        # Convert the numpy array to a PyTorch tensor with shape (batch_size, channel, width, height)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        terminated = torch.tensor([terminated], dtype=torch.float32).to(self.device)
        truncated = torch.tensor([truncated], dtype=torch.float32).to(self.device)

        # Flatten the state tensor to match the expected input dimension
        state = state.flatten()

        return state, reward, terminated, truncated

    def collect_experiences(self, buffer_size) -> None:
        """
        Run the training loop for the actor-critic agent.

        :param batch_size: The size of the batch of transitions
            of state, action, reward, next_state, and terminated to learn from.
        :return: A list of episode rewards.
        """
        print("Collect experiences and add them to the replay buffer...")

        # Check if the replay buffer has enough samples to start training
        while not self.memory.ready(capacity=buffer_size):
            # Fill the replay buffer before starting training
            state_ndarray, info = self.env.reset()

            # Convert the state to a PyTorch tensor with shape (batch_size, channel, width, height)
            state = torch.tensor(state_ndarray, dtype=torch.float32).to(self.device)

            # Flatten the state tensor to match the expected input dimension
            state = state.flatten()

            step_count = 0
            done = False

            #while step_count <= self.max_episode_steps:
            while not done:
                print(f"Step: {step_count}")

                # Create a random logits tensor
                logits = torch.randn(1, self.agent.action_dim)

                # Create a categorical distribution over the action values
                action_distribution = Categorical(
                    logits=logits
                )

                # Sample an action from the distribution
                action = action_distribution.sample()

                # Take a step in the environment with the chosen action
                next_state, reward, terminated, truncated = self.env_step(action)

                # Update if the environment is done
                done = terminated or truncated

                # Flatten the state tensor to match the expected input dimension
                next_state = next_state.flatten()

                # Collect experience trajectory in replay buffer
                self.memory.add(state, action, reward, next_state, done)

                # Update the current state
                state = next_state
                step_count += 1

    def train_step(self, beta: float=0.0):
        """
        Run a single training step in the OpenAI Gym environment.

        :param batch: The batch of transitions of state, action, reward, next_state, and done to learn from.
        :return: The next state, reward, and whether the episode is done.
        """
        # Sample a batch from the replay buffer
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
        action, action_distribution = self.agent.actor_critic.sample_action(
            state
        )

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = self.agent.actor_critic.sample_action(
            next_state
        )

        # Update the neural networks
        indices, td_errors = self.agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            terminated=terminated,
            action_distribution=action_distribution,
            next_action_distribution=next_action_distribution,
            indices=indices.to(self.device),
            weight=weight.to(self.device),
        )

        # Update prority scores of experiences
        self.memory.update_priorities(
            indices,
            td_errors,
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
            state_ndarray, info = self.env.reset()  # Reset the environment

            # Set variables
            episode_cumulative_reward = 0.0
            done = False
            self.step = 0

            #while self.step <= self.max_episode_steps:
            while not done:
                # Update model parameters using TD error
                beta = self.beta_scheduler.get_beta(self.step)
                self.train_step(beta)

                # Convert next state to shape (batch_size, channel, width, height)
                state = torch.tensor(state_ndarray, dtype=torch.float32).to(self.device)

                # Flatten the state tensor to match the expected input dimension
                state = state.flatten()

                # Pass the state through the Actor model to obtain a probability distribution over the actions
                action, action_probs = self.agent.actor_critic.sample_action(state)

                action = action.unsqueeze(0)

                # Take the action in the environment and observe the next state, reward, and done flag
                next_state, reward, terminated, truncated = self.env_step(
                    action
                )

                done = terminated or truncated
                self.memory.add(state, action, reward, next_state, done)

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
                        self.agent.state_dict(),
                        episode,
                        episode_cumulative_reward,
                    )

                state = next_state
                self.step += 1

            # Save the agent's model checkpoint at the end of each episode
            self.checkpoint.save_checkpoint(
                self.agent.state_dict(),
                episode,
                episode_cumulative_reward,
            )

            # Calculate and log episode average reward
            self.data_logger.log_scalar(
                tag="Episode Average Reward",
                value=episode_cumulative_reward/self.step,
                step_count=episode
            )

            # Calculate and log the episode total, actor, and critic loss after each episode
            self.data_logger.log_episode_average_total_loss()
            self.data_logger.log_actor_critic()
            self.data_logger.log_entropy()
            self.data_logger.log_gradient()

            # Update the episode number
            self.data_logger.update_episode_num()

            # Store the episode reward and reward standard deviation in the lists
            episode_rewards.append(episode_cumulative_reward)

            # Calculate the reward standard deviation for this episode
            reward_std_dev = torch.std(torch.cat(rewards_in_episode))
            reward_std_devs.append(float(reward_std_dev))

            # Reset the data_logger for the next episode
            self.data_logger.reset()

        # Export the model to ONNX format
        self.checkpoint.save_model(model=self.agent.actor_critic.actor, input=state)

        return episode_rewards, reward_std_devs



if __name__ == "__main__":
    """Highway-env-v0 Gym environment"""
    import gymnasium as gym
    import matplotlib.pyplot as plt

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration file
    config_manager = ConfigurationManager("highway_config.yaml")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env: gym.Env[Any, Any] = gym.make(
        config_manager.env_name,
        render_mode=config_manager.render_mode, # "human",
        max_episode_steps=config_manager.max_episode_steps,
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

    state_dim = (15, 5) #env.observation_space.shape

    # Get action spaces
    action_space = env.action_space
    action_dim = 1

    # Initialize Data logging
    data_logger = DataLogger()
    data_logger.initialize_writer()

    # Initialize the replay buffer
    memory = PrioritizedReplayBuffer(
        capacity=config_manager.replay_buffer_capacity,
        alpha=config_manager.replay_buffer_alpha,
    )

    # Initialize the Checkpoint object
    checkpoint = CheckpointManager(
        checkpoint_dir=config_manager.checkpoint_dir,
        checkpoint_freq=config_manager.checkpoint_freq,
        num_checkpoints=config_manager.num_checkpoints,
    )

    # Initialize Actor-Critic network
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config_manager.hidden_dim,
        gamma=config_manager.gamma,
        lr=config_manager.lr,
        value_coef=config_manager.value_coef,
        entropy_coef=config_manager.entropy_coef,
        device=device,
        data_logger=data_logger,
        lr_step_size=config_manager.max_episode_steps,
        lr_gamma=config_manager.lr_gamma,
    )

    # Create trainer to train agent
    trainer = Trainer(
        env=env,
        agent=agent,
        memory=memory,
        max_episodes=config_manager.num_episodes,
        max_episode_steps=config_manager.max_episode_steps,
        batch_size=config_manager.batch_size,
        device=device,
        checkpoint=checkpoint,
        data_logger=data_logger,
    )

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

    # Add this line to close the environment after training
    env.close()  # type: ignore
