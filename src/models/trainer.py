# Importing necessary libraries
import os
from typing import Any, Tuple

import torch
from actor_critic_agent import ActorCriticAgent

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
        max_episodes: int,
        low: Any,
        high: Any,
        device: Any,
        checkpoint_path: str = "model_checkpoints",
        checkpoint_freq: int = 100,
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
        self.low = low
        self.high = high
        self.device = device

    def env_step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns state, reward and done flag given an action."""
        # Convert action from torch.Tensor to np.ndarray
        action = action.squeeze().cpu().detach().numpy()

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

    def train_step(
        self,
        state: torch.Tensor,
    ) -> Tuple[Any, Any]:
        """
        Run a single training step in the OpenAI Gym environment.

        :param state: The current state of the environment.
        :return: Whether the episode is done and the reward obtained.
        """
        # Use `with torch.no_grad():` to disable gradient calculations when performing inference.
        with torch.no_grad():
            # Obtain mean and std action given state
            action, action_distribution = self.agent.actor_critic.sample_action(state)

            # Rescale, then clip the action to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(
                (((action + 1) / 2) * (self.high - self.low) + self.low),
                self.low,
                self.high,
            )

        # Take one step in the environment given the agent action
        next_state, reward, terminated, truncated = self.env_step(clipped_action)

        # Obtain mean and std of next action given next state
        next_action, next_action_distribution = self.agent.actor_critic.sample_action(
            next_state
        )

        # Rescale, then clip the action to ensure it falls within the bounds of the action space
        clipped_next_action = torch.clamp(
            (((next_action + 1) / 2) * (self.high - self.low) + self.low),
            self.low,
            self.high,
        )

        self.agent.update(
            state,
            clipped_action,
            reward,
            next_state,
            clipped_next_action,
            terminated,
            action_distribution,
            next_action_distribution,
        )

        done = terminated or truncated

        return done, reward

    def train(self, batch_size: int) -> None:
        """
        Run the training loop for the actor-critic agent.

        :param batch_size: The size of the batch of transitions
            of state, action, reward, next_state, and done to learn from.
        :return: A list of episode rewards.
        """
        for episode in range(self.max_episodes):
            # Get state spaces
            state_ndarray, info = env.reset()
            state = (
                torch.tensor(state_ndarray, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            episode_reward = 0.0
            done = False
            step = 0

            while not done:
                # Update model parameters using TD error
                done, reward = self.train_step(state)

                # Update episode reward
                episode_reward += float(reward)
                print(
                    f"Episode {episode}: Step {step}: Total reward = {episode_reward:.2f}"
                )

                # Save model checkpoint after each 10 episode
                if episode % self.checkpoint_freq == 0:
                    self.save_checkpoint(self.checkpoint_path, episode, episode_reward)

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
    num_episodes = 3

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
    low = torch.from_numpy(action_space.low)
    high = torch.from_numpy(action_space.high)

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
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=hidden_dim,
        gamma=gamma,
        lr=lr,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
    )

    # Create trainer to train agent
    trainer = Trainer(
        env=env,
        agent=agent,
        max_episodes=num_episodes,
        checkpoint_path=checkpoint_dir,
        low=low,
        high=high,
        device=device,
    )

    trainer.train(batch_size=batch_size)
    # add this line to close the environment after training
    env.close()  # type: ignore
