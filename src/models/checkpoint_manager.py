import os

import torch


class CheckpointManager:
    def __init__(
        self, checkpoint_dir="checkpoints", checkpoint_freq=1, num_checkpoints=5
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.num_checkpoints = num_checkpoints
        self.episode_last_checkpoint = {}

    def save_checkpoint(
        self, actor_critic_state, optimizer_state, episode_num, episode_reward
    ):
        """
        Save the current state of the agent to a file.

        :param episode_num: The current episode number.
        :param episode_reward: The current episode reward.
        """
        checkpoint = {
            "actor_critic_state_dict": actor_critic_state,
            "optimizer_state_dict": optimizer_state,
            "episode": episode_num,
            "reward": episode_reward,
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        last_checkpoint = self.episode_last_checkpoint.get(episode_num, None)
        if last_checkpoint and os.path.exists(last_checkpoint):
            os.remove(last_checkpoint)
        fpath = os.path.join(self.checkpoint_dir, f"checkpoint_{episode_num}.pth")
        torch.save(checkpoint, fpath)
        self.episode_last_checkpoint[episode_num] = fpath

        # Remove the oldest checkpoint if we have reached the maximum number of checkpoints
        if len(self.episode_last_checkpoint) > self.num_checkpoints:
            oldest_episode = min(self.episode_last_checkpoint.keys())
            oldest_checkpoint = self.episode_last_checkpoint.pop(oldest_episode)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)

    def load_checkpoint(self, checkpoint_path):
        """
        Load the state of the agent from a file.

        :param checkpoint_path: path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.agent.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["episode"], checkpoint["reward"]
