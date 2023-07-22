# Importing necessary libraries
import os
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        saved_model_dir: str = "saved_model",
        checkpoint_freq: int = 1,
        num_checkpoints: int = 5,
    ) -> None:
        """
        Initialize the CheckpointManager.

        :param checkpoint_dir (str): Directory to save checkpoints.
        :param saved_model_dir (str): Directory to save the final saved model.
        :param checkpoint_freq (int): Frequency of saving checkpoints.
        :param num_checkpoints (int): Maximum number of checkpoints to keep.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.num_checkpoints = num_checkpoints
        self.episode_last_checkpoint: Dict[int, str] = {}
        self.saved_model_dir = saved_model_dir

        # Create directories for checkpoint and model saving
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.saved_model_dir, exist_ok=True)

    def save_checkpoint(self, state_dict: Dict[str, Any], episode_num: int, episode_reward: float) -> None:
        """
        Save the current state of the agent to a file.

        :param state_dict (Dict[str, Any]): Dictionary containing the state of the agent.
        :param episode_num (int): The current episode number.
        :param episode_reward (float): The current episode reward.

        Returns:
            None
        """
        state_dict["episode"] = episode_num
        state_dict["reward"] = episode_reward

        last_checkpoint = self.episode_last_checkpoint.get(episode_num, None)

        if last_checkpoint and os.path.exists(last_checkpoint):
            os.remove(last_checkpoint)
        fpath = os.path.join(self.checkpoint_dir, f"checkpoint_{episode_num}.pth")
        torch.save(state_dict, fpath)
        self.episode_last_checkpoint[episode_num] = fpath

        # Remove the oldest checkpoint if we have reached the maximum number of checkpoints
        if len(self.episode_last_checkpoint) > self.num_checkpoints:
            oldest_episode = min(self.episode_last_checkpoint.keys())
            oldest_checkpoint = self.episode_last_checkpoint.pop(oldest_episode)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """
        Load the state of the agent from a file.

        :param checkpoint_path (str): Path to the checkpoint file.

        Returns:
            Tuple[int, float]: The episode number and episode reward.
        """
        checkpoint = torch.load(checkpoint_path)
        self.agent.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["episode"], checkpoint["reward"]

    def save_model(self, model: nn.Module, input) -> None:
        """
        Save the final model in ONNX format.

        :param model (nn.Module): The final model to be saved.
        """
        model_path = os.path.join(self.saved_model_dir, "model.onnx")

        torch.onnx.export(
            model,                      # model being run
            input,                      # model input (or a tuple for multiple inputs)
            model_path,                 # where to save the model (can be a file or file-like object)
            export_params=True,         # store the trained parameter weights inside the model file
            input_names=['state'],      # the model's input names
            output_names=['mean_action', 'mean_std'],  # The model's output names
        )
