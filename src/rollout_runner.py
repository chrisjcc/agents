# Importing necessary libraries
import datetime
import json
import shutil

from typing import Any
import numpy as np
import torch
from torch.distributions import Categorical, Normal

import gymnasium as gym
import matplotlib.pyplot as plt

import onnxruntime as ort
import onnx

from neural_networks.actor_network import Actor
from configuration_manager import ConfigurationManager

# Set the seed for reproducibility
torch.manual_seed(42)


def evaluate_agent(env, n_eval_episodes, model, session, device):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The agent
    """
    # Prepare the input tensor
    input_name = session.get_inputs()[0].name

    episode_rewards = []
    episode_rewards_std = []

    for episode in range(n_eval_episodes):
        state, _ = env.reset()

        step = 0
        done = False
        total_rewards_ep = 0

        while not done:
            # Convert the state to a PyTorch tensor with shape (batch_size, channel, width, height)
            state = torch.tensor(state, dtype=torch.float32).to(device)

            # Flatten the state tensor to match the expected input dimension
            state = state.flatten()

            # Convert the tensor to a NumPy array
            state_array = state.cpu().detach().numpy()

            # Create a list with the input dictionary
            input_feed = {input_name: state_array}

            # Run the model inference with input data
            action_logits = session.run(None, input_feed)

            # Assuming action_logits is a list of NumPy arrays
            action_logits = np.array(action_logits)

            # Convert the state to a PyTorch tensor with shape (batch_size, channel, width, height)
            action_logits = torch.tensor(action_logits, dtype=torch.float32).to(device)

            # Create a categorical distribution from which to sample an action
            action_distribution = Categorical(logits=action_logits)
            action = action_distribution.sample().unsqueeze(0)  # type: ignore

            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())

            total_rewards_ep += reward

            done = terminated or truncated

            state = next_state

        # Append the episode cumulative reward to the list
        episode_rewards.append(total_rewards_ep)

        # Calculate the reward standard deviation for the episode
        rewards_std = torch.std(torch.tensor(episode_rewards))
        episode_rewards_std.append(float(rewards_std.item()))

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": n_eval_episodes,
            "eval_datetime": eval_form_datetime,
    }

    # Write a JSON file
    with open("rollout_results.json", "w") as outfile:
        json.dump(evaluate_data, outfile, indent=4)


    return episode_rewards, episode_rewards_std


if __name__ == "__main__":
    """highway-fast-v0 Gym environment"""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_manager = ConfigurationManager("highway_config.yaml")

    # Define the environment
    # Passing continuous=True converts the environment to use continuous action.
    # The continuous action space has 3 actions: [steering, gas, brake].
    env: gym.Env[Any, Any] = gym.make(
        config_manager.env_name,
        render_mode=config_manager.render_mode, #"human",
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

    state_dim = (15, 5) #int(state_shape[0])

    # Get action spaces
    action_space = env.action_space

    action_dim = int(env.action_space.n)

    # Load the saved model or checkpoint
    model = onnx.load("saved_model/model.onnx")

    # Check the model
    try:
        onnx.checker.check_model(model) # will fail if given >2GB model
        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("The model is valid!")

    # Create an inference session
    session = ort.InferenceSession(model.SerializeToString())

    # Prepare the input tensor
    input_name = session.get_inputs()[0].name

    # Evaluate the model and build JSON
    episode_rewards, episode_rewards_std = evaluate_agent(env, config_manager.num_episodes, model, session, device)


    # Plot the average reward with an uncertainty band (standard deviation)
    plt.plot(episode_rewards, label="Average Reward")
    plt.fill_between(
        range(len(episode_rewards)),
        [reward - std_dev for reward, std_dev in zip(episode_rewards, episode_rewards_std)],
        [reward + std_dev for reward, std_dev in zip(episode_rewards, episode_rewards_std)],
        alpha=0.3,
        label="Uncertainty Band",
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Rollout: Average Reward with Uncertainty Band")
    plt.legend()
    plt.show()

    # Save the plot to a file
    plt.savefig('rollout_average_reward_per_episode.png')

    # Close the environment
    env.close()
