# Importing necessary libraries
from typing import Any
import numpy as np
import torch
from torch.distributions import Categorical, Normal

import gymnasium as gym
import matplotlib.pyplot as plt

import onnxruntime as ort
import onnx

from neural_networks.actor_network import Actor

# Setting the seed for reproducibility
torch.manual_seed(42)


if __name__ == "__main__":
    """CarRacing-v2 Gym environment"""
    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the environment
    env = gym.make(
        "CarRacing-v2",
        continuous=True,
        max_episode_steps=2000,
        render_mode="human",
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

    # Load the saved model or checkpoint
    model = onnx.load("model.onnx")

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

    # Create lists to store episode rewards and reward standard deviations
    episode_rewards = []
    reward_std_devs = []

    # Run a 10 episode rollout using the pre-trained model
    for episode in range(10):
        print(f"Episode: {episode}")
        state_ndarray, _ = env.reset(seed=42)
        done = False
        episode_cumulative_reward = 0.0
        rewards_in_episode = []

        # Convert next state to shape (batch_size, channe, width, height)
        state = (
            torch.tensor(state_ndarray, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
            .permute(0, 3, 1, 2)
        )
        step_count = 0
        while not done:
            print(f"Step: {step_count}")

            # Convert the tensor to a NumPy array
            state_array = state.cpu().detach().numpy()

            # Create a list with the input dictionary
            input_feed = {input_name: state_array}

            # Run the model inference with input data
            output = session.run(None, input_feed)

            # Process the output
            action_mean = torch.from_numpy(output[0]).to(device)
            action_std  =  torch.from_numpy(output[1]).to(device)

            # Sample an action from the distribution
            action_distribution = Normal(action_mean, action_std)
            action = action_distribution.sample()  # type: ignore

            # Rescale the action to the range of the action space
            rescaled_action = ((action + 1) / 2) * (high - low) + low

            # Clip the rescaledaction to ensure it falls within the bounds of the action space
            clipped_action = torch.clamp(rescaled_action, low, high)

            # Convert the action tensor to a numpy array
            action = clipped_action.squeeze().cpu().detach().numpy()

            # Take a step in the environment using the chosen action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Update the cumulative reward for the episode
            episode_cumulative_reward += int(reward)

            # Accumulate episode rewards (reward per step)
            rewards_in_episode.append(reward)

            # Print to terminal the reward received for action taken
            print(f"\t- Reward: {reward}")

            # Update the state for the next iteration
            state = (
                torch.tensor(next_state, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
                .permute(0, 3, 1, 2)
            )
            step_count += 1

            done = terminated or truncated

        # Append the episode cumulative reward to the list
        episode_rewards.append(int(episode_cumulative_reward))

        # Calculate the reward standard deviation for the episode
        reward_std_dev = torch.std(torch.tensor(rewards_in_episode))
        reward_std_devs.append(float(reward_std_dev.item()))

        print(f"Episode: {episode+1}, Total Reward: {episode_cumulative_reward}")

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
    plt.title("Rollout: Average Reward with Uncertainty Band")
    plt.legend()
    plt.show()

    # Save the plot to a file
    plt.savefig('rollout_average_reward_per_episode.png')

    # Close the environment
    env.close()
