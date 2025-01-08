import torch
import gymnasium as gym
import imageio

# Assuming you have an Agent class or model defined in pacman.py
from pacman import Agent  # Replace with your actual model/agent class

# Environment setup
env_name = 'ALE/MsPacman-v5'  # Gymnasium-compatible Atari environment
env = gym.make(env_name, render_mode='rgb_array')

# Load the saved model (checkpoint.ph)
agent = Agent(action_size=env.action_space.n)  # Ensure this matches the saved model
agent.local_qnetwork.load_state_dict(torch.load('DCQL\checkpoint.ph'))  # Load weights
agent.local_qnetwork.eval()  # Set to evaluation mode

# Function to record gameplay
def show_video_of_model(agent, env):
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()  # Get the current frame
        frames.append(frame)  # Save the frame

        # Use the trained model to get an action
        action = agent.act(state, epsilon=0.0)  # Disable exploration
        state, reward, done, _, _ = env.step(action)

    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)  # Save video

# Call the function to record video
show_video_of_model(agent, env)
print("Gameplay video saved as 'video.mp4'")
