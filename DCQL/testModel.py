import torch
import gymnasium as gym
import imageio
import cv2
import random
import numpy as np
from pacman import Agent  # Assuming you have saved the Agent class in agent_module.py

# Create an environment
env_name = 'MsPacmanDeterministic-v0'
env = gym.make(env_name, render_mode='rgb_array')

# Load the trained agent
agent = Agent(action_size=env.action_space.n)  # Initialize agent with the number of actions
agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth'))  # Load the saved weights
agent.local_qnetwork.eval()  # Set the model to evaluation mode

# Function to record the agent's gameplay and save as a video
def show_video_of_model(agent, env_name):
    state, _ = env.reset()
    done = False
    frames = []
    
    while not done:
        frame = env.render()  # Capture the frame
        frames.append(frame)  # Store the frame in the list
        
        action = agent.act(state, epsilon=0.0)  # Use learned policy (no exploration)
        state, reward, done, _, _ = env.step(action)
    
    env.close()  # Close the environment after use
    
    # Save the frames as a video
    imageio.mimsave('video.mp4', frames, fps=30)

# Function to display the saved video locally using OpenCV
def display_video_locally():
    cap = cv2.VideoCapture('video.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Agent Gameplay', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Record and display the video
show_video_of_model(agent, env_name)
display_video_locally()
