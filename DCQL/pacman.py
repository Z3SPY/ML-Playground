#MsPacmanDeterministic-v0

# Installing Gymnasium

# !pip install gymnasium
# !pip install "gymnasium[atari, accept-rom-license]"
# !conda install anaconda::swig
# !pip install gymnasium[box2d]

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset


# BUILDING THE AI
# creating the architecture

#Inherits from library nn
class Network(nn.Module):

    def __init__(self, action_size, seed = 42):
        super(Network, self).__init__() #Calls the init of nn.Module    
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4) #First convolution
        self.bn1 = nn.BatchNorm2d(32) #Batch Norm, takes the number of features
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2) 
        self.bn2 = nn.BatchNorm2d(64) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1) 
        self.bn3 = nn.BatchNorm2d(64) 
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1) 
        self.bn4 = nn.BatchNorm2d(128) 

        # Full Connections
        # The input should be the number of output features that results after FLATTENING all the previous convolutions
        self.fc1 = nn.Linear(10*10*128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)


    def forward(self, state):
      # From pacman images -> eyes -> to fully conncected -> to actions       
      x = F.relu(self.bn1(self.conv1(state)))
      x = F.relu(self.bn2(self.conv2(x)))
      x = F.relu(self.bn3(self.conv3(x)))
      x = F.relu(self.bn4(self.conv4(x)))
      x = x.view(x.size(0), -1)  # Reshapes the tensor, the first dimension remains the same while the other dimensions are flattend
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      return self.fc3(x)
        


# TRAINING THE AI

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("MsPacmanDeterministic-v0", full_action_space = False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)


#Initializing the hyperparameters
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99


#Preprocessing the frame
from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
    frame = Image.fromarray(frame) #Converts into numpy array. Also converts into a PIL Image

    #What this is doing is its transforming the frame into a 128 x 128 and converting it into a tensor array
    preprocess = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()]) # look at the VALUES in state_shape variable for Compose([])
    return preprocess(frame).unsqueeze(0)


# Implementing the DCQN class

class Agent():

  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.local_qnetwork = Network(action_size).to(self.device)
    self.target_qnetwork = Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = deque(maxlen = 10000)

  def step(self, state, action, reward, next_state, done):
    state = preprocess_frame(state)
    next_state = preprocess_frame(next_state)
    self.memory.append((state, action, reward, next_state, done))
    if len(self.memory) > minibatch_size:
      experiences = random.sample(self.memory, k = minibatch_size)
      self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    state = preprocess_frame(state).to(self.device) #Remember, we do preprocess frames because we need to convert the images into readable vectors
    self.local_qnetwork.eval()
    with torch.no_grad():
      action_values = self.local_qnetwork(state)
    self.local_qnetwork.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, actions, rewards, next_states, dones = zip(*experiences)
    states = torch.from_numpy(np.vstack(states)).float().to(self.device)
    actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


# Initialize the DCQN agent
    

def CallAgentFunction():
  agent = Agent(number_actions) # Define agent here

  print(f"Using device: {agent.device}")


  # Lets train the CNN implementation
  number_episodes = 2000
  maximum_number_timesteps_per_episode = 1000
  epsilon_starting_value = 1.0
  epsilon_ending_value = 0.01
  epsilon_decay_value = 0.995
  epsilon = epsilon_starting_value
  scores_on_100_episodes = deque(maxlen=100)

  for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
      action = agent.act(state, epsilon)
      next_state, reward, done, _, _ = env.step(action)
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
        break

    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    print('\rEpisode {}\tAverage S core: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
    if episode % 100 == 0:
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    if np.mean(scores_on_100_episodes) >= 200.0:
      print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
      torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.ph')
      break

#Uncomment for training
#CallAgentFunction()

# VISUALIZING
