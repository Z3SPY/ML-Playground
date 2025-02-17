import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

#############################################
#         DOUBLE DUELING DQN NETWORK        #
#############################################
class DuelingNetwork(nn.Module):
    """
    A Double Dueling DQN architecture:
      - 'Double DQN' is actually a method for the learn() step, not the architecture itself,
        but we typically just call it "Dueling DQN" for the network portion.
      - The network has two streams: one for state-value, one for advantages.
    """
    def __init__(self, state_size, action_size, seed=42):
        super(DuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared feature layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Dueling streams
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Split into value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine them to get Q-values
        avg_advantage = torch.mean(advantage, dim=1, keepdim=True)
        q_values = value + (advantage - avg_advantage)
        return q_values

#############################################
#             REPLAY MEMORY                 #
#############################################
class ReplayMemory:
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones

#############################################
#          SIMPLE DYNAMICS MODEL            #
#############################################
def dynamics_model(state, action):
    noise = np.random.normal(0, 0.05, size=state.shape)
    bias = (action - 1) * 0.02  # For discrete actions (0,1,2,3)
    new_state = state + noise + bias
    return new_state

#############################################
#  MULTI-STAGE STOCHASTIC OPTIMIZER (CEM)   #
#############################################
class MultiStageStochasticOptimizer:
    """
    Shortened horizon, fewer candidates, fewer iterations for faster updates:
      horizon=3, num_candidates=20, num_elite=5, num_iterations=3
    """
    def __init__(self, horizon, num_candidates, num_elite, num_iterations, discount_factor, dynamics_model):
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elite = num_elite
        self.num_iterations = num_iterations
        self.discount_factor = discount_factor
        self.dynamics_model = dynamics_model
        self.action_dim = None

    def optimize(self, state, agent):
        if self.action_dim is None:
            self.action_dim = agent.action_size

        # Candidate generation
        candidate_sequences = np.random.randint(low=0, high=self.action_dim,
                                                size=(self.num_candidates, self.horizon))
        
        # Iterative refinement
        for _ in range(self.num_iterations):
            cumulative_rewards = np.zeros(self.num_candidates)
            sim_states = np.repeat(state[np.newaxis, :], self.num_candidates, axis=0)
            discount = 1.0

            for t in range(self.horizon):
                actions_t = candidate_sequences[:, t]
                sim_states = self.vectorized_dynamics(sim_states, actions_t)
                sim_states_tensor = torch.from_numpy(sim_states).float().to(agent.device)
                with torch.no_grad():
                    q_vals = agent.local_qnetwork(sim_states_tensor)
                rewards = q_vals.max(dim=1)[0].cpu().numpy()
                cumulative_rewards += discount * rewards
                discount *= self.discount_factor

            # Elite selection
            elite_indices = np.argsort(cumulative_rewards)[-self.num_elite:]
            elite_sequences = candidate_sequences[elite_indices]

            # Update distribution
            new_candidate_sequences = np.zeros_like(candidate_sequences)
            for t in range(self.horizon):
                elite_actions_t = elite_sequences[:, t]
                counts = np.bincount(elite_actions_t, minlength=self.action_dim)
                prob = counts / counts.sum()
                new_candidate_sequences[:, t] = np.random.choice(self.action_dim,
                                                                 size=self.num_candidates,
                                                                 p=prob)
            candidate_sequences = new_candidate_sequences

        # Final evaluation
        cumulative_rewards = np.zeros(self.num_candidates)
        sim_states = np.repeat(state[np.newaxis, :], self.num_candidates, axis=0)
        discount = 1.0
        for t in range(self.horizon):
            actions_t = candidate_sequences[:, t]
            sim_states = self.vectorized_dynamics(sim_states, actions_t)
            sim_states_tensor = torch.from_numpy(sim_states).float().to(agent.device)
            with torch.no_grad():
                q_vals = agent.local_qnetwork(sim_states_tensor)
            rewards = q_vals.max(dim=1)[0].cpu().numpy()
            cumulative_rewards += discount * rewards
            discount *= self.discount_factor

        best_index = np.argmax(cumulative_rewards)
        best_sequence = candidate_sequences[best_index]
        return best_sequence[0]

    def vectorized_dynamics(self, states, actions):
        next_states = np.array([self.dynamics_model(state, action) for state, action in zip(states, actions)])
        return next_states

#############################################
#                 AGENT                     #
#############################################
class Agent:
    """
    - Uses a Double DQN approach in learn().
    - DuelingNetwork architecture for better state-value representation.
    - Multi-stage planner with shorter horizon & fewer candidates.
    """
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        
        # Dueling network
        self.local_qnetwork = DuelingNetwork(state_size, action_size).to(self.device)
        self.target_qnetwork = DuelingNetwork(state_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=5e-4)
        self.memory = ReplayMemory(int(1e5))
        self.t_step = 0
        
        # Multi-stage optimizer with a shorter horizon and fewer candidates
        self.multi_stage_optimizer = MultiStageStochasticOptimizer(
            horizon=3,
            num_candidates=20,
            num_elite=5,
            num_iterations=3,
            discount_factor=0.99,
            dynamics_model=dynamics_model
        )

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.memory) > 100:
            experiences = self.memory.sample(64)  # Example batch size: 64
            self.learn(experiences, 0.99)

    def act(self, state, epsilon=0.0, use_planner_prob=0.2):
        if random.random() < use_planner_prob:
            return self.multi_stage_optimizer.optimize(state, self)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.local_qnetwork.eval()
            with torch.no_grad():
                action_values = self.local_qnetwork(state_tensor)
            self.local_qnetwork.train()
            if random.random() > epsilon:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        """
        Double DQN approach:
          - Next actions are chosen by the local network.
          - Their values are evaluated by the target network.
        """
        states, next_states, actions, rewards, dones = experiences
        
        # 1) Choose best actions in next_states via local network
        best_actions_local = self.local_qnetwork(next_states).argmax(dim=1, keepdim=True)
        
        # 2) Evaluate those actions with the target network
        q_targets_next = self.target_qnetwork(next_states).gather(1, best_actions_local).detach()
        
        # 3) Construct the Q-targets
        q_targets = rewards + (discount_factor * q_targets_next * (1 - dones))
        
        # 4) Get expected Q-values from local network
        q_expected = self.local_qnetwork(states).gather(1, actions)
        
        # 5) Compute loss & update
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 6) Soft update target
        self.soft_update(self.local_qnetwork, self.target_qnetwork, 1e-3)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                interpolation_parameter * local_param.data +
                (1.0 - interpolation_parameter) * target_param.data
            )

#############################################
#         ENVIRONMENT SETUP (LUNAR LANDER)  #
#############################################
env = gym.make('LunarLander-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print('State size:', state_size)
print('Number of actions:', action_size)

#############################################
#       HYPERPARAMETERS & INITIALIZATION    #
#############################################
number_episodes = 2000
max_timesteps_per_episode = 1000

# Slower epsilon decay
epsilon_start = 1.0
epsilon_end = 0.02
epsilon_decay = 0.99
epsilon = epsilon_start

agent = Agent(state_size, action_size)
scores_window = deque(maxlen=100)

#############################################
#             TRAINING LOOP                 #
#############################################
for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    
    for t in range(max_timesteps_per_episode):
        # 20% chance to use the multi-stage planner
        action = agent.act(state, epsilon, use_planner_prob=0.2)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    
    scores_window.append(score)
    epsilon = max(epsilon_end, epsilon_decay * epsilon)
    
    print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}", end="")
    if episode % 100 == 0:
        print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}")
    
    if np.mean(scores_window) >= 200.0:
        print(f"\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

env.close()
torch.save(agent.local_qnetwork.state_dict(), "trained_model.pth")
print("\nTraining complete. Model saved as 'trained_model.pth'.")

#############################################
#         DEMONSTRATION & VIDEO             #
#############################################
from gymnasium.wrappers import RecordVideo
demo_env = gym.make('LunarLander-v3', render_mode="rgb_array")
demo_env = RecordVideo(demo_env, video_folder="videos", episode_trigger=lambda episode_id: True)
state, _ = demo_env.reset()
done = False

while not done:
    # Fully greedy for demonstration
    action = agent.act(state, epsilon=0.0, use_planner_prob=0.0)
    state, reward, done, truncated, info = demo_env.step(action)
    if done or truncated:
        break

demo_env.close()
print("Demonstration video recorded in the 'videos' folder.")
