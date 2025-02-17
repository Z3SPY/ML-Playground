import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time
from Environment import WardEnv

def flatten_state(state):
    t = (state['time'] % 24) / 24.0
    occ = state['occupied_ratio']
    free = state['free_beds'] / 50.0
    wait = state['waiting_patients'] / 50.0
    nurses = state['staff_nurses']
    doctors = state['staff_doctors']
    pa_mean = state['predicted_arrivals_mean']
    pa_std = state['predicted_arrivals_std']
    shift = 1.0 if state['shift'] == 'day' else 0.0
    workload = state['workload']
    fatigue = state['nurse_fatigue']
    return np.array([t, occ, free, wait, nurses, doctors, pa_mean, pa_std, shift, workload, fatigue], dtype=np.float32)

staff_adjustments = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1)
]
admission_priorities = ['severity', 'fifo']

def action_from_index(index):
    staff_idx = index // 2
    priority_idx = index % 2
    return {
        'staff_adjustment': {
            'nurses': staff_adjustments[staff_idx][0],
            'doctors': staff_adjustments[staff_idx][1]
        },
        'admission_priority': admission_priorities[priority_idx]
    }

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.from_numpy(np.vstack(states)).float(),
            torch.from_numpy(np.array(actions)).long(),
            torch.from_numpy(np.vstack(rewards)).float(),
            torch.from_numpy(np.vstack(next_states)).float(),
            torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()
        )
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, seed=0, lr=1e-4, buffer_size=10000, batch_size=64,
                 gamma=0.99, tau=1e-3, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        random.seed(seed)
        self.seed = torch.manual_seed(seed)
        
        self.qnetwork_local = QNetwork(state_size, action_size).to("cpu")
        self.qnetwork_target = QNetwork(state_size, action_size).to("cpu")
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0

    def act(self, state, epsilon=0.0):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        if random.random() > epsilon:
            action_index = np.argmax(action_values.cpu().data.numpy())
        else:
            action_index = random.choice(np.arange(self.action_size))
        return action_from_index(action_index)

    def step(self, state, action, reward, next_state, done):
        action_index = None
        for idx in range(self.action_size):
            if action_from_index(idx) == action:
                action_index = idx
                break
        if action_index is None:
            action_index = 0
        self.memory.add(state, action_index, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions = actions.unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

if __name__ == "__main__":
    ward_configs = {
        'general': {
            'num_beds': 40,
            'base_staff': {
                'day': {'nurses': 10, 'doctors': 5},
                'night': {'nurses': 6, 'doctors': 3}
            },
            'sim_duration': 168,
            'arrival_lambda': 0.8,
            'nurse_efficiency': 1.5,
            'treatment_params': {'shape': 3, 'scale': 20}
        }
    }
    
    env = WardEnv(ward_name='general', **ward_configs['general'])
    initial_state = flatten_state(env.get_state())
    state_size = initial_state.shape[0]  # 11 features
    action_size = 18  # 9 staff adjustments x 2 priorities
    agent = DQNAgent(state_size, action_size, seed=0)
    
    num_episodes = 2000
    max_steps = env.sim_duration
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    scores_window = deque(maxlen=100)
    all_scores = []
    all_wait_times = []  # Store average wait times per episode

    for episode in range(1, num_episodes + 1):
        env.reset()
        state = flatten_state(env.get_state())
        score = 0
        for t in range(max_steps):
            action = agent.act(state, epsilon)
            next_state_dict, reward, done = env.step(action)
            next_state = flatten_state(next_state_dict)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        all_scores.append(score)
        scores_window.append(score)
        avg_wait_time = np.mean(env.wait_time_log) if env.wait_time_log else 0
        all_wait_times.append(avg_wait_time)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            avg_wait = np.mean(all_wait_times[-100:])
            print(f"Episode {episode} | Avg Score: {avg_score:.2f} | Avg Wait Time: {avg_wait:.2f}")
    
    print("Training finished.")
