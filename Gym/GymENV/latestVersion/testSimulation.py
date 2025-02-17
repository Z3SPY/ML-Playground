# test_simulation.py
from Environment import WardEnv
from DRLhospital import DQNAgent, action_from_index  # Ensure you import your agent and any needed functions.
from utils import flatten_state

# Define ward configuration (this should be aligned with your thesis and real data if available)
ward_configs = {
    'general': {
        'num_beds': 40,
        'base_staff': {
            'day': {'nurses': 10, 'doctors': 5},
            'night': {'nurses': 6, 'doctors': 3}
        },
        'sim_duration': 168,          # One week of timesteps
        'arrival_lambda': 0.8,          # Mean arrival rate (calibrated from data)
        'nurse_efficiency': 1.5,
        'treatment_params': {'shape': 3, 'scale': 20}  # Gamma distribution parameters
    }
}

# Instantiate environment
env = WardEnv(ward_name='general', **ward_configs['general'])

# Instantiate the DQN agent
# Note: For testing, you might even use a dummy agent that returns random actions.
initial_state = flatten_state(env.get_state())
state_size = initial_state.shape[0]  # Should be 11 based on flatten_state
action_size = 18  # 9 staff adjustments x 2 admission priorities
agent = DQNAgent(state_size, action_size, seed=0)

# Run a test episode with epsilon set to 0 for deterministic behavior
print("\nRunning test episode...\n")
episode_log = env.test_episode(agent, max_timesteps=env.sim_duration, epsilon=0.0)

# Optionally, analyze the log further (e.g., plot cumulative rewards or waiting times)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(episode_log['timestep'], episode_log['cumulative_reward'], marker='o')
plt.title("Cumulative Reward Over Timesteps in Test Episode")
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.show()
