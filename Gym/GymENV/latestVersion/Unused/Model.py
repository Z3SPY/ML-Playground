import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class WardEnv:
    def __init__(self, ward_name, num_beds, base_staff, sim_duration, arrival_lambda, nurse_efficiency, treatment_params):
        """
        Environment for a single ward.
        
        Parameters:
          ward_name: Identifier for the ward (e.g., 'general', 'ICU')
          num_beds: Number of beds in the ward.
          base_staff: Dictionary with keys 'day' and 'night', each mapping to staff levels, e.g.,
                      {'day': {'nurses': 10, 'doctors': 5}, 'night': {'nurses': 6, 'doctors': 3}}
          sim_duration: Timesteps for this ward simulation.
          arrival_lambda: Average patient arrivals per hour.
          nurse_efficiency: Number of patients a nurse can process per hour.
          treatment_params: Dictionary for Gamma distribution parameters for treatment duration.
        """
        self.ward_name = ward_name
        self.num_beds = num_beds
        self.base_staff = base_staff.copy()
        self.sim_duration = sim_duration
        self.arrival_lambda = arrival_lambda
        self.nurse_efficiency = nurse_efficiency
        self.treatment_params = treatment_params
        self.time = 0
        self.last_shift = self.get_shift_info()
        self.current_staff = self.base_staff[self.last_shift].copy()
        self.reset_metrics()
        self.reset()
    
    def reset_metrics(self):
        self.occupancy_log = []
        self.queue_log = []
        self.wait_time_log = []
        self.staff_log = []
        self.throughput = 0
        self.event_log = []
    
    def reset(self):
        self.time = 0
        self.beds = [None] * self.num_beds
        self.waiting_patients = deque()
        self.last_shift = self.get_shift_info()
        self.current_staff = self.base_staff[self.last_shift].copy()
        self.reset_metrics()
        return self.get_state()
    
    def get_state(self):
        occupancy = sum(1 for bed in self.beds if bed is not None)
        state = {
            'time': self.time,
            'occupied_beds': occupancy,
            'free_beds': self.num_beds - occupancy,
            'waiting_patients': len(self.waiting_patients),
            'staff_available': self.current_staff.copy(),
            'predicted_arrivals': self.forecast_arrivals(),
            'shift': self.get_shift_info()
        }
        return state
    
    def get_shift_info(self):
        return 'day' if 6 <= self.time % 24 < 18 else 'night'
    
    def update_shift(self):
        current_shift = self.get_shift_info()
        if current_shift != self.last_shift:
            self.current_staff = self.base_staff[current_shift].copy()
            self.last_shift = current_shift
    
    def forecast_arrivals(self):
        predicted_mean = np.random.poisson(lam=self.arrival_lambda)
        predicted_std = np.sqrt(predicted_mean)
        return {'mean': predicted_mean, 'std': predicted_std}
    
    def simulate_arrivals(self):
        arrivals = np.random.poisson(lam=self.arrival_lambda)
        for _ in range(arrivals):
            patient = {
                'arrival_time': self.time,
                'severity': random.randint(1, 5),
                'triage': random.choice(['elective', 'emergency']),
                'resource_cost': random.uniform(0.1, 1.0)
            }
            self.waiting_patients.append(patient)
        self.queue_log.append(len(self.waiting_patients))
    
    def admit_patients(self, priority_rule='severity'):
        if priority_rule == 'severity':
            sorted_patients = sorted(list(self.waiting_patients), key=lambda p: p['severity'], reverse=True)
        else:
            sorted_patients = list(self.waiting_patients)
        
        free_bed_indices = [i for i, bed in enumerate(self.beds) if bed is None]
        effective_capacity = self.current_staff.get('nurses', 0) * self.nurse_efficiency
        possible_admissions = int(min(len(sorted_patients), len(free_bed_indices), effective_capacity))
        
        admitted_wait_times = []
        for _ in range(possible_admissions):
            bed_idx = free_bed_indices.pop(0)
            if priority_rule == 'fifo':
                patient = self.waiting_patients.popleft()
            else:
                patient = sorted_patients.pop(0)
                self.waiting_patients.remove(patient)
            wait_time = self.time - patient['arrival_time']
            admitted_wait_times.append(wait_time)
            
            treatment_duration = np.random.gamma(shape=self.treatment_params['shape'], scale=self.treatment_params['scale'])
            expected_discharge = self.time + treatment_duration
            
            patient_record = {
                'arrival_time': patient['arrival_time'],
                'severity': patient['severity'],
                'triage': patient['triage'],
                'resource_cost': patient['resource_cost'],
                'expected_discharge': expected_discharge
            }
            self.beds[bed_idx] = patient_record
            self.throughput += 1
        
        return admitted_wait_times
    
    def simulate_discharges(self):
        discharges = 0
        for i, bed in enumerate(self.beds):
            if bed is not None and self.time >= bed['expected_discharge']:
                self.beds[i] = None
                discharges += 1
        return discharges
    
    def update_resources(self):
        for role in ['doctors']:
            fatigue = random.uniform(0, 0.05)
            shift = self.get_shift_info()
            self.current_staff[role] = max(0, self.current_staff[role] - fatigue)
            self.current_staff[role] = min(self.current_staff[role], self.base_staff[shift][role])
    
    def calculate_reward(self):
        occupancy = sum(1 for bed in self.beds if bed is not None)
        occupancy_reward = 50 * (occupancy / self.num_beds)
        queue_penalty = -10 * len(self.waiting_patients)
        shift = self.get_shift_info()
        staffing_penalty = -5 * (abs(self.current_staff.get('nurses', 0) - self.base_staff[shift]['nurses']) +
                                 abs(self.current_staff.get('doctors', 0) - self.base_staff[shift]['doctors']))
        total_resource_cost = sum(b['resource_cost'] for b in self.beds if b is not None)
        resource_penalty = -0.5 * total_resource_cost
        
        reward = occupancy_reward + queue_penalty + staffing_penalty + resource_penalty
        return reward
    
    def record_event(self, admissions=0, discharges=0):
        occupied = sum(1 for bed in self.beds if bed is not None)
        avg_wait = self.wait_time_log[-1] if self.wait_time_log else 0
        nurse_utilization = occupied / self.num_beds
        event = {
            'time': self.time,
            'occupied_beds': occupied,
            'waiting_patients': len(self.waiting_patients),
            'avg_wait_time': avg_wait,
            'nurse_utilization': nurse_utilization,
            'admissions': admissions,
            'discharges': discharges,
            'shift': self.get_shift_info()
        }
        self.event_log.append(event)
    
    def step(self, action):
        self.update_shift()
        for role in ['nurses', 'doctors']:
            adj = action.get('staff_adjustment', {}).get(role, 0)
            self.current_staff[role] = max(0, self.current_staff[role] + adj)
        self.simulate_arrivals()
        admitted_wait_times = self.admit_patients(priority_rule=action.get('admission_priority', 'severity'))
        admissions = len(admitted_wait_times)
        avg_wait_time = np.mean(admitted_wait_times) if admitted_wait_times else 0
        self.wait_time_log.append(avg_wait_time)
        discharges = self.simulate_discharges()
        self.update_resources()
        self.occupancy_log.append(sum(1 for bed in self.beds if bed is not None))
        self.staff_log.append(self.current_staff.copy())
        self.record_event(admissions=admissions, discharges=discharges)
        reward = self.calculate_reward()
        self.time += 1
        done = (self.time >= self.sim_duration)
        next_state = self.get_state()
        return next_state, reward, done
    
    def plot_event_log(self):
        times = [e['time'] for e in self.event_log]
        occupied_beds = [e['occupied_beds'] for e in self.event_log]
        waiting_patients = [e['waiting_patients'] for e in self.event_log]
        avg_wait_times = [e['avg_wait_time'] for e in self.event_log]
        nurse_utilization = [e['nurse_utilization'] for e in self.event_log]
        shifts = [e['shift'] for e in self.event_log]
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(times, occupied_beds, marker='o', label="Occupied Beds")
        plt.xlabel("Time (hr)")
        plt.ylabel("Occupied Beds")
        plt.title(f"{self.ward_name} Ward - Bed Occupancy")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(times, waiting_patients, marker='x', color='orange', label="Waiting Patients")
        plt.xlabel("Time (hr)")
        plt.ylabel("Queue Length")
        plt.title(f"{self.ward_name} Ward - Patient Queue")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(times, avg_wait_times, marker='s', color='green', label="Avg Wait Time")
        plt.xlabel("Time (hr)")
        plt.ylabel("Avg Wait Time (hr)")
        plt.title(f"{self.ward_name} Ward - Average Wait Time")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(times, nurse_utilization, marker='^', color='purple', label="Nurse Utilization")
        plt.xlabel("Time (hr)")
        plt.ylabel("Utilization Rate")
        plt.title(f"{self.ward_name} Ward - Nurse Utilization")
        plt.legend()
        
        for i, t in enumerate(times):
            if shifts[i] == 'day':
                plt.axvspan(t, t+1, facecolor='yellow', alpha=0.1)
            else:
                plt.axvspan(t, t+1, facecolor='blue', alpha=0.05)
        plt.tight_layout()
        plt.show()


class MultiWardHospitalEnv:
    def __init__(self, ward_configs):
        """
        Multi-ward hospital environment that aggregates several WardEnv instances.
        
        ward_configs: dict of {ward_name: config_dict}
          Each config_dict should have keys:
            - num_beds
            - base_staff (with keys 'day' and 'night')
            - sim_duration
            - arrival_lambda
            - nurse_efficiency
            - treatment_params (for Gamma distribution)
        """
        self.wards = {}
        for ward_name, config in ward_configs.items():
            self.wards[ward_name] = WardEnv(
                ward_name=ward_name,
                num_beds=config['num_beds'],
                base_staff=config['base_staff'],
                sim_duration=config['sim_duration'],
                arrival_lambda=config['arrival_lambda'],
                nurse_efficiency=config['nurse_efficiency'],
                treatment_params=config['treatment_params']
            )
        self.sim_duration = min([w.sim_duration for w in self.wards.values()])
        self.time = 0
    
    def reset(self):
        self.time = 0
        for ward in self.wards.values():
            ward.reset()
        return self.get_state()
    
    def get_state(self):
        state = {}
        for ward_name, ward in self.wards.items():
            state[ward_name] = ward.get_state()
        return state
    
    def step(self, actions):
        """
        actions: dict of {ward_name: action_dict}
        Each ward processes its own action.
        Returns aggregated next_state, total reward, and done flag.
        """
        total_reward = 0
        next_states = {}
        done = False
        for ward_name, ward in self.wards.items():
            action = actions.get(ward_name, {'staff_adjustment': {'nurses': 0, 'doctors': 0},
                                               'admission_priority': 'severity'})
            next_state, reward, ward_done = ward.step(action)
            total_reward += reward
            next_states[ward_name] = next_state
            done = done or ward_done
        self.time += 1
        return next_states, total_reward, done
    
    def plot_all_event_logs(self):
        for ward in self.wards.values():
            ward.plot_event_log()


def train_multiward_simulation(env, num_episodes=50):
    episode_rewards = []
    episode_avg_wait = {ward_name: [] for ward_name in env.wards.keys()}
    episode_throughput = {ward_name: [] for ward_name in env.wards.keys()}
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            actions = {}
            for ward_name in env.wards.keys():
                actions[ward_name] = {
                    'staff_adjustment': {
                        'nurses': random.choice([-1, 0, 1]),
                        'doctors': random.choice([-1, 0, 1])
                    },
                    'admission_priority': random.choice(['severity', 'fifo'])
                }
            state, reward, done = env.step(actions)
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        for ward_name, ward in env.wards.items():
            avg_wait = np.mean(ward.wait_time_log) if ward.wait_time_log else 0
            episode_avg_wait[ward_name].append(avg_wait)
            episode_throughput[ward_name].append(ward.throughput)
        if (episode+1)%10==0:
            print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")
            for ward_name in env.wards.keys():
                print(f"   {ward_name} Ward: Avg Wait = {episode_avg_wait[ward_name][-1]:.2f} hr, Throughput = {env.wards[ward_name].throughput}")
    return episode_rewards, episode_avg_wait, episode_throughput


def plot_episode_metrics(episode_rewards, episode_avg_wait, episode_throughput):
    """Plot aggregated KPI metrics across episodes after training."""
    episodes = list(range(1, len(episode_rewards) + 1))
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episodes, episode_rewards, marker='o')
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 3, 2)
    plt.plot(episodes, episode_avg_wait, marker='s', color='green')
    plt.title("Average Wait Time per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Wait Time (hr)")
    
    plt.subplot(1, 3, 3)
    plt.plot(episodes, episode_throughput, marker='^', color='purple')
    plt.title("Throughput per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Throughput")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define ward configurations.
    ward_configs = {
        'general': {
            'num_beds': 40,
            'base_staff': {'day': {'nurses': 10, 'doctors': 5},
                           'night': {'nurses': 6, 'doctors': 3}},
            'sim_duration': 168,
            'arrival_lambda': 0.8,
            'nurse_efficiency': 1.5,
            'treatment_params': {'shape': 3, 'scale': 20}
        },
        'ICU': {
            'num_beds': 10,
            'base_staff': {'day': {'nurses': 5, 'doctors': 3},
                           'night': {'nurses': 3, 'doctors': 2}},
            'sim_duration': 168,
            'arrival_lambda': 0.3,   # Lower arrivals for ICU
            'nurse_efficiency': 1.0, # ICU requires more intensive care
            'treatment_params': {'shape': 3, 'scale': 30}  # Longer treatment durations
        }
    }
    
    multi_env = MultiWardHospitalEnv(ward_configs)
    
    print("Running a multi-ward demonstration episode for event log recording...")
    state = multi_env.reset()
    done = False
    while not done:
        actions = {}
        for ward_name in multi_env.wards.keys():
            actions[ward_name] = {
                'staff_adjustment': {
                    'nurses': random.choice([-1, 0, 1]),
                    'doctors': random.choice([-1, 0, 1])
                },
                'admission_priority': random.choice(['severity', 'fifo'])
            }
        state, reward, done = multi_env.step(actions)
    print("Multi-ward demonstration episode complete.\n")
    
    multi_env.plot_all_event_logs()
    
    print("\nStarting training for multi-ward simulation over episodes...")
    episode_rewards, episode_avg_wait, episode_throughput = train_multiward_simulation(multi_env, num_episodes=50)
    
    print("Training complete. Generating final plots...")
    plot_episode_metrics(episode_rewards, episode_avg_wait['general'], episode_throughput['general'])
    plot_episode_metrics(episode_rewards, episode_avg_wait['ICU'], episode_throughput['ICU'])
    multi_env.plot_all_event_logs()
