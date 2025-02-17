import numpy as np
import matplotlib.pyplot as plt
import random

from utils import flatten_state
import numpy as np
import matplotlib.pyplot as plt
import random

class WardEnv:
    def __init__(self, ward_name, num_beds, base_staff, sim_duration,
                 arrival_lambda, nurse_efficiency, treatment_params):
        """
        Initialize the ward environment.
        Args:
            ward_name (str): Name of the ward.
            num_beds (int): Total number of beds.
            base_staff (dict): Baseline staffing for 'day' and 'night'.
            sim_duration (int): Timesteps in an episode.
            arrival_lambda (float): Mean patient arrival rate.
            nurse_efficiency (float): Nurse capacity multiplier.
            treatment_params (dict): Gamma distribution parameters for treatment.
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
        """Resets logging metrics for the episode."""
        self.occupancy_log = []
        self.queue_log = []
        self.wait_time_log = []
        self.staff_log = []
        self.cumulative_throughput = 0
        self.event_log = []
        self.nurse_fatigue = 0
        self.recent_waits = []
        self.prev_queue_length = 0

    def reset(self):
        """Resets the environment to the initial state."""
        self.time = 0
        self.beds = [None] * self.num_beds
        self.waiting_patients = []
        self.last_shift = self.get_shift_info()
        self.current_staff = self.base_staff[self.last_shift].copy()
        self.reset_metrics()
        return self.get_state()

    def get_shift_info(self):
        """Returns 'day' or 'night' based on current time."""
        return 'day' if 6 <= self.time % 24 < 18 else 'night'

    def update_shift(self):
        """Checks and updates the shift if needed."""
        current_shift = self.get_shift_info()
        if current_shift != self.last_shift:
            self.current_staff = self.base_staff[current_shift].copy()
            self.last_shift = current_shift

    def forecast_arrivals(self):
        """Forecasts arrivals using a Poisson process."""
        predicted_mean = np.random.poisson(lam=self.arrival_lambda)
        predicted_std = np.sqrt(predicted_mean)
        return {'mean': predicted_mean, 'std': predicted_std}

    def simulate_arrivals(self):
        """Simulates patient arrivals for the current timestep."""
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
        """
        Admits patients based on the priority rule.
        Returns:
            List of wait times for admitted patients.
        """
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
                patient = self.waiting_patients.pop(0)
            else:
                patient = sorted_patients.pop(0)
                self.waiting_patients.remove(patient)
            wait_time = self.time - patient['arrival_time']
            admitted_wait_times.append(wait_time)
            self.recent_waits.append(wait_time)
            if len(self.recent_waits) > 10:
                self.recent_waits.pop(0)
            treatment_duration = np.random.gamma(shape=self.treatment_params['shape'],
                                                 scale=self.treatment_params['scale'])
            expected_discharge = self.time + treatment_duration
            patient_record = {
                'arrival_time': patient['arrival_time'],
                'severity': patient['severity'],
                'triage': patient['triage'],
                'resource_cost': patient['resource_cost'],
                'expected_discharge': expected_discharge
            }
            self.beds[bed_idx] = patient_record
            self.cumulative_throughput += 1
        return admitted_wait_times

    def simulate_discharges(self):
        """Simulates discharges at the current timestep."""
        discharges = 0
        for i, bed in enumerate(self.beds):
            if bed is not None and self.time >= bed['expected_discharge']:
                self.beds[i] = None
                discharges += 1
        return discharges

    def update_resources(self):
        """Updates staffing and nurse fatigue."""
        shift = self.get_shift_info()
        for role in ['nurses', 'doctors']:
            fatigue = random.uniform(0, 0.02)
            self.current_staff[role] = max(0, self.current_staff[role] - fatigue)
            self.current_staff[role] = min(self.current_staff[role], self.base_staff[shift][role])
        self.nurse_fatigue += (self.base_staff[self.get_shift_info()]['nurses'] - self.current_staff.get('nurses', 0))

    def compute_workload(self):
        """Computes average severity of active patients."""
        active_patients = [b for b in self.beds if b is not None]
        if not active_patients:
            return 0
        return sum(b['severity'] for b in active_patients) / len(active_patients)

    def get_state(self):
        """Returns a normalized state dictionary."""
        occupancy = sum(1 for bed in self.beds if bed is not None)
        shift = self.get_shift_info()
        state = {
            'time': self.time,
            'occupied_ratio': occupancy / self.num_beds,
            'free_beds': self.num_beds - occupancy,
            'waiting_patients': len(self.waiting_patients),
            'staff_nurses': self.current_staff.get('nurses', 0) / self.base_staff[shift]['nurses'],
            'staff_doctors': self.current_staff.get('doctors', 0) / self.base_staff[shift]['doctors'],
            'predicted_arrivals_mean': self.forecast_arrivals()['mean'] / (self.arrival_lambda + 1),
            'predicted_arrivals_std': self.forecast_arrivals()['std'] / (np.sqrt(self.arrival_lambda) + 1),
            'shift': shift,
            'workload': self.compute_workload() / 5.0,
            'nurse_fatigue': self.nurse_fatigue / 100.0
        }
        return state

    def calculate_reward(self, admitted_wait_times):
        """
        Computes reward based on multiple objectives:
          - Throughput, queue length, wait times, staffing, resource costs, fatigue, and shaping.
        """
        throughput_reward = 15 * len(admitted_wait_times)
        queue_length = len(self.waiting_patients)
        queue_penalty = -1 * min(queue_length, 30)
        avg_wait = np.mean(admitted_wait_times) if admitted_wait_times else 0
        wait_time_penalty = -2.0 * (avg_wait / 10.0)
        shift = self.get_shift_info()
        nurse_diff = abs(self.current_staff.get('nurses', 0) - self.base_staff[shift]['nurses'])
        doctor_diff = abs(self.current_staff.get('doctors', 0) - self.base_staff[shift]['doctors'])
        staffing_penalty = -1.0 * (nurse_diff + doctor_diff)
        total_resource_cost = sum(b['resource_cost'] for b in self.beds if b is not None)
        resource_penalty = -0.05 * total_resource_cost
        fatigue_penalty = -1.0 * (self.nurse_fatigue / 10.0)
        queue_improvement = self.prev_queue_length - queue_length
        shaping_reward = 1.0 * queue_improvement if queue_improvement > 0 else 0
        reward = (throughput_reward + queue_penalty + wait_time_penalty +
                  staffing_penalty + resource_penalty + fatigue_penalty + shaping_reward)
        return reward

    def record_event(self, admissions=0, discharges=0):
        """Records metrics for later analysis."""
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
        """
        Executes one timestep: updates shift, applies action, simulates arrivals/admissions/discharges,
        updates resources, logs event, computes reward, and returns the next state.
        """
        self.update_shift()
        for role in ['nurses', 'doctors']:
            adj = action.get('staff_adjustment', {}).get(role, 0)
            self.current_staff[role] = max(0, self.current_staff[role] + adj)
        self.simulate_arrivals()
        admitted_wait_times = self.admit_patients(priority_rule=action.get('admission_priority', 'severity'))
        if admitted_wait_times:
            self.wait_time_log.append(np.mean(admitted_wait_times))
        else:
            self.wait_time_log.append(0)
        discharges = self.simulate_discharges()
        self.update_resources()
        self.occupancy_log.append(sum(1 for bed in self.beds if bed is not None))
        self.staff_log.append(self.current_staff.copy())
        self.record_event(admissions=len(admitted_wait_times), discharges=discharges)
        reward = self.calculate_reward(admitted_wait_times)
        self.prev_queue_length = len(self.waiting_patients)
        self.time += 1
        done = (self.time >= self.sim_duration)
        next_state = self.get_state()
        return next_state, reward, done

    def test_episode(self, agent, max_timesteps=None, epsilon=0.0):
        """
        Runs a single episode in test mode, printing details at each timestep.
        Args:
            agent: The DRL agent with an act() method.
            max_timesteps (int): Optionally limit the timesteps.
            epsilon (float): Exploration parameter.
        Returns:
            dict: Log containing timestep, action, state, reward, and cumulative reward.
        """
        if max_timesteps is None:
            max_timesteps = self.sim_duration
        self.reset()
        state = flatten_state(self.get_state())
        episode_log = {
            'timestep': [],
            'action': [],
            'state': [],
            'reward': [],
            'cumulative_reward': []
        }
        cumulative_reward = 0.0
        for t in range(max_timesteps):
            action = agent.act(state, epsilon=epsilon)
            next_state_dict, reward, done = self.step(action)
            next_state = flatten_state(next_state_dict)
            cumulative_reward += reward
            episode_log['timestep'].append(t)
            episode_log['action'].append(action)
            episode_log['state'].append(state)
            episode_log['reward'].append(reward)
            episode_log['cumulative_reward'].append(cumulative_reward)
            print(f"\n--- Timestep {t} ---")
            print(f"Action: {action}")
            print(f"State: {state}")
            print(f"Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}")
            state = next_state
            if done:
                print("\nEpisode ended at timestep", t)
                break
        print("\nTest episode complete. Final cumulative reward: {:.2f}".format(cumulative_reward))
        return episode_log

        
#####################################
# View: Visualization (unchanged for brevity)
#####################################

class HospitalView:
    @staticmethod
    def plot_event_log(ward: WardEnv):
        times = [e['time'] for e in ward.event_log]
        occupied_beds = [e['occupied_beds'] for e in ward.event_log]
        waiting_patients = [e['waiting_patients'] for e in ward.event_log]
        avg_wait_times = [e['avg_wait_time'] for e in ward.event_log]
        nurse_utilization = [e['nurse_utilization'] for e in ward.event_log]
        shifts = [e['shift'] for e in ward.event_log]
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(times, occupied_beds, marker='o', label="Occupied Beds")
        plt.xlabel("Time (hr)")
        plt.ylabel("Occupied Beds")
        plt.title(f"{ward.ward_name} Ward - Bed Occupancy")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(times, waiting_patients, marker='x', color='orange', label="Waiting Patients")
        plt.xlabel("Time (hr)")
        plt.ylabel("Queue Length")
        plt.title(f"{ward.ward_name} Ward - Patient Queue")
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(times, avg_wait_times, marker='s', color='green', label="Avg Wait Time")
        plt.xlabel("Time (hr)")
        plt.ylabel("Avg Wait Time (hr)")
        plt.title(f"{ward.ward_name} Ward - Average Wait Time")
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(times, nurse_utilization, marker='^', color='purple', label="Nurse Utilization")
        plt.xlabel("Time (hr)")
        plt.ylabel("Utilization Rate")
        plt.title(f"{ward.ward_name} Ward - Nurse Utilization")
        plt.legend()
        plt.tight_layout()
        plt.show()

