import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class HospitalWardEnv:
    def __init__(self, num_beds=50, 
                 base_staff={'day': {'nurses': 10, 'doctors': 5},
                             'night': {'nurses': 6,  'doctors': 3}},
                 sim_duration=168, 
                 arrival_lambda=0.8,        # ~0.8 arrivals per hour (~20 per day)
                 nurse_efficiency=1.5):      # Each nurse can process ~1.5 admissions per hour
        """
        Single-ward simulation environment for inpatient ward resource allocation.
        
        Designed for a hybrid DRL-MS0 framework study.
        
        Processes per timestep (1 hour):
          1. Apply staff adjustments (DRL actions).
          2. Process patient arrivals (via a Poisson process).
          3. Admit patients using severity-based prioritization.
          4. Assign treatment durations (Gamma distribution, mean ~60 hr) so that patients remain for several days.
          5. Process scheduled discharges.
          6. Update resource states (simulate doctor fatigue).
          7. Record detailed events.
        
        Parameters:
          num_beds: Total beds available in the ward.
          base_staff: Baseline staffing levels per shift (e.g., day and night).
          sim_duration: Timesteps per episode (default 168 hr = 1 week).
          arrival_lambda: Average number of patient arrivals per hour.
          nurse_efficiency: Number of patients a nurse can process per hour.
          
        Future placeholders:
          - Diversion actions if waiting queue exceeds a threshold.
          - Extended nurse/shift dynamics.
          - Multi-ward or ICU extensions.
          - Integration with an MSO module.
        """
        self.num_beds = num_beds
        self.base_staff = base_staff.copy()
        self.sim_duration = sim_duration  
        self.arrival_lambda = arrival_lambda
        self.nurse_efficiency = nurse_efficiency
        self.time = 0
        # Track the last shift to update staffing on shift changes.
        self.last_shift = self.get_shift_info()
        # Placeholder for additional resources (e.g., diagnostic machines)
        self.resources = {'diagnostic_machines': 5}  
        self.reset_metrics()
        self.reset()

    def reset_metrics(self):
        """Initialize logs for occupancy, waiting queue, wait times, staffing, throughput, and events."""
        self.occupancy_log = []    
        self.queue_log = []        
        self.wait_time_log = []    
        self.staff_log = []        
        self.throughput = 0        
        self.event_log = []        

    def reset(self):
        """Reset the environment state for a new episode."""
        self.time = 0
        self.beds = [None] * self.num_beds  
        self.waiting_patients = deque()     
        self.last_shift = self.get_shift_info()
        self.current_staff = self.base_staff[self.last_shift].copy()
        self.reset_metrics()
        return self.get_state()

    def get_state(self):
        """
        Construct the current state of the environment.
        The state includes:
          - Current time,
          - Number of occupied and free beds,
          - Number of waiting patients,
          - Current staffing,
          - Predicted arrivals,
          - Current shift.
        """
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
        """Return 'day' if current hour (mod 24) is between 6 and 18; else 'night'."""
        return 'day' if 6 <= self.time % 24 < 18 else 'night'

    def update_shift(self):
        """Check for shift change and update current_staff if necessary."""
        current_shift = self.get_shift_info()
        if current_shift != self.last_shift:
            self.current_staff = self.base_staff[current_shift].copy()
            self.last_shift = current_shift

    def forecast_arrivals(self):
        """
        Forecast patient arrivals using a Poisson process.
        With arrival_lambda=0.8, expect roughly 0.8 patients per hour.
        Returns a dictionary with 'mean' and 'std'.
        """
        predicted_mean = np.random.poisson(lam=self.arrival_lambda)
        predicted_std = np.sqrt(predicted_mean)
        return {'mean': predicted_mean, 'std': predicted_std}

    def simulate_arrivals(self):
        """
        Generate patient arrivals.
        Each patient is assigned:
          - arrival_time,
          - severity (1 to 5),
          - triage type ('elective' or 'emergency'),
          - a dummy resource cost.
        """
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
        Admit patients from the waiting queue if beds are available and nurse capacity allows.
        Nurses are treated as a per-hour resource:
          effective capacity = current_staff['nurses'] * nurse_efficiency.
        Patients are sorted by severity if 'severity' is selected; otherwise, FIFO.
        
        For each admitted patient:
          - Sample a treatment duration from a Gamma distribution (shape=3, scale=20) → mean ~60 hours.
          - Calculate expected discharge = current time + treatment duration.
          - Place the patient record in a free bed.
        
        Returns a list of wait times for admitted patients.
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
                patient = self.waiting_patients.popleft()
            else:
                patient = sorted_patients.pop(0)
                self.waiting_patients.remove(patient)
            wait_time = self.time - patient['arrival_time']
            admitted_wait_times.append(wait_time)
            
            treatment_duration = np.random.gamma(shape=3, scale=20)
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
        """
        Discharge patients whose expected discharge time has passed.
        For each discharged patient, free the bed.
        Returns the number of discharges this timestep.
        """
        discharges = 0
        for i, bed in enumerate(self.beds):
            if bed is not None and self.time >= bed['expected_discharge']:
                self.beds[i] = None
                discharges += 1
        return discharges

    def update_resources(self):
        """
        Update resource states to simulate doctor fatigue.
        Apply a small random fatigue adjustment to doctors and cap at baseline.
        (Future work: Incorporate nurse fatigue and additional resource dynamics.)
        """
        for role in ['doctors']:
            fatigue = random.uniform(0, 0.05)
            # Use current shift to get the appropriate baseline.
            shift = self.get_shift_info()
            self.current_staff[role] = max(0, self.current_staff[role] - fatigue)
            self.current_staff[role] = min(self.current_staff[role], self.base_staff[shift][role])

    def calculate_reward(self):
        """
        Compute a multi-objective reward for the timestep.
        Components:
          - Occupancy reward: +50 * (occupied beds / total beds)
          - Queue penalty: -10 per waiting patient
          - Staffing penalty: -5 per unit deviation from baseline (shift-specific)
          - Resource penalty: -0.5 * total resource cost of patients currently in beds
        """
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
        """
        Record an event for the current timestep with key metrics:
          - Time, occupied beds, waiting patients, average wait time,
          - Nurse utilization (proxy: occupied beds / total beds),
          - Admissions, discharges, and current shift.
        """
        occupied = sum(1 for bed in self.beds if bed is not None)
        avg_wait = self.wait_time_log[-1] if self.wait_time_log else 0
        nurse_utilization = occupied / self.num_beds  # Simplistic proxy.
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
        Advance the simulation by one timestep (1 hour) following these steps:
          1. Update shift if needed.
          2. Apply staff adjustments (via action).
          3. Process patient arrivals.
          4. Admit patients (using the specified admission priority).
          5. Process scheduled discharges.
          6. Update resource states.
          7. Record event details.
          8. Calculate reward.
          9. Advance time.
        
        Parameters:
          action: Dict with keys:
            - 'staff_adjustment': e.g., {'nurses': +1, 'doctors': 0}
            - 'admission_priority': 'severity' or 'fifo'
        
        Returns:
          next_state, reward, done (bool indicating end of episode).
        """
        # (1) Update shift if there's a change.
        self.update_shift()
        
        # (2) Apply staff adjustments.
        for role in ['nurses', 'doctors']:
            adj = action.get('staff_adjustment', {}).get(role, 0)
            self.current_staff[role] = max(0, self.current_staff[role] + adj)
        
        # (3) Process patient arrivals.
        self.simulate_arrivals()
        
        # (4) Admit patients.
        admitted_wait_times = self.admit_patients(priority_rule=action.get('admission_priority', 'severity'))
        admissions = len(admitted_wait_times)
        avg_wait_time = np.mean(admitted_wait_times) if admitted_wait_times else 0
        self.wait_time_log.append(avg_wait_time)
        
        # (5) Process scheduled discharges.
        discharges = self.simulate_discharges()
        
        # (6) Update resources.
        self.update_resources()
        
        # (7) Log occupancy and staffing.
        self.occupancy_log.append(sum(1 for bed in self.beds if bed is not None))
        self.staff_log.append(self.current_staff.copy())
        
        # (8) Record event.
        self.record_event(admissions=admissions, discharges=discharges)
        
        # (9) Calculate reward.
        reward = self.calculate_reward()
        
        # (10) Advance time.
        self.time += 1
        done = (self.time >= self.sim_duration)
        next_state = self.get_state()
        return next_state, reward, done

    def plot_event_log(self):
        """
        Generate a comprehensive plot showing:
          - Occupied beds over time,
          - Waiting patient count,
          - Average wait time,
          - Nurse utilization.
        Background shading indicates shift changes.
        """
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
        plt.title("Bed Occupancy Over Time")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(times, waiting_patients, marker='x', color='orange', label="Waiting Patients")
        plt.xlabel("Time (hr)")
        plt.ylabel("Queue Length")
        plt.title("Patient Queue Dynamics")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(times, avg_wait_times, marker='s', color='green', label="Avg Wait Time")
        plt.xlabel("Time (hr)")
        plt.ylabel("Avg Wait Time (hr)")
        plt.title("Average Patient Wait Time")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(times, nurse_utilization, marker='^', color='purple', label="Nurse Utilization")
        plt.xlabel("Time (hr)")
        plt.ylabel("Utilization Rate")
        plt.title("Nurse Utilization Over Time")
        plt.legend()
        
        for i, t in enumerate(times):
            if shifts[i] == 'day':
                plt.axvspan(t, t+1, facecolor='yellow', alpha=0.1)
            else:
                plt.axvspan(t, t+1, facecolor='blue', alpha=0.05)
        
        plt.tight_layout()
        plt.show()


def train_simulation(env, num_episodes=100):
    """
    Run multiple episodes (each one week long) and log episode-level KPIs.
    Every 10 episodes, print aggregated metrics (total reward, avg wait time, throughput).
    Returns lists of episode-level metrics.
    """
    episode_rewards = []
    episode_avg_wait = []
    episode_throughput = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = {
                'staff_adjustment': {
                    'nurses': random.choice([-1, 0, 1]),
                    'doctors': random.choice([-1, 0, 1])
                },
                'admission_priority': random.choice(['severity', 'fifo'])
            }
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        
        episode_rewards.append(total_reward)
        avg_wait = np.mean(env.wait_time_log) if env.wait_time_log else 0
        episode_avg_wait.append(avg_wait)
        episode_throughput.append(env.throughput)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Avg Wait = {avg_wait:.2f} hr, Throughput = {env.throughput}")
    
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
    # Initialize the environment with our updated realistic parameters.
    env = HospitalWardEnv(sim_duration=168, arrival_lambda=0.8, nurse_efficiency=1.5)
    
    # Run one demonstration episode and record detailed events.
    print("Running a demonstration episode for event log recording...")
    state = env.reset()
    while True:
        action = {
            'staff_adjustment': {
                'nurses': random.choice([-1, 0, 1]),
                'doctors': random.choice([-1, 0, 1])
            },
            'admission_priority': random.choice(['severity', 'fifo'])
        }
        state, reward, done = env.step(action)
        if done:
            break
    print("Demonstration episode complete.\n")
    
    # Print the event log for the demonstration episode.
    print("Final Episode Event History:")
    for event in env.event_log:
        print(f"Time: {event['time']} hr | Occupied Beds: {event['occupied_beds']} | Waiting Patients: {event['waiting_patients']} | "
              f"Avg Wait: {event['avg_wait_time']:.2f} hr | Nurse Utilization: {event['nurse_utilization']:.2f} | "
              f"Admissions: {event['admissions']} | Discharges: {event['discharges']} | Shift: {event['shift']}")
    
    # Train the simulation over multiple episodes (print indicators every 10 episodes).
    print("\nStarting training over episodes...")
    episode_rewards, episode_avg_wait, episode_throughput = train_simulation(env, num_episodes=100)
    
    # After training, plot aggregated episode metrics and the event log from the demonstration episode.
    print("Training complete. Generating final plots...")
    plot_episode_metrics(episode_rewards, episode_avg_wait, episode_throughput)
    env.plot_event_log()
