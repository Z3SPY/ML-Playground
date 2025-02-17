# utils.py
import numpy as np

def flatten_state(state):
    """
    Flattens the state dictionary into a fixed-size numpy array.
    Args:
        state (dict): The state from the environment.
    Returns:
        np.array: Flattened state vector.
    """
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
