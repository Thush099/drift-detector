from typing import List
import logging
import numpy as np

def generate_synthetic_data(data_size: int, drift_size: int):
    data = np.random.normal(0, 1, data_size - drift_size)
    data = np.concatenate((data, np.random.normal(1, 1, drift_size)))
    return data.tolist()

def calculate_drift(data: List[float], prediction: float):
    # Implement ADWIN and Page-Hinkley algorithms
    # For simplicity, use a simple threshold-based approach
    threshold = 0.5
    return np.mean(data) > threshold