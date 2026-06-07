from dataclasses import dataclass

@dataclass
class DriftDetectorConfig:
    window_size: int = 200
    data_size: int = 1000
    drift_size: int = 500