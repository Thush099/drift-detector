from dataclasses import dataclass


@dataclass
class DriftDetectorConfig:
    """Configuration for streaming concept drift detection and online adaptation."""

    detector: str = "both"
    data_size: int = 1000
    drift_at: int = 500
    shift_magnitude: float = 3.0
    drift_type: str = "abrupt"
    warmup_size: int = 100
    adwin_delta: float = 0.002
    adwin_min_window: int = 10
    ph_delta: float = 0.005
    ph_threshold: float = 50.0
    ph_alpha: float = 1.0
    ph_min_instances: int = 30
    learning_rate: float = 0.0005
    seed: int = 42
