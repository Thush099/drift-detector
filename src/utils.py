"""Synthetic concept drift stream generators and error/change-point helpers."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np


def generate_concept_drift_stream(
    data_size: int = 1000,
    drift_at: int = 500,
    shift_magnitude: float = 3.0,
    drift_type: Literal["abrupt", "gradual"] = "abrupt",
    n_features: int = 3,
    noise_std: float = 0.2,
    transition_width: Optional[int] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression stream ``(X, y)`` with optional concept drift.

    Feature vectors ``X_t ~ N(0, I)`` remain stationary while the conditional target
    mapping ``P(y | X)`` shifts at ``drift_at`` either abruptly or via a linear ramp.
    Passing ``drift_at >= data_size`` or ``shift_magnitude == 0`` yields a stationary
    stream.
    """
    if data_size < 20:
        raise ValueError(f"data_size must be >= 20, got {data_size}")
    if not 0 <= drift_at <= data_size:
        raise ValueError(f"drift_at must be in [0, {data_size}], got {drift_at}")
    if drift_type not in ("abrupt", "gradual"):
        raise ValueError(f"drift_type must be 'abrupt' or 'gradual', got {drift_type!r}")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")

    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(data_size, n_features))

    base_weights = np.array([1.5, -1.0, 0.5, 0.8, -0.6], dtype=np.float64)
    if n_features <= len(base_weights):
        w0 = base_weights[:n_features]
    else:
        w0 = rng.normal(0.0, 1.0, size=n_features)

    b0 = 0.0
    if shift_magnitude == 0.0 or drift_at >= data_size:
        w1 = w0.copy()
        b1 = b0
    else:
        w1 = -w0
        b1 = float(shift_magnitude)

    ramp_len = transition_width
    if ramp_len is None:
        remaining = max(1, data_size - drift_at)
        ramp_len = max(40, min(150, remaining // 3))

    alphas = np.zeros(data_size, dtype=np.float64)
    if drift_at < data_size and shift_magnitude != 0.0:
        if drift_type == "abrupt":
            alphas[drift_at:] = 1.0
        else:
            end_ramp = min(data_size, drift_at + max(1, ramp_len))
            for t in range(drift_at, end_ramp):
                alphas[t] = (t - drift_at + 1) / float(max(1, ramp_len))
            alphas[end_ramp:] = 1.0

    noise = rng.normal(0.0, noise_std, size=data_size)
    weights_t = (1.0 - alphas)[:, None] * w0[None, :] + alphas[:, None] * w1[None, :]
    biases_t = (1.0 - alphas) * b0 + alphas * b1
    y = np.sum(X * weights_t, axis=1) + biases_t + noise
    return X, y


def estimate_change_point(errors: Sequence[float], min_segment: int = 5) -> int:
    """Estimate the 1D 2-mean change-point index maximizing between-segment variance."""
    arr = np.asarray(errors, dtype=np.float64)
    n = len(arr)
    if n < 2 * min_segment:
        return 0
    prefix = np.cumsum(arr)
    total = prefix[-1]
    cuts = np.arange(min_segment, n - min_segment + 1)
    mu0 = prefix[cuts - 1] / cuts
    mu1 = (total - prefix[cuts - 1]) / (n - cuts)
    bss = (cuts * (n - cuts) / float(n)) * (mu0 - mu1) ** 2
    return int(cuts[int(np.argmax(bss))])


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, MSE, and RMSE for predictions against ground truth."""
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    return {"mae": mae, "mse": mse, "rmse": float(np.sqrt(mse))}
