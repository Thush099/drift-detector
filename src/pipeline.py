"""Streaming drift detection and adaptive retraining pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Literal, Optional

import numpy as np

from src.detectors import ADWINDetector, BaseDriftDetector, DriftEvent, PageHinkleyDetector
from src.models import OnlineRegressor
from src.utils import compute_regression_metrics, estimate_change_point

logger = logging.getLogger(__name__)


@dataclass
class DetectorRunSummary:
    """Metrics and detected events for a single detector run over the stream."""

    detector: str
    events: list[DriftEvent]
    predictions: np.ndarray
    errors: np.ndarray
    overall_mae: float
    overall_mse: float
    post_drift_mae: float
    post_drift_mse: float
    n_retrains: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "n_detections": len(self.events),
            "n_retrains": self.n_retrains,
            "overall_mae": round(self.overall_mae, 6),
            "overall_mse": round(self.overall_mse, 6),
            "post_drift_mae": round(self.post_drift_mae, 6),
            "post_drift_mse": round(self.post_drift_mse, 6),
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class PipelineResult:
    """Full comparison result between adaptive detector(s) and static baseline."""

    detector_mode: str
    data_size: int
    warmup_size: int
    drift_at: Optional[int]
    static_overall_mae: float
    static_overall_mse: float
    static_post_drift_mae: float
    static_post_drift_mse: float
    adaptive_overall_mae: float
    adaptive_overall_mse: float
    adaptive_post_drift_mae: float
    adaptive_post_drift_mse: float
    events: list[DriftEvent] = field(default_factory=list)
    runs: dict[str, DetectorRunSummary] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_mode": self.detector_mode,
            "data_size": self.data_size,
            "warmup_size": self.warmup_size,
            "drift_at": self.drift_at,
            "static": {
                "overall_mae": round(self.static_overall_mae, 6),
                "overall_mse": round(self.static_overall_mse, 6),
                "post_drift_mae": round(self.static_post_drift_mae, 6),
                "post_drift_mse": round(self.static_post_drift_mse, 6),
            },
            "adaptive": {
                "overall_mae": round(self.adaptive_overall_mae, 6),
                "overall_mse": round(self.adaptive_overall_mse, 6),
                "post_drift_mae": round(self.adaptive_post_drift_mae, 6),
                "post_drift_mse": round(self.adaptive_post_drift_mse, 6),
            },
            "runs": {name: run.to_dict() for name, run in self.runs.items()},
            "events": [e.to_dict() for e in self.events],
        }

    def format_summary(self) -> str:
        """Render a human-readable CLI summary comparing static vs adaptive models."""
        lines = [
            f"Stream length: {self.data_size} (warmup: {self.warmup_size}, evaluated: {self.data_size - self.warmup_size})",
            f"Drift point: {self.drift_at if self.drift_at is not None else 'none'} | Mode: {self.detector_mode}",
            "",
            "Detected Drift Events:",
        ]
        if not self.events:
            lines.append("  (none)")
        else:
            for ev in self.events:
                lines.append(
                    f"  - step={ev.step:<4d} detector={ev.detector:<12s} "
                    f"pre_mean={ev.pre_mean:.4f} post_mean={ev.post_mean:.4f} "
                    f"magnitude={ev.magnitude:.4f}"
                )

        lines.extend(
            [
                "",
                "Model Error Comparison (Static Baseline vs. Adaptive Retraining):",
                f"  {'Model':<25s} {'Overall MAE':>12s} {'Overall MSE':>12s} {'Post-Drift MAE':>16s} {'Post-Drift MSE':>16s} {'Retrains':>10s}",
                f"  {'-' * 95}",
                f"  {'static_baseline':<25s} {self.static_overall_mae:12.4f} {self.static_overall_mse:12.4f} "
                f"{self.static_post_drift_mae:16.4f} {self.static_post_drift_mse:16.4f} {0:10d}",
            ]
        )
        for name, run in self.runs.items():
            lines.append(
                f"  {f'adaptive ({name})':<25s} {run.overall_mae:12.4f} {run.overall_mse:12.4f} "
                f"{run.post_drift_mae:16.4f} {run.post_drift_mse:16.4f} {run.n_retrains:10d}"
            )
        return "\n".join(lines)


class DriftDetectorPipeline:
    """Streaming ML monitoring pipeline with ADWIN / Page-Hinkley drift detection.

    Fits an initial online regressor on a warm-up prefix, streams predictions
    step-by-step (prequential evaluation), monitors absolute prediction errors with
    ADWIN and/or Page-Hinkley, retrains on clean post-drift observations when drift
    fires, and compares adaptive errors against a frozen static baseline.
    """

    def __init__(
        self,
        detector: Literal["adwin", "page_hinkley", "both"] = "both",
        warmup_size: int = 100,
        adwin_delta: float = 0.002,
        adwin_min_window: int = 10,
        ph_delta: float = 0.005,
        ph_threshold: float = 50.0,
        ph_alpha: float = 1.0,
        ph_min_instances: int = 30,
        learning_rate: float = 0.0005,
        min_retrain_samples: int = 10,
        seed: int = 42,
    ) -> None:
        if detector not in ("adwin", "page_hinkley", "both"):
            raise ValueError(f"detector must be 'adwin', 'page_hinkley', or 'both', got {detector!r}")
        if warmup_size < 10:
            raise ValueError(f"warmup_size must be >= 10, got {warmup_size}")

        self.detector_mode = detector
        self.warmup_size = int(warmup_size)
        self.adwin_delta = float(adwin_delta)
        self.adwin_min_window = int(adwin_min_window)
        self.ph_delta = float(ph_delta)
        self.ph_threshold = float(ph_threshold)
        self.ph_alpha = float(ph_alpha)
        self.ph_min_instances = int(ph_min_instances)
        self.learning_rate = float(learning_rate)
        self.min_retrain_samples = int(min_retrain_samples)
        self.seed = int(seed)

    def _build_detector(self, name: str) -> BaseDriftDetector:
        if name == "adwin":
            return ADWINDetector(delta=self.adwin_delta, min_window=self.adwin_min_window)
        if name == "page_hinkley":
            return PageHinkleyDetector(
                delta=self.ph_delta,
                threshold=self.ph_threshold,
                alpha=self.ph_alpha,
                min_instances=self.ph_min_instances,
                mode="up",
            )
        raise ValueError(f"Unknown detector: {name!r}")

    def _run_single_detector(
        self,
        det_name: str,
        base_model: OnlineRegressor,
        X: np.ndarray,
        y: np.ndarray,
        post_eval_slice: slice,
    ) -> DetectorRunSummary:
        det = self._build_detector(det_name)
        model = base_model.clone()

        n_eval = len(X) - self.warmup_size
        preds = np.empty(n_eval, dtype=np.float64)
        errors = np.empty(n_eval, dtype=np.float64)
        events: list[DriftEvent] = []
        n_retrains = 0

        x_buf: list[np.ndarray] = []
        y_buf: list[float] = []
        err_buf: list[float] = []

        for idx, t in enumerate(range(self.warmup_size, len(X))):
            x_t = X[t]
            y_t = float(y[t])
            y_hat = float(model.predict(x_t)[0])
            err = abs(y_t - y_hat)

            preds[idx] = y_hat
            errors[idx] = err
            x_buf.append(x_t)
            y_buf.append(y_t)
            err_buf.append(err)

            if det.update(err):
                pre_mean = float(det.last_pre_mean if det.last_pre_mean is not None else 0.0)
                post_mean = float(det.last_post_mean if det.last_post_mean is not None else err)
                magnitude = float(det.last_magnitude if det.last_magnitude is not None else abs(post_mean - pre_mean))
                event = DriftEvent(
                    step=t,
                    detector=det_name,
                    pre_mean=pre_mean,
                    post_mean=post_mean,
                    magnitude=magnitude,
                )
                events.append(event)
                logger.info(
                    "Drift detected at step %d (%s): pre_mean=%.4f, post_mean=%.4f, magnitude=%.4f",
                    t,
                    det_name,
                    pre_mean,
                    post_mean,
                    magnitude,
                )

                if isinstance(det, ADWINDetector) and det.window_size >= self.min_retrain_samples:
                    post_start = max(0, len(x_buf) - det.window_size)
                else:
                    post_start = estimate_change_point(err_buf, min_segment=5)
                    if len(x_buf) - post_start < self.min_retrain_samples:
                        post_start = max(0, len(x_buf) - self.min_retrain_samples)

                post_X = np.asarray(x_buf[post_start:], dtype=np.float64)
                post_y = np.asarray(y_buf[post_start:], dtype=np.float64)
                model.retrain(post_X, post_y)
                n_retrains += 1

                # Reset detector and buffer so pre-retrain errors from the stale model are cleared.
                det.reset()
                x_buf.clear()
                y_buf.clear()
                err_buf.clear()
            else:
                model.partial_fit(x_t.reshape(1, -1), np.array([y_t], dtype=np.float64))

        y_eval = y[self.warmup_size:]
        overall = compute_regression_metrics(y_eval, preds)
        post = compute_regression_metrics(y_eval[post_eval_slice], preds[post_eval_slice])

        return DetectorRunSummary(
            detector=det_name,
            events=events,
            predictions=preds,
            errors=errors,
            overall_mae=overall["mae"],
            overall_mse=overall["mse"],
            post_drift_mae=post["mae"],
            post_drift_mse=post["mse"],
            n_retrains=n_retrains,
        )

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        drift_at: Optional[int] = None,
    ) -> PipelineResult:
        """Execute streaming evaluation, drift detection, and adaptive retraining."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if len(X_arr) != len(y_arr):
            raise ValueError(f"X and y length mismatch: {len(X_arr)} vs {len(y_arr)}")
        if len(X_arr) <= self.warmup_size:
            raise ValueError(
                f"Stream length ({len(X_arr)}) must exceed warmup_size ({self.warmup_size})"
            )

        base_model = OnlineRegressor(learning_rate=self.learning_rate, seed=self.seed)
        base_model.fit(X_arr[: self.warmup_size], y_arr[: self.warmup_size])

        y_eval = y_arr[self.warmup_size:]
        static_preds = base_model.predict(X_arr[self.warmup_size:])

        if drift_at is not None and self.warmup_size <= drift_at < len(X_arr):
            post_eval_slice = slice(drift_at - self.warmup_size, None)
        else:
            post_eval_slice = slice(0, None)

        static_overall = compute_regression_metrics(y_eval, static_preds)
        static_post = compute_regression_metrics(y_eval[post_eval_slice], static_preds[post_eval_slice])

        detector_names = ["adwin", "page_hinkley"] if self.detector_mode == "both" else [self.detector_mode]
        runs: dict[str, DetectorRunSummary] = {}
        all_events: list[DriftEvent] = []

        for det_name in detector_names:
            summary = self._run_single_detector(det_name, base_model, X_arr, y_arr, post_eval_slice)
            runs[det_name] = summary
            all_events.extend(summary.events)

        all_events.sort(key=lambda e: (e.step, e.detector))
        best_run = min(runs.values(), key=lambda r: r.post_drift_mae)

        return PipelineResult(
            detector_mode=self.detector_mode,
            data_size=len(X_arr),
            warmup_size=self.warmup_size,
            drift_at=drift_at,
            static_overall_mae=static_overall["mae"],
            static_overall_mse=static_overall["mse"],
            static_post_drift_mae=static_post["mae"],
            static_post_drift_mse=static_post["mse"],
            adaptive_overall_mae=best_run.overall_mae,
            adaptive_overall_mse=best_run.overall_mse,
            adaptive_post_drift_mae=best_run.post_drift_mae,
            adaptive_post_drift_mse=best_run.post_drift_mse,
            events=all_events,
            runs=runs,
        )
