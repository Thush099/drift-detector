"""Streaming concept drift detectors: ADWIN and Page-Hinkley."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class DriftEvent:
    """Structured record of a detected concept drift event."""

    step: int
    detector: str
    pre_mean: float
    post_mean: float
    magnitude: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the drift event to a JSON-compatible dictionary."""
        return asdict(self)


class BaseDriftDetector(ABC):
    """Abstract base class for streaming scalar drift detectors."""

    name: str = "base"

    def __init__(self) -> None:
        self.total_seen: int = 0
        self.n_detections: int = 0
        self.last_pre_mean: Optional[float] = None
        self.last_post_mean: Optional[float] = None
        self.last_magnitude: Optional[float] = None

    @abstractmethod
    def update(self, value: float) -> bool:
        """Ingest one scalar observation and return True if drift is detected."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal window/accumulator statistics."""

    def _validate_value(self, value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Expected finite scalar observation, got {value!r}")
        return numeric


class ADWINDetector(BaseDriftDetector):
    """Adaptive Windowing (ADWIN) streaming concept drift detector.

    Maintains a variable-length sliding window of recent scalar observations and
    evaluates the Hoeffding cut criterion ``|mu_0 - mu_1| >= eps_cut`` across split
    points ``(W_0, W_1)``. When a statistically significant shift is found, the
    stale pre-drift subwindow ``W_0`` is dropped.

    Reference:
        Bifet, A., & Gavalda, R. (2007). Learning from time-changing data with
        adaptive windowing. SDM.
    """

    name: str = "adwin"

    def __init__(self, delta: float = 0.002, min_window: int = 10) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if min_window < 2:
            raise ValueError(f"min_window must be >= 2, got {min_window}")
        super().__init__()
        self.delta = float(delta)
        self.min_window = int(min_window)
        self._window: list[float] = []

    @property
    def window(self) -> list[float]:
        """Return a copy of the active sliding window."""
        return list(self._window)

    @property
    def window_size(self) -> int:
        """Return the number of observations currently held in the window."""
        return len(self._window)

    @property
    def mean(self) -> float:
        """Return the mean of the current window (0.0 if empty)."""
        return float(np.mean(self._window)) if self._window else 0.0

    def reset(self) -> None:
        """Clear the active sliding window while preserving detection counters."""
        self._window.clear()

    def _candidate_cuts(self, n: int) -> np.ndarray:
        """Return candidate split indices [min_window, n - min_window]."""
        low = self.min_window
        high = n - self.min_window
        if n <= 4000:
            return np.arange(low, high + 1, dtype=np.int64)
        # Keep dense recent splits plus evenly spaced historical splits for bounded cost.
        recent_start = max(low, n - 500)
        coarse = np.unique(np.linspace(low, recent_start, 500, dtype=np.int64))
        recent = np.arange(recent_start, high + 1, dtype=np.int64)
        return np.unique(np.concatenate([coarse, recent]))

    def _find_best_cut(self) -> Optional[tuple[int, float, float]]:
        """Evaluate Hoeffding cut criterion and return (cut_idx, mu_0, mu_1) if triggered."""
        n = len(self._window)
        if n < 2 * self.min_window:
            return None

        arr = np.asarray(self._window, dtype=np.float64)
        value_range = float(arr.max() - arr.min())
        if value_range <= 1e-12:
            return None

        prefix_sums = np.cumsum(arr)
        total_sum = prefix_sums[-1]
        cuts = self._candidate_cuts(n)

        n0 = cuts.astype(np.float64)
        n1 = (n - cuts).astype(np.float64)
        s0 = prefix_sums[cuts - 1]
        mu0 = s0 / n0
        mu1 = (total_sum - s0) / n1

        harmonic_m = (n0 * n1) / float(n)
        delta_prime = self.delta / math.log(n)
        eps_cut = value_range * np.sqrt((1.0 / (2.0 * harmonic_m)) * math.log(4.0 / delta_prime))

        margins = np.abs(mu0 - mu1) - eps_cut
        best_idx = int(np.argmax(margins))
        if margins[best_idx] >= 0.0:
            return int(cuts[best_idx]), float(mu0[best_idx]), float(mu1[best_idx])
        return None

    def update(self, value: float) -> bool:
        """Append observation to window, evaluate Hoeffding cuts, and prune stale prefix."""
        numeric = self._validate_value(value)
        self._window.append(numeric)
        self.total_seen += 1

        first_cut = self._find_best_cut()
        if first_cut is None:
            return False

        dropped_elements: list[float] = []
        cut_result: Optional[tuple[int, float, float]] = first_cut
        while cut_result is not None:
            cut_idx, _, _ = cut_result
            dropped_elements.extend(self._window[:cut_idx])
            self._window = self._window[cut_idx:]
            cut_result = self._find_best_cut()

        pre_mean = float(np.mean(dropped_elements))
        post_mean = float(np.mean(self._window))
        self.last_pre_mean = pre_mean
        self.last_post_mean = post_mean
        self.last_magnitude = abs(post_mean - pre_mean)
        self.n_detections += 1
        return True


class PageHinkleyDetector(BaseDriftDetector):
    """Page-Hinkley cumulative-deviation test for streaming concept drift.

    Supports two-sided detection (``mode="both"``) as well as directional upward
    (``mode="up"``) and downward (``mode="down"``) change detection. Resets its
    cumulative deviation accumulators automatically upon detection.
    """

    name: str = "page_hinkley"

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1.0,
        min_instances: int = 30,
        mode: Literal["both", "up", "down"] = "both",
    ) -> None:
        if delta < 0.0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if min_instances < 1:
            raise ValueError(f"min_instances must be >= 1, got {min_instances}")
        if mode not in ("both", "up", "down"):
            raise ValueError(f"mode must be 'both', 'up', or 'down', got {mode!r}")

        super().__init__()
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.min_instances = int(min_instances)
        self.mode: Literal["both", "up", "down"] = mode
        self.last_direction: Optional[str] = None

        self._n: int = 0
        self._mean: float = 0.0
        self._sum: float = 0.0

        self._sum_up: float = 0.0
        self._min_up: float = 0.0
        self._pre_count_up: int = 0
        self._pre_sum_up: float = 0.0

        self._sum_down: float = 0.0
        self._max_down: float = 0.0
        self._pre_count_down: int = 0
        self._pre_sum_down: float = 0.0
        self._history: deque[float] = deque(maxlen=2000)

    @property
    def statistic_up(self) -> float:
        """Current upward Page-Hinkley statistic."""
        return self._sum_up - self._min_up

    @property
    def statistic_down(self) -> float:
        """Current downward Page-Hinkley statistic."""
        return self._max_down - self._sum_down

    def reset(self) -> None:
        """Reset running mean and cumulative deviation statistics."""
        self._n = 0
        self._mean = 0.0
        self._sum = 0.0
        self._sum_up = 0.0
        self._min_up = 0.0
        self._pre_count_up = 0
        self._pre_sum_up = 0.0
        self._sum_down = 0.0
        self._max_down = 0.0
        self._pre_count_down = 0
        self._pre_sum_down = 0.0
        self._history.clear()

    def _compute_segment_means(self, pre_count: int, pre_sum: float) -> tuple[float, float]:
        """Estimate pre-change and post-change means around the shift point."""
        n_hist = len(self._history)
        if n_hist >= 10:
            arr = np.asarray(self._history, dtype=np.float64)
            prefix = np.cumsum(arr)
            total = prefix[-1]
            cuts = np.arange(3, n_hist - 2)
            mu0 = prefix[cuts - 1] / cuts
            mu1 = (total - prefix[cuts - 1]) / (n_hist - cuts)
            bss = (cuts * (n_hist - cuts) / float(n_hist)) * (mu0 - mu1) ** 2
            best = int(np.argmax(bss))
            return float(mu0[best]), float(mu1[best])

        if pre_count <= 0 or pre_count >= self._n:
            return self._mean, self._mean
        pre_mean = pre_sum / pre_count
        post_mean = (self._sum - pre_sum) / (self._n - pre_count)
        return pre_mean, post_mean

    def update(self, value: float) -> bool:
        """Update cumulative statistics with one observation and test against threshold."""
        numeric = self._validate_value(value)
        self.total_seen += 1
        self._n += 1
        self._sum += numeric
        self._mean += (numeric - self._mean) / self._n
        self._history.append(numeric)

        self._sum_up = self.alpha * self._sum_up + (numeric - self._mean - self.delta)
        if self._sum_up <= self._min_up:
            self._min_up = self._sum_up
            self._pre_count_up = self._n
            self._pre_sum_up = self._sum

        self._sum_down = self.alpha * self._sum_down + (numeric - self._mean + self.delta)
        if self._sum_down >= self._max_down:
            self._max_down = self._sum_down
            self._pre_count_down = self._n
            self._pre_sum_down = self._sum

        if self._n < self.min_instances:
            return False

        ph_up = self.statistic_up
        ph_down = self.statistic_down

        up_hit = self.mode in ("up", "both") and ph_up > self.threshold
        down_hit = self.mode in ("down", "both") and ph_down > self.threshold
        if not (up_hit or down_hit):
            return False

        if up_hit and (not down_hit or ph_up >= ph_down):
            direction = "up"
            pre_mean, post_mean = self._compute_segment_means(self._pre_count_up, self._pre_sum_up)
        else:
            direction = "down"
            pre_mean, post_mean = self._compute_segment_means(self._pre_count_down, self._pre_sum_down)

        self.last_direction = direction
        self.last_pre_mean = float(pre_mean)
        self.last_post_mean = float(post_mean)
        self.last_magnitude = abs(float(post_mean) - float(pre_mean))
        self.n_detections += 1
        self.reset()
        return True
