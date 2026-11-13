"""Unit tests for ADWINDetector and PageHinkleyDetector."""

from __future__ import annotations

import numpy as np
import pytest

from src.detectors import ADWINDetector, DriftEvent, PageHinkleyDetector


class TestADWINDetector:
    def test_stationary_stream_zero_false_positives(self) -> None:
        rng = np.random.default_rng(42)
        detector = ADWINDetector(delta=0.002, min_window=10)
        stream = rng.uniform(0.0, 1.0, size=1000)
        detections = [i for i, val in enumerate(stream) if detector.update(val)]
        assert detections == []
        assert detector.n_detections == 0
        assert detector.window_size == 1000

    def test_abrupt_shift_detection_and_window_pruning(self) -> None:
        rng = np.random.default_rng(7)
        detector = ADWINDetector(delta=0.002, min_window=10)
        pre = rng.normal(0.2, 0.15, size=300)
        post = rng.normal(2.5, 0.15, size=300)
        stream = np.concatenate([pre, post])

        detections = []
        window_size_at_detection = None
        for i, val in enumerate(stream):
            if detector.update(val):
                detections.append(i)
                if window_size_at_detection is None:
                    window_size_at_detection = detector.window_size

        assert len(detections) >= 1
        first_hit = detections[0]
        assert 300 <= first_hit < 360
        assert window_size_at_detection is not None and window_size_at_detection < 100
        assert detector.window_size <= 310
        assert detector.last_pre_mean is not None and detector.last_pre_mean < 0.5
        assert detector.last_post_mean is not None and detector.last_post_mean > 2.0
        assert detector.last_magnitude is not None and detector.last_magnitude > 1.5

    def test_gradual_ramp_detection(self) -> None:
        rng = np.random.default_rng(123)
        detector = ADWINDetector(delta=0.002, min_window=10)
        pre = rng.normal(0.0, 0.3, size=300)
        ramp = rng.normal(np.linspace(0.0, 3.0, 150), 0.3)
        post = rng.normal(3.0, 0.3, size=250)
        stream = np.concatenate([pre, ramp, post])

        detections = [i for i, val in enumerate(stream) if detector.update(val)]
        assert any(300 <= step < 500 for step in detections)

    def test_constant_stream_and_reset(self) -> None:
        detector = ADWINDetector(delta=0.002, min_window=10)
        for _ in range(100):
            assert not detector.update(1.5)
        assert detector.window_size == 100
        assert detector.mean == pytest.approx(1.5)

        detector.reset()
        assert detector.window_size == 0
        assert detector.mean == 0.0

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            ADWINDetector(delta=1.5)
        with pytest.raises(ValueError):
            ADWINDetector(min_window=1)

        detector = ADWINDetector()
        with pytest.raises(ValueError):
            detector.update(float("nan"))
        with pytest.raises(ValueError):
            detector.update(float("inf"))


class TestPageHinkleyDetector:
    def test_stationary_stream_minimal_false_detections(self) -> None:
        rng = np.random.default_rng(99)
        detector = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=1.0, min_instances=30)
        stream = np.abs(rng.normal(0.0, 0.25, size=1000))
        detections = [i for i, val in enumerate(stream) if detector.update(val)]
        assert detections == []

    def test_abrupt_upward_and_downward_two_sided(self) -> None:
        rng = np.random.default_rng(11)
        up_stream = np.concatenate([rng.normal(0.2, 0.1, 200), rng.normal(2.5, 0.1, 200)])
        det_up = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=1.0, min_instances=30, mode="both")
        hits_up = [i for i, val in enumerate(up_stream) if det_up.update(val)]
        assert len(hits_up) >= 1
        assert 200 <= hits_up[0] < 245
        assert det_up.last_direction == "up"
        assert det_up.last_pre_mean is not None and det_up.last_pre_mean < 0.5
        assert det_up.last_post_mean is not None and det_up.last_post_mean > 2.0

        down_stream = np.concatenate([rng.normal(2.5, 0.1, 200), rng.normal(0.2, 0.1, 200)])
        det_down = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=1.0, min_instances=30, mode="both")
        hits_down = [i for i, val in enumerate(down_stream) if det_down.update(val)]
        assert len(hits_down) >= 1
        assert 200 <= hits_down[0] < 245
        assert det_down.last_direction == "down"

    def test_directional_mode_ignores_opposite_shift(self) -> None:
        rng = np.random.default_rng(5)
        down_stream = np.concatenate([rng.normal(3.0, 0.1, 200), rng.normal(0.2, 0.1, 200)])
        det_up_only = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=1.0, min_instances=30, mode="up")
        hits = [i for i, val in enumerate(down_stream) if det_up_only.update(val)]
        assert hits == []

    def test_gradual_ramp_detection(self) -> None:
        rng = np.random.default_rng(42)
        pre = rng.normal(0.2, 0.15, size=250)
        ramp = rng.normal(np.linspace(0.2, 3.0, 120), 0.15)
        post = rng.normal(3.0, 0.15, size=200)
        stream = np.concatenate([pre, ramp, post])

        detector = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=1.0, min_instances=30)
        hits = [i for i, val in enumerate(stream) if detector.update(val)]
        assert any(250 <= step < 420 for step in hits)

    def test_min_instances_and_reset(self) -> None:
        detector = PageHinkleyDetector(delta=0.0, threshold=1.0, alpha=1.0, min_instances=30)
        for _ in range(29):
            assert not detector.update(10.0)
        detector.reset()
        assert detector.statistic_up == 0.0
        assert detector.statistic_down == 0.0

    def test_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            PageHinkleyDetector(delta=-0.1)
        with pytest.raises(ValueError):
            PageHinkleyDetector(threshold=0.0)
        with pytest.raises(ValueError):
            PageHinkleyDetector(alpha=1.5)
        with pytest.raises(ValueError):
            PageHinkleyDetector(mode="invalid")  # type: ignore[arg-type]

        detector = PageHinkleyDetector()
        with pytest.raises(ValueError):
            detector.update(float("nan"))

    def test_drift_event_serialization(self) -> None:
        ev = DriftEvent(step=512, detector="adwin", pre_mean=0.15, post_mean=3.85, magnitude=3.70)
        payload = ev.to_dict()
        assert payload == {
            "step": 512,
            "detector": "adwin",
            "pre_mean": 0.15,
            "post_mean": 3.85,
            "magnitude": 3.70,
        }
