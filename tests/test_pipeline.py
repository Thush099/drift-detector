"""Integration and CLI tests for DriftDetectorPipeline and main.py."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import main as cli_main
from src.pipeline import DriftDetectorPipeline
from src.utils import generate_concept_drift_stream


class TestDriftDetectorPipeline:
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"adwin_delta": 0.0}, "adwin_delta must be in (0, 1)"),
            ({"adwin_min_window": 1}, "adwin_min_window must be >= 2"),
            ({"ph_threshold": 0.0}, "ph_threshold must be finite and positive"),
            ({"ph_alpha": 1.1}, "ph_alpha must be in (0, 1]"),
            ({"min_retrain_samples": 0}, "min_retrain_samples must be >= 1"),
            ({"learning_rate": -0.1}, "learning_rate must be finite and positive"),
        ],
    )
    def test_invalid_detector_parameters_fail_fast(
        self, options: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(message)):
            DriftDetectorPipeline(**options)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("bad_array", "expected_message"),
        [
            ("features", r"X contains a non-finite value at row 3, column 1"),
            ("target", r"y contains a non-finite value at row 3"),
        ],
    )
    def test_non_finite_stream_values_fail_with_location(
        self, bad_array: str, expected_message: str
    ) -> None:
        X = np.arange(40, dtype=np.float64).reshape(20, 2)
        y = np.arange(20, dtype=np.float64)
        if bad_array == "features":
            X[3, 1] = np.nan
        else:
            y[3] = np.inf

        pipeline = DriftDetectorPipeline(detector="adwin", warmup_size=10)
        with pytest.raises(ValueError, match=expected_message):
            pipeline.run(X, y)

    def test_stationary_stream_no_false_detections(self) -> None:
        X, y = generate_concept_drift_stream(
            data_size=800,
            drift_at=800,
            shift_magnitude=0.0,
            seed=42,
        )
        pipeline = DriftDetectorPipeline(detector="both", warmup_size=100, seed=42)
        result = pipeline.run(X, y)

        assert result.events == []
        assert result.runs["adwin"].n_retrains == 0
        assert result.runs["page_hinkley"].n_retrains == 0
        assert result.adaptive_overall_mae < 0.3

    @pytest.mark.parametrize("detector", ["adwin", "page_hinkley", "both"])
    def test_abrupt_drift_retraining_outperforms_static(self, detector: str) -> None:
        X, y = generate_concept_drift_stream(
            data_size=1000,
            drift_at=500,
            shift_magnitude=3.0,
            drift_type="abrupt",
            seed=42,
        )
        pipeline = DriftDetectorPipeline(detector=detector, warmup_size=100, seed=42)  # type: ignore[arg-type]
        result = pipeline.run(X, y, drift_at=500)

        assert len(result.events) >= (2 if detector == "both" else 1)
        for ev in result.events:
            assert ev.step >= 500
            assert ev.magnitude > 0.5

        for run in result.runs.values():
            assert run.n_retrains >= 1
            assert run.post_drift_mae < result.static_post_drift_mae * 0.35
            assert run.post_drift_mse < result.static_post_drift_mse * 0.25

    def test_gradual_drift_triggers_both_and_improves_error(self) -> None:
        X, y = generate_concept_drift_stream(
            data_size=1000,
            drift_at=500,
            shift_magnitude=3.0,
            drift_type="gradual",
            seed=42,
        )
        pipeline = DriftDetectorPipeline(detector="both", warmup_size=100, seed=42)
        result = pipeline.run(X, y, drift_at=500)

        detectors_fired = {ev.detector for ev in result.events}
        assert detectors_fired == {"adwin", "page_hinkley"}
        for run in result.runs.values():
            assert run.post_drift_mae < result.static_post_drift_mae * 0.65
            assert run.post_drift_mse < result.static_post_drift_mse * 0.45

    def test_edge_cases_and_serialization(self) -> None:
        X = np.zeros((50, 2))
        y = np.zeros(50)
        pipeline = DriftDetectorPipeline(detector="adwin", warmup_size=100)
        with pytest.raises(ValueError):
            pipeline.run(X, y)

        X_good, y_good = generate_concept_drift_stream(data_size=300, drift_at=150, seed=7)
        res = DriftDetectorPipeline(detector="both", warmup_size=50, seed=7).run(X_good, y_good, drift_at=150)
        payload = res.to_dict()
        assert payload["detector_mode"] == "both"
        assert "static" in payload and "adaptive" in payload and "runs" in payload
        summary = res.format_summary()
        assert "static_baseline" in summary
        assert "adaptive (adwin)" in summary
        assert "adaptive (page_hinkley)" in summary


class TestCLI:
    def test_cli_main_function(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = cli_main.main(
            [
                "--detector",
                "both",
                "--data-size",
                "600",
                "--drift-at",
                "300",
                "--shift-magnitude",
                "3.0",
                "--drift-type",
                "abrupt",
                "--seed",
                "42",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Detected Drift Events:" in captured
        assert "adaptive (adwin)" in captured
        assert "adaptive (page_hinkley)" in captured

    def test_cli_subprocess_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "main.py", "--detector", "both"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "static_baseline" in proc.stdout
        assert "Detected Drift Events:" in proc.stdout


class TestFailOnExitCode:
    @staticmethod
    def _result(events: list) -> "object":
        from src.pipeline import PipelineResult

        return PipelineResult(
            detector_mode="both",
            data_size=10,
            warmup_size=5,
            drift_at=None,
            static_overall_mae=0.0,
            static_overall_mse=0.0,
            static_post_drift_mae=0.0,
            static_post_drift_mse=0.0,
            adaptive_overall_mae=0.0,
            adaptive_overall_mse=0.0,
            adaptive_post_drift_mae=0.0,
            adaptive_post_drift_mse=0.0,
            events=events,
        )

    def test_fail_on_none_always_succeeds(self) -> None:
        from src.detectors import DriftEvent

        event = DriftEvent(step=1, detector="adwin", pre_mean=0.0, post_mean=1.0, magnitude=1.0)
        assert cli_main.exit_code_for(self._result([event]), "none") == 0

    def test_fail_on_drift_gates_on_events(self) -> None:
        from src.detectors import DriftEvent

        event = DriftEvent(step=1, detector="adwin", pre_mean=0.0, post_mean=1.0, magnitude=1.0)
        assert cli_main.exit_code_for(self._result([event]), "drift") == 1
        assert cli_main.exit_code_for(self._result([]), "drift") == 0

    def test_fail_on_rejects_unknown_level(self) -> None:
        with pytest.raises(ValueError, match="fail_on must be one of"):
            cli_main.exit_code_for(self._result([]), "bogus")

    def test_cli_fail_on_drift_flags_injected_drift(self) -> None:
        code = cli_main.main(
            [
                "--detector",
                "both",
                "--data-size",
                "600",
                "--drift-at",
                "300",
                "--shift-magnitude",
                "3.0",
                "--drift-type",
                "abrupt",
                "--seed",
                "42",
                "--fail-on",
                "drift",
            ]
        )
        assert code == 1

    def test_cli_fail_on_drift_passes_on_stationary_stream(self) -> None:
        code = cli_main.main(
            [
                "--detector",
                "both",
                "--data-size",
                "600",
                "--drift-at",
                "0",
                "--shift-magnitude",
                "0.0",
                "--seed",
                "42",
                "--fail-on",
                "drift",
            ]
        )
        assert code == 0
