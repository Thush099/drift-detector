"""Integration and CLI tests for DriftDetectorPipeline and main.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import main as cli_main
from src.pipeline import DriftDetectorPipeline
from src.utils import generate_concept_drift_stream


class TestDriftDetectorPipeline:
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
