"""Unit tests for OnlineRegressor and drift utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import OnlineRegressor, _ensure_2d
from src.utils import compute_regression_metrics, estimate_change_point


class TestEnsure2D:
    def test_1d_array_is_n_samples_of_one_feature(self) -> None:
        arr = _ensure_2d(np.array([1.0, 2.0, 3.0]))
        assert arr.shape == (3, 1)

    def test_2d_array_passes_through(self) -> None:
        arr = _ensure_2d(np.zeros((4, 2)))
        assert arr.shape == (4, 2)

    def test_3d_array_rejected(self) -> None:
        with pytest.raises(ValueError, match="1D or 2D"):
            _ensure_2d(np.zeros((2, 2, 2)))


class TestComputeRegressionMetrics:
    def test_perfect_predictions_are_zero_error(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(y, y)
        assert metrics == {"mae": 0.0, "mse": 0.0, "rmse": 0.0}

    def test_known_residuals(self) -> None:
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(1.0)
        assert metrics["mse"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(1.0)

    def test_rmse_is_sqrt_of_mse(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=50)
        y_pred = rng.normal(size=50)
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


class TestEstimateChangePoint:
    def test_clear_step_change_is_localized(self) -> None:
        errors = [0.0] * 20 + [10.0] * 20
        idx = estimate_change_point(errors, min_segment=5)
        assert 15 <= idx <= 25

    def test_short_series_returns_zero(self) -> None:
        assert estimate_change_point([1.0, 2.0], min_segment=5) == 0

    def test_flat_series_still_yields_valid_candidate(self) -> None:
        # No variance to separate: any cut ties, but the result must be a
        # usable candidate inside [min_segment, n - min_segment].
        n = 20
        idx = estimate_change_point([1.0] * n, min_segment=5)
        assert 5 <= idx <= n - 5


class TestOnlineRegressor:
    def test_predict_before_fit_raises(self) -> None:
        model = OnlineRegressor()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(np.zeros((1, 2)))

    def test_fit_rejects_empty_dataset(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            OnlineRegressor().fit(np.zeros((0, 2)), np.zeros(0))

    def test_fit_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="identical sample counts"):
            OnlineRegressor().fit(np.zeros((3, 2)), np.zeros(2))

    def test_fit_recovers_linear_relationship(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0
        model = OnlineRegressor(ridge_alpha=0.01).fit(X, y)
        preds = model.predict(X)
        assert compute_regression_metrics(y, preds)["mae"] < 0.5

    def test_partial_fit_without_fit_acts_as_fit(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 2))
        y = X.sum(axis=1)
        model = OnlineRegressor().partial_fit(X, y)
        assert model.predict(X).shape == (50,)

    def test_clone_is_independent(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(size=(30, 2))
        y = X[:, 0] + 2.0 * X[:, 1]
        model = OnlineRegressor().fit(X, y)
        clone = model.clone()
        assert clone.predict(X) == pytest.approx(model.predict(X))
        clone.partial_fit(X[:10], y[:10])
        # mutating the clone must not touch the original's weights
        assert clone._sgd.coef_ is not model._sgd.coef_
