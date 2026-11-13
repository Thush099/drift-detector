"""Online incremental linear regression model for concept drift adaptation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1) if arr.shape[0] > 1 else arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D feature array, got shape {arr.shape}")
    return arr


class OnlineRegressor:
    """Online linear regressor combining closed-form Ridge resets with SGD updates.

    Uses ``Ridge`` on warm-up and post-drift retraining windows for stable recovery,
    and ``SGDRegressor.partial_fit`` for step-by-step incremental updates between
    drift events.
    """

    def __init__(
        self,
        learning_rate: float = 0.0005,
        ridge_alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.ridge_alpha = float(ridge_alpha)
        self.seed = int(seed)
        self._sgd = self._new_sgd()
        self._is_fitted: bool = False
        self._n_features: int = 0

    def _new_sgd(self) -> SGDRegressor:
        return SGDRegressor(
            learning_rate="constant",
            eta0=self.learning_rate,
            penalty="l2",
            alpha=1e-4,
            random_state=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OnlineRegressor":
        """Fit initial or reset weights via regularized least squares."""
        X_arr = _ensure_2d(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must have identical sample counts")
        if len(X_arr) == 0:
            raise ValueError("Cannot fit OnlineRegressor on empty dataset")

        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(X_arr, y_arr)

        self._sgd = self._new_sgd()
        self._sgd.coef_ = np.asarray(ridge.coef_, dtype=np.float64).copy()
        self._sgd.intercept_ = np.array([float(ridge.intercept_)], dtype=np.float64)
        self._sgd.t_ = 1
        self._n_features = X_arr.shape[1]
        self._is_fitted = True
        return self

    def retrain(self, X: np.ndarray, y: np.ndarray) -> "OnlineRegressor":
        """Retrain the model cleanly on a post-drift window."""
        return self.fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "OnlineRegressor":
        """Perform one incremental stochastic gradient update."""
        X_arr = _ensure_2d(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if not self._is_fitted:
            return self.fit(X_arr, y_arr)
        self._sgd.partial_fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for a single sample or batch."""
        if not self._is_fitted:
            raise RuntimeError("OnlineRegressor must be fitted before calling predict()")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] == self._n_features:
                arr = arr.reshape(1, -1)
            else:
                arr = arr.reshape(-1, 1)
        return np.asarray(self._sgd.predict(arr), dtype=np.float64)

    def clone(self) -> "OnlineRegressor":
        """Return an independent copy with identical weights and hyperparameters."""
        copy_model = OnlineRegressor(
            learning_rate=self.learning_rate,
            ridge_alpha=self.ridge_alpha,
            seed=self.seed,
        )
        if self._is_fitted:
            copy_model._sgd.coef_ = self._sgd.coef_.copy()
            copy_model._sgd.intercept_ = self._sgd.intercept_.copy()
            copy_model._sgd.t_ = self._sgd.t_
            copy_model._n_features = self._n_features
            copy_model._is_fitted = True
        return copy_model


# Backwards-compatible alias
MachineLearningModel = OnlineRegressor
