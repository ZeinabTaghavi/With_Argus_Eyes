"""
baselines.py
-------------

This module implements a collection of simple baseline models for the regression
task described in the project.  Baselines provide reference points to gauge
whether more complex models such as neural networks offer meaningful
improvements.  All baselines here implement a common interface with ``fit``
and ``predict`` methods, allowing them to be used interchangeably in the
``main.py`` experiment script.

The provided baselines include:

* ``MeanRegressor`` – Always predicts the mean of the training targets.
* ``LinearRegressor`` – Ordinary least squares linear regression with optional
  ℓ₂ regularisation (Ridge).
* ``RandomForestRegressorWrapper`` – A thin wrapper around scikit‑learn's
  ``RandomForestRegressor``.

Each baseline class exposes sensible hyperparameters with default values.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "scikit-learn is required for baseline models. Install via 'pip install scikit-learn'."
    ) from exc
try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The 'xgboost' package is required for the XGBoost baseline. Install via 'pip install xgboost'."
    ) from exc

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Constant Baselines
# ----------------------------------------------------------------------

class ConstantZeroRegressor:
    """Baseline that always predicts 0 as the rank score.
    
    This is a simple reference baseline that returns zero for all predictions.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantZeroRegressor":
        # No training needed for constant prediction
        logger.debug("Configured ConstantZeroRegressor")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=float)


class ConstantOneRegressor:
    """Baseline that always predicts 1 as the rank score.
    
    This is a simple reference baseline that returns one for all predictions.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantOneRegressor":
        # No training needed for constant prediction
        logger.debug("Configured ConstantOneRegressor")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0], dtype=float)


# ----------------------------------------------------------------------
# Cosine Similarity Baseline
# ----------------------------------------------------------------------

class CosineSimilarityRegressor:
    """Unsupervised baseline using cosine similarity between sentence embeddings.

    Assumes that the feature matrix X was created with the 'concat' feature_type
    in `load_stsb_embeddings`, i.e. X = [e1, e2, |e1-e2|, e1*e2].
    Only the original sentence embeddings e1 and e2 are used to compute cosine
    similarity, which is then mapped from [-1, 1] to [0, 1].
    """

    def __init__(self) -> None:
        self.embedding_dim_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CosineSimilarityRegressor":
        n_features = X.shape[1]
        if n_features % 4 != 0:
            raise ValueError(
                "CosineSimilarityRegressor expects X with 4*d features (e1,e2,|e1-e2|,e1*e2). "
                f"Got {n_features} features."
            )
        self.embedding_dim_ = n_features // 4
        logger.debug("Configured CosineSimilarityRegressor with embedding_dim=%d", self.embedding_dim_)
        # Unsupervised: labels y are not used.
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.embedding_dim_ is None:
            raise RuntimeError("CosineSimilarityRegressor has not been fitted yet.")
        d = self.embedding_dim_
        # Recover original embeddings e1 and e2 from the first 2*d features
        e1 = X[:, :d]
        e2 = X[:, d:2 * d]
        # Compute cosine similarity for each pair
        eps = 1e-8
        norm1 = np.linalg.norm(e1, axis=1) + eps
        norm2 = np.linalg.norm(e2, axis=1) + eps
        sim = np.sum(e1 * e2, axis=1) / (norm1 * norm2)
        # Map from [-1, 1] to [0, 1]
        scores = 0.5 * (sim + 1.0)
        return scores.astype(float)


@dataclass
class LinearRegressor:
    """Wrapper for Ridge regression (ℓ₂‑regularised linear regression).

    Parameters
    ----------
    alpha : float, optional
        Regularisation strength; must be a non‑negative float.  Larger values
        specify stronger regularisation.  When ``alpha=0``, this reduces to
        ordinary least squares.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.  If set to False,
        no intercept will be used in calculations.
    """

    alpha: float = 1.0
    fit_intercept: bool = True

    def __post_init__(self) -> None:
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        self.model.fit(X, y)
        logger.debug("Fitted LinearRegressor with alpha=%.3f", self.alpha)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ----------------------------------------------------------------------
# XGBoost and SVR Baseline Wrappers
# ----------------------------------------------------------------------


@dataclass
class XGBoostRegressorWrapper:
    """Wrapper around xgboost.XGBRegressor as a non-linear tree-based baseline."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: Optional[int] = 42
    n_jobs: Optional[int] = -1

    def __post_init__(self) -> None:
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="reg:squarederror",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressorWrapper":
        self.model.fit(X, y)
        logger.debug(
            "Fitted XGBoostRegressor with n_estimators=%d, max_depth=%s, learning_rate=%.3f",
            self.n_estimators,
            str(self.max_depth),
            self.learning_rate,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class SVRRegressorWrapper:
    """Support Vector Regression with an RBF kernel as a classic kernel baseline."""

    C: float = 1.0
    gamma: str | float = "scale"
    epsilon: float = 0.1
    kernel: str = "rbf"

    def __post_init__(self) -> None:
        self.model = SVR(C=self.C, gamma=self.gamma, epsilon=self.epsilon, kernel=self.kernel)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRRegressorWrapper":
        self.model.fit(X, y)
        logger.debug(
            "Fitted SVR with C=%.3f, gamma=%s, epsilon=%.3f, kernel=%s",
            self.C,
            str(self.gamma),
            self.epsilon,
            self.kernel,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class RandomForestRegressorWrapper:
    """Wrapper around scikit‑learn's RandomForestRegressor.

    Parameters
    ----------
    n_estimators : int, optional
        The number of trees in the forest.
    max_depth : int or None, optional
        The maximum depth of the trees.  If None, nodes are expanded until
        all leaves are pure or until all leaves contain fewer than
        ``min_samples_split`` samples.
    random_state : int or None, optional
        Controls both the randomness of the bootstrapping of the samples
        used when building trees (if ``bootstrap=True``) and the sampling
        of the features to consider when looking for the best split at each
        node.
    n_jobs : int or None, optional
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
    """

    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: Optional[int] = None
    n_jobs: Optional[int] = -1

    def __post_init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorWrapper":
        self.model.fit(X, y)
        logger.debug(
            "Fitted RandomForestRegressor with n_estimators=%d, max_depth=%s",
            self.n_estimators,
            str(self.max_depth),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def get_baseline_model(name: str, **kwargs: Any) -> Any:
    """Factory function to instantiate a baseline model by name.

    Parameters
    ----------
    name : {'zero', 'one', 'mean', 'cosine', 'linear', 'xgboost', 'svr', 'random_forest'}
        Identifier for the baseline model.  Aliases are case‑insensitive.
    kwargs : dict
        Additional keyword arguments passed to the model's constructor.

    Returns
    -------
    model : object
        An instance of the requested model.

    Raises
    ------
    ValueError
        If ``name`` is not one of the recognised baseline identifiers.
    """
    name_lower = name.lower()
    if name_lower in ["zero", "constant_zero", "zero_baseline"]:
        return ConstantZeroRegressor()
    if name_lower in ["one", "constant_one", "one_baseline"]:
        return ConstantOneRegressor()
    if name_lower in ["cosine", "cosine_similarity", "unsupervised"]:
        return CosineSimilarityRegressor()
    if name_lower in ["linear", "ridge"]:
        return LinearRegressor(**kwargs)
    if name_lower in ["xgboost", "xgb"]:
        return XGBoostRegressorWrapper(**kwargs)
    if name_lower in ["svr", "svr_rbf", "kernel"]:
        return SVRRegressorWrapper(**kwargs)
    if name_lower in ["random_forest", "rf", "forest"]:
        return RandomForestRegressorWrapper(**kwargs)
    raise ValueError(f"Unknown baseline model name: {name}")