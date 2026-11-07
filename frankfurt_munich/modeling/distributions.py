"""Distributional delay estimators for Phase 6.5.

Each factory returns a scikit-learn-style regressor with ``fit``, ``predict``
(the fitted conditional mean), and ``predict_cdf(X, cutoff)`` (the conditional
probability of a delay strictly below ``cutoff``). CDFs from NGBoost and
LightGBMLSS are continuous, so their CDF values equal the strict probability;
the quantile forest adapter explicitly requests the package's strict rank.

Inputs use the same numeric passthrough and dense one-hot encoding as the
project's forest models. Optional distribution packages are imported only in
``fit`` so importing this module does not require all three packages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from models import SEED, forest_preprocessing


def _as_target(y, n_rows: int) -> np.ndarray:
    target = np.asarray(y, dtype=float)
    if target.ndim != 1 or len(target) != n_rows or not np.isfinite(target).all():
        raise ValueError("delay target must be one finite value per training row")
    return target


def _as_cutoff(cutoff: float) -> float:
    value = np.asarray(cutoff, dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError("cutoff must be a finite scalar")
    return float(value)


class _PreprocessedDelayRegressor(RegressorMixin, BaseEstimator):
    """Shared fitted feature preprocessing and input validation."""

    def _fit_preprocessor(self, X, y, *, impute_missing=False):
        target = _as_target(y, len(X))
        self.preprocessor_ = forest_preprocessing(self.numeric, self.categorical,
                                                   impute_missing=impute_missing)
        transformed = self.preprocessor_.fit_transform(X, target)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_transformed_features_ = transformed.shape[1]
        return np.asarray(transformed), target

    def _transform(self, X):
        check_is_fitted(self, "preprocessor_")
        return np.asarray(self.preprocessor_.transform(X))


class NGBoostDelayRegressor(_PreprocessedDelayRegressor):
    """NGBoost with a Normal response distribution."""

    def __init__(self, numeric, categorical, n_estimators=500, learning_rate=0.01,
                 base_max_depth=3, random_state=SEED):
        self.numeric = numeric
        self.categorical = categorical
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_max_depth = base_max_depth
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
        except ImportError as exc:
            raise ImportError("NGBoostDelayRegressor requires ngboost") from exc
        from sklearn.tree import DecisionTreeRegressor

        features, target = self._fit_preprocessor(X, y)
        self.model_ = NGBRegressor(
            Dist=Normal,
            Base=DecisionTreeRegressor(max_depth=self.base_max_depth,
                                       random_state=self.random_state),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=False,
        ).fit(features, target)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(self._transform(X)), dtype=float)

    def predict_cdf(self, X, cutoff):
        threshold = _as_cutoff(cutoff)
        check_is_fitted(self, "model_")
        features = self._transform(X)
        distribution = self.model_.pred_dist(features)
        return np.asarray(distribution.cdf(np.full(len(features), threshold)), dtype=float)


class QuantileForestDelayRegressor(_PreprocessedDelayRegressor):
    """Quantile regression forest with an empirical strict conditional CDF."""

    def __init__(self, numeric, categorical, n_estimators=300, min_samples_leaf=5,
                 max_features=1.0, max_depth=None, max_samples=None, random_state=SEED):
        self.numeric = numeric
        self.categorical = categorical
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from quantile_forest import RandomForestQuantileRegressor
        except ImportError as exc:
            raise ImportError("QuantileForestDelayRegressor requires quantile-forest") from exc

        features, target = self._fit_preprocessor(X, y, impute_missing=True)
        self.model_ = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_depth=self.max_depth,
            max_samples=self.max_samples,
            n_jobs=1,
            random_state=self.random_state,
        ).fit(features, target)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(self._transform(X), quantiles="mean"), dtype=float)

    def predict_cdf(self, X, cutoff):
        threshold = _as_cutoff(cutoff)
        check_is_fitted(self, "model_")
        features = self._transform(X)
        cutoff_values = np.full(len(features), threshold)
        return np.asarray(
            self.model_.quantile_ranks(features, cutoff_values, kind="strict"),
            dtype=float,
        )


class LightGBMLSSDelayRegressor(_PreprocessedDelayRegressor):
    """LightGBMLSS Gaussian mixture for real-valued delays.

    LightGBMLSS 0.6.1 currently fixes its LightGBM random seed to 123 in its
    ``set_params`` method, so this adapter intentionally exposes no seed arg.
    """

    def __init__(self, numeric, categorical, n_estimators=100, learning_rate=0.05,
                 num_leaves=15, min_data_in_leaf=20, mixture_components=2):
        self.numeric = numeric
        self.categorical = categorical
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.mixture_components = mixture_components

    def fit(self, X, y):
        try:
            import lightgbm as lgb
            from lightgbmlss.distributions.Gaussian import Gaussian
            from lightgbmlss.distributions.Mixture import Mixture
            from lightgbmlss.model import LightGBMLSS
        except ImportError as exc:
            raise ImportError("LightGBMLSSDelayRegressor requires lightgbmlss and lightgbm") from exc

        features, target = self._fit_preprocessor(X, y)
        # Fit the Gaussian mixture on the training fold's scale. Its default
        # starting values are too narrow for raw delays in minutes.
        self.target_mean_ = float(target.mean())
        self.target_scale_ = max(float(target.std()), 1.0)
        target = (target - self.target_mean_) / self.target_scale_
        frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
        self.distribution_ = Mixture(Gaussian(), M=self.mixture_components)
        self.model_ = LightGBMLSS(self.distribution_)
        train_set = lgb.Dataset(frame, label=target, free_raw_data=False)
        params = {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "num_threads": 1,
        }
        self.model_.train(params, train_set, num_boost_round=self.n_estimators)
        return self

    def _predict_distribution(self, X):
        check_is_fitted(self, "model_")
        features = self._transform(X)
        frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
        import torch

        raw = self.model_.booster.predict(frame, raw_score=True, num_threads=1)
        raw = torch.as_tensor(raw + self.model_.start_values, dtype=torch.float32)
        loc, scale, logits = raw.split(self.mixture_components, dim=1)
        loc = self.distribution_.param_dict["loc"](loc) * self.target_scale_ + self.target_mean_
        scale = self.distribution_.param_dict["scale"](scale) * self.target_scale_
        # Gumbel draws are useful during training. The fitted categorical
        # probabilities give a distribution independent of prediction batch order.
        return self.distribution_.create_mixture_distribution([loc, scale, logits.softmax(dim=1)])

    def predict(self, X):
        distribution = self._predict_distribution(X)
        return distribution.mean.detach().cpu().numpy().astype(float, copy=False)

    def predict_cdf(self, X, cutoff):
        threshold = _as_cutoff(cutoff)
        distribution = self._predict_distribution(X)
        import torch

        values = torch.full(distribution.mean.shape, threshold, dtype=distribution.mean.dtype)
        return distribution.cdf(values).detach().cpu().numpy().astype(float, copy=False)


def make_ngboost(numeric, categorical, *, n_estimators=500, learning_rate=0.01,
                 base_max_depth=3, random_state=SEED):
    """Factory for NGBoost's Normal conditional delay distribution."""
    return NGBoostDelayRegressor(numeric, categorical, n_estimators, learning_rate,
                                 base_max_depth, random_state)


def make_quantile_forest(numeric, categorical, *, n_estimators=300, min_samples_leaf=5,
                         max_features=1.0, max_depth=None, max_samples=None,
                         random_state=SEED):
    """Factory for quantile regression forest delay distributions."""
    return QuantileForestDelayRegressor(
        numeric, categorical, n_estimators, min_samples_leaf, max_features,
        max_depth, max_samples, random_state,
    )


def make_lightgbmlss(numeric, categorical, *, n_estimators=100, learning_rate=0.05,
                     num_leaves=15, min_data_in_leaf=20, mixture_components=2):
    """Factory for a LightGBMLSS Gaussian mixture delay distribution."""
    return LightGBMLSSDelayRegressor(
        numeric, categorical, n_estimators, learning_rate, num_leaves,
        min_data_in_leaf, mixture_components,
    )
