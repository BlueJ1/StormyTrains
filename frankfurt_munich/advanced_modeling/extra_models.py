"""Additional tabular estimators and robust delay targets for Phase 6."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import _paths  # noqa: F401  shared modelling code and dataset builders
from models import SEED


def string_categories(X, *, categorical):
    X = X.copy()
    for column in categorical:
        X[column] = X[column].astype("string").fillna("unknown").astype(object)
    return X


def make_catboost(numeric, categorical, *, task, iterations, depth, learning_rate,
                  l2_leaf_reg, random_strength, bagging_temperature, seed=SEED):
    from catboost import CatBoostClassifier, CatBoostRegressor

    model = CatBoostClassifier if task == "severity" else CatBoostRegressor
    return make_pipeline(
        FunctionTransformer(string_categories, kw_args={"categorical": categorical}),
        model(iterations=iterations, depth=depth, learning_rate=learning_rate,
              l2_leaf_reg=l2_leaf_reg, random_strength=random_strength,
              bagging_temperature=bagging_temperature, bootstrap_type="Bayesian",
              cat_features=categorical,
              thread_count=1, random_seed=seed, verbose=False, allow_writing_files=False))


def make_ebm(numeric, categorical, *, task):
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

    model = ExplainableBoostingClassifier if task == "severity" else ExplainableBoostingRegressor
    features = numeric + categorical
    return make_pipeline(
        FunctionTransformer(string_categories, kw_args={"categorical": categorical}),
        model(feature_names=features,
              feature_types=["continuous"] * len(numeric) + ["nominal"] * len(categorical),
              interactions=10, validation_size=0, max_rounds=1500, outer_bags=1,
              n_jobs=1, random_state=SEED))


class ShiftedLog(TransformerMixin, BaseEstimator):
    """Shift the smallest training target to zero before taking log1p."""

    def fit(self, y, *_):
        self.shift_ = max(0.0, -float(np.min(y)))
        return self

    def transform(self, y):
        return np.log1p(y + self.shift_)

    def inverse_transform(self, y):
        return np.expm1(y) - self.shift_


def log_shifted(factory, numeric, categorical):
    return TransformedTargetRegressor(regressor=factory(numeric, categorical),
                                      transformer=ShiftedLog())
