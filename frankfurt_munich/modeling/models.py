"""Model factories. Each returns a fresh, unfitted scikit-learn estimator.

A factory takes the numeric and categorical feature columns. All
preprocessing is inside the returned pipeline, so it is fitted on the
training rows of each fold only.

Linear models and SVR: missing values (only `current_delay`) become 0 plus a
0/1 missing indicator, numeric columns are standardised, and categorical
columns are one-hot encoded; categories not seen in training get all-zero
columns.

Tree models leave `current_delay` missing. The random forests get the same
one-hot columns (numeric columns unscaled); XGBoost and LightGBM use their
own categorical handling, with the training rows' levels and unseen levels
missing. Each tree model has a regressor (`delay`) and a classifier
(`severity`) factory with the same hyperparameters.

The tuned factories take their hyperparameters as keyword arguments, so
tuning.py can bind a sampled configuration with functools.partial. They run
single-threaded (n_jobs=1), so results do not depend on the number of cores.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

SEED = 0
RF_TREES = 300  # not tuned: more trees only cost time


def linear_preprocessing(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
            StandardScaler()), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def make_logistic_regression(numeric, categorical) -> Pipeline:
    """Multinomial logistic regression, scikit-learn's default L2 penalty (C=1)."""
    return Pipeline([("preprocess", linear_preprocessing(numeric, categorical)),
                     ("model", LogisticRegression(C=1.0, max_iter=1000))])


def make_linear_regression(numeric, categorical) -> Pipeline:
    """Ordinary least squares."""
    return Pipeline([("preprocess", linear_preprocessing(numeric, categorical)),
                     ("model", LinearRegression())])


def make_class_shares(numeric, categorical) -> DummyClassifier:
    """Reference: the training class shares for every journey."""
    return DummyClassifier(strategy="prior")


def make_training_mean(numeric, categorical) -> DummyRegressor:
    """Reference: the training mean delay for every journey."""
    return DummyRegressor(strategy="mean")


def make_svr(numeric, categorical, *, C, epsilon, gamma) -> Pipeline:
    """RBF support vector regression on the linear models' columns.

    The target is standardised inside the model, so C and epsilon are on the
    scale of standard deviations of the delay (epsilon 0.1 is about 2.5 min).
    """
    svr = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma, cache_size=1000)
    return Pipeline([("preprocess", linear_preprocessing(numeric, categorical)),
                     ("model", TransformedTargetRegressor(svr, transformer=StandardScaler()))])


def forest_preprocessing(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Numeric columns unchanged (missing stays missing), categorical one-hot."""
    return ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def make_random_forest(numeric, categorical, *, max_features, min_samples_leaf,
                       max_depth, max_samples) -> Pipeline:
    """Random forest of RF_TREES trees; missing `current_delay` stays missing."""
    forest = RandomForestRegressor(
        n_estimators=RF_TREES, max_features=max_features, min_samples_leaf=min_samples_leaf,
        max_depth=max_depth, max_samples=max_samples, n_jobs=1, random_state=SEED)
    return Pipeline([("preprocess", forest_preprocessing(numeric, categorical)),
                     ("model", forest)])


def make_random_forest_classifier(numeric, categorical, *, max_features, min_samples_leaf,
                                  max_depth, max_samples) -> Pipeline:
    """Random forest classifier (Gini splits) of RF_TREES trees; the class
    probabilities are the mean of the trees' leaf class shares."""
    forest = RandomForestClassifier(
        n_estimators=RF_TREES, max_features=max_features, min_samples_leaf=min_samples_leaf,
        max_depth=max_depth, max_samples=max_samples, n_jobs=1, random_state=SEED)
    return Pipeline([("preprocess", forest_preprocessing(numeric, categorical)),
                     ("model", forest)])


class CategoryCodes(BaseEstimator, TransformerMixin):
    """Categorical columns as pandas categories with the training rows' levels.

    Values are compared as strings; a level not seen in training becomes
    missing. Other columns pass through unchanged.
    """

    def __init__(self, categorical):
        self.categorical = categorical

    def fit(self, X, y=None):
        self.levels_ = {c: sorted(X[c].astype(str).unique()) for c in self.categorical}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.categorical:
            values = X[c].astype(str)
            X[c] = pd.Categorical(values.where(values.isin(self.levels_[c])),
                                  categories=self.levels_[c])
        return X


class IntegerLabels(ClassifierMixin, BaseEstimator):
    """Fit a classifier on the labels' positions in sorted order (0, 1, ...).

    XGBoost needs integer labels. `classes_` holds the original labels, so
    predict_proba's columns are in `classes_` order as usual.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self.estimator.fit(X, codes)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_xgboost(numeric, categorical, *, n_estimators, learning_rate, max_depth,
                 min_child_weight, subsample, colsample_bytree, reg_lambda,
                 reg_alpha) -> Pipeline:
    """XGBoost (hist trees, squared error) with native categorical splits."""
    booster = xgb.XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        min_child_weight=min_child_weight, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        tree_method="hist", enable_categorical=True, n_jobs=1, random_state=SEED)
    return Pipeline([("categories", CategoryCodes(categorical)), ("model", booster)])


def make_xgboost_classifier(numeric, categorical, *, n_estimators, learning_rate, max_depth,
                            min_child_weight, subsample, colsample_bytree, reg_lambda,
                            reg_alpha) -> Pipeline:
    """XGBoost multi-class classifier (softmax, log loss); otherwise as make_xgboost.

    Each boosting round fits one tree per class.
    """
    booster = xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        tree_method="hist", enable_categorical=True, n_jobs=1, random_state=SEED)
    return Pipeline([("categories", CategoryCodes(categorical)),
                     ("model", IntegerLabels(booster))])


def make_lightgbm(numeric, categorical, *, n_estimators, learning_rate, num_leaves,
                  min_child_samples, subsample, colsample_bytree, reg_lambda, reg_alpha,
                  cat_smooth) -> Pipeline:
    """LightGBM (squared error) with native categorical splits.

    subsample_freq=1, so that `subsample` takes effect (bagging every iteration).
    """
    booster = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        cat_smooth=cat_smooth, n_jobs=1, random_state=SEED, verbose=-1)
    return Pipeline([("categories", CategoryCodes(categorical)), ("model", booster)])


def make_lightgbm_classifier(numeric, categorical, *, n_estimators, learning_rate, num_leaves,
                             min_child_samples, subsample, colsample_bytree, reg_lambda,
                             reg_alpha, cat_smooth) -> Pipeline:
    """LightGBM multi-class classifier (softmax, log loss); otherwise as make_lightgbm.

    Each boosting round fits one tree per class.
    """
    booster = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=n_estimators, learning_rate=learning_rate,
        num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=subsample,
        subsample_freq=1, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
        reg_alpha=reg_alpha, cat_smooth=cat_smooth, n_jobs=1, random_state=SEED, verbose=-1)
    return Pipeline([("categories", CategoryCodes(categorical)), ("model", booster)])
