"""Model factories. Each returns a fresh, unfitted scikit-learn estimator.

A factory takes the numeric and categorical feature columns. All
preprocessing is inside the returned pipeline, so it is fitted on the
training rows of each fold only.

Linear models: missing values (only `current_delay`) become 0 plus a 0/1
missing indicator, numeric columns are standardised, and categorical columns
are one-hot encoded; categories not seen in training get all-zero columns.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
