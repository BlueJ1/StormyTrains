from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decomposed import (  # noqa: E402
    make_cancellation_then_class,
    make_cancellation_then_distribution,
    make_cancellation_then_ordinal,
    make_change_then_level,
    make_mixture_of_experts,
)
from metrics import CLASSES  # noqa: E402


class FrequencyClassifier:
    def __init__(self, numeric, categorical):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.shares_ = counts / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.shares_, (len(X), 1))


class MeanRegressor:
    fitted_rows = []

    def __init__(self, numeric, categorical):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.n_fit_ = len(y)
        self.mean_ = y.mean()
        type(self).fitted_rows.append(len(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _X(n=4):
    return pd.DataFrame({"x": np.arange(n, dtype=float)})


def test_cancellation_then_class_combines_stage_probabilities_in_metrics_order():
    X = _X()
    y = np.array(["canceled", "normal_minor", "moderate", "severe"])

    model = make_cancellation_then_class(["x"], [], FrequencyClassifier).fit(X, y)
    probabilities = model.predict_proba(X.iloc[:1])

    assert model.classes_.tolist() == CLASSES
    assert probabilities.shape == (1, 4)
    assert probabilities[0].tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert model.predict(X.iloc[:1]).tolist() == ["normal_minor"]


class FixedDistribution:
    def __init__(self, numeric, categorical):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y):
        self.fit_delays_ = np.asarray(y, dtype=float)
        self.mean_ = self.fit_delays_.mean()
        return self

    def predict_cdf(self, X, threshold):
        return np.full(len(X), 0.4 if threshold == 20 else 0.8)

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_cancellation_then_distribution_uses_strict_cutoffs_and_shared_mean():
    X = _X()
    severity = np.array(["canceled", "normal_minor", "moderate", "severe"])
    delays = np.array([np.nan, 10.0, 20.0, 60.0])

    model = make_cancellation_then_distribution(
        ["x"], [], FrequencyClassifier, FixedDistribution
    ).fit(X, severity, delay_y=delays)

    assert model.delay_model_.fit_delays_.tolist() == [10.0, 20.0, 60.0]
    assert model.classes_.tolist() == CLASSES
    assert model.predict_proba(X.iloc[:1])[0].tolist() == pytest.approx(
        [0.30, 0.30, 0.15, 0.25]
    )
    assert model.predict(X.iloc[:2]).tolist() == [30.0, 30.0]


def test_distribution_wrapper_requires_aligned_delay_target():
    with pytest.raises(TypeError, match="delay_y"):
        make_cancellation_then_distribution(
            ["x"], [], FrequencyClassifier, FixedDistribution
        ).fit(_X(), ["normal_minor"] * 4)


def test_distribution_wrapper_rejects_a_nonmonotone_cdf():
    class NonmonotoneDistribution(FixedDistribution):
        def predict_cdf(self, X, threshold):
            return np.full(len(X), 0.8 if threshold == 20 else 0.4)

    model = make_cancellation_then_distribution(
        ["x"], [], FrequencyClassifier, NonmonotoneDistribution
    ).fit(_X(), ["normal_minor", "moderate", "severe", "canceled"],
          delay_y=[0.0, 20.0, 70.0, np.nan])
    with pytest.raises(ValueError, match="nondecreasing"):
        model.predict_proba(_X(1))


class IndexedBinaryFactory:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def __call__(self, numeric, categorical):
        return FixedBinaryClassifier(next(self.probabilities))


class FixedBinaryClassifier:
    def __init__(self, positive_probability):
        self.positive_probability = positive_probability

    def fit(self, X, y):
        self.classes_ = np.array([False, True])
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive_probability)
        return np.column_stack((1 - positive, positive))


def test_ordinal_wrapper_cancels_then_projects_crossed_thresholds_to_midpoint():
    X = _X()
    y = np.array(["canceled", "normal_minor", "moderate", "severe"])
    factory = IndexedBinaryFactory([0.2, 0.4, 0.8])

    model = make_cancellation_then_ordinal(["x"], [], factory).fit(X, y)
    probabilities = model.predict_proba(X.iloc[:1])[0]

    # p20 and p60 meet at (0.4 + 0.8) / 2; completed mass is 0.8.
    assert model.classes_.tolist() == CLASSES
    assert probabilities.tolist() == pytest.approx([0.32, 0.0, 0.48, 0.20])
    assert probabilities.sum() == pytest.approx(1.0)


def test_mixture_of_experts_weights_each_class_regressor_by_probability():
    MeanRegressor.fitted_rows.clear()
    X = _X()
    delays = np.array([0.0, 0.0, 20.0, 100.0])

    model = make_mixture_of_experts(
        ["x"], [], FrequencyClassifier, MeanRegressor
    ).fit(X, delays)
    prediction = model.predict(_X(1))

    # Class means are 0, 20, and 100; class shares are 1/2, 1/4, 1/4.
    assert prediction.tolist() == pytest.approx([30.0])
    assert MeanRegressor.fitted_rows == [2, 1, 1]


def test_mixture_of_experts_rejects_missing_completed_class():
    with pytest.raises(ValueError, match="no rows for delay class 'severe'"):
        make_mixture_of_experts(
            ["x"], [], FrequencyClassifier, MeanRegressor
        ).fit(_X(2), [0.0, 20.0])


def test_change_model_uses_current_plus_delta_and_level_fallback_on_same_cohort():
    MeanRegressor.fitted_rows.clear()
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0], "current_delay": [90.0, 100.0, np.nan]})
    y = np.array([100.0, 120.0, 200.0])

    model = make_change_then_level(
        ["x", "current_delay"], [], MeanRegressor
    ).fit(X, y)
    prediction = model.predict(pd.DataFrame({
        "x": [3.0, 4.0], "current_delay": [50.0, np.nan]
    }))

    assert model.level_model_.n_fit_ == 3
    assert model.change_model_.n_fit_ == 2
    assert MeanRegressor.fitted_rows == [3, 2]
    assert prediction.tolist() == pytest.approx([65.0, 140.0])


def test_change_model_can_fall_back_for_all_rows_when_training_delay_is_missing():
    X = pd.DataFrame({"x": [0.0, 1.0], "current_delay": [np.nan, np.nan]})
    model = make_change_then_level(
        ["x", "current_delay"], [], MeanRegressor
    ).fit(X, [10.0, 20.0])

    assert model.change_model_ is None
    assert model.predict(X).tolist() == pytest.approx([15.0, 15.0])
