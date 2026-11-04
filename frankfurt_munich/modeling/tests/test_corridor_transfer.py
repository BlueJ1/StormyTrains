from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import corridor_transfer as transfer
from folds import N_FOLDS
from metrics import CLASSES
from models import CategoryCodes


class SpyRegressor(BaseEstimator):
    def __init__(self, records):
        self.records = records

    def fit(self, X, y, init_model=None):
        markers = tuple(X["marker"].astype(int))
        category_levels = tuple(X["station"].cat.categories) if "station" in X else ()
        self.mean_ = float(np.mean(y))
        self.booster_ = {"fit": len(self.records)}
        self.records.append({"kind": "fit", "markers": markers,
                             "features": tuple(X.columns), "category_levels": category_levels,
                             "init_model": init_model})
        return self

    def predict(self, X):
        if "station" in X:
            assert isinstance(X["station"].dtype, pd.CategoricalDtype)
        markers = tuple(X["marker"].astype(int))
        fitted = self.records[-1]
        training_markers = set(fitted["markers"])
        self.records.append({"kind": "predict", "markers": markers,
                             "training_markers": training_markers,
                             "features": fitted["features"]})
        return np.full(len(X), self.mean_, dtype=float)


class SpyClassifier(SpyRegressor):
    def fit(self, X, y, init_model=None):
        markers = tuple(X["marker"].astype(int))
        category_levels = tuple(X["station"].cat.categories) if "station" in X else ()
        self.records.append({"kind": "fit", "markers": markers,
                             "features": tuple(X.columns), "category_levels": category_levels,
                             "init_model": init_model})
        self.classes_ = np.asarray(sorted(CLASSES))
        return self

    def predict_proba(self, X):
        fitted = self.records[-1]
        self.records.append({"kind": "predict", "markers": tuple(X["marker"].astype(int)),
                             "training_markers": set(fitted["markers"]),
                             "features": fitted["features"]})
        return np.tile(np.asarray([0.1, 0.2, 0.3, 0.4]), (len(X), 1))


def spy_factory(records):
    def make_model(numeric, categorical, **params):
        return Pipeline([("categories", CategoryCodes(categorical)),
                         ("model", SpyRegressor(records))])
    return make_model


def spy_classifier_factory(records):
    def make_model(numeric, categorical, **params):
        return Pipeline([("categories", CategoryCodes(categorical)),
                         ("model", SpyClassifier(records))])
    return make_model


def _rows(n=6):
    dates = pd.date_range("2025-01-06", periods=n, freq="7D")
    target = pd.DataFrame({
        "journey_id": [f"j{i}" for i in range(n)],
        "prediction_time": "1h",
        "origin_departure_planned_local": dates,
        "corridor": "target",
        "marker": np.arange(n, dtype=int),
        "station": ["A", "B", "A", "B", "A", "B"][:n],
        "arrival_delay_min": np.arange(n, dtype=float) * 10,
        "severity_class": ["normal_minor", "moderate", "severe", "canceled", "normal_minor", "moderate"][:n],
        "fold": [1, 1, 2, 2, 3, 3][:n],
    })
    return target


def _source_for(target):
    source = target.drop(columns="fold").copy()
    source["corridor"] = "other"
    return source


def _single_outer_split(rows):
    train = rows.index.to_series().lt(4)
    validation = ~train
    yield 1, train, validation


def _two_inner_splits(rows):
    first_valid = rows.index.to_series().lt(2)
    second_valid = rows.index.to_series().ge(2)
    yield 1, ~first_valid, first_valid
    yield 2, ~second_valid, second_valid


def test_source_training_is_week_limited_and_excludes_validation_journeys():
    target = _rows()
    source = _source_for(target)
    source["run_week"] = source.origin_departure_planned_local
    source["destination_week"] = source.origin_departure_planned_local
    source["outcome_week"] = source.origin_departure_planned_local
    target["run_week"] = target.origin_departure_planned_local
    target_train, target_valid = target.iloc[:4], target.iloc[4:]
    # Same physical validation journeys are assigned source-corridor origins in
    # training weeks, so the ID check must remove them despite week eligibility.
    source.loc[source.journey_id.eq("j4"), "origin_departure_planned_local"] = target.iloc[0][
        "origin_departure_planned_local"]
    source.loc[source.journey_id.eq("j5"), "origin_departure_planned_local"] = target.iloc[1][
        "origin_departure_planned_local"]
    # A second physical-run guard catches a cross-week origin when run_week is
    # present in both extracts.
    source.loc[source.journey_id.eq("j0"), "run_week"] = target_valid.iloc[0][
        "run_week"]
    source.loc[source.journey_id.eq("j2"), "outcome_week"] = target_valid.iloc[0][
        "origin_departure_planned_local"]

    eligible = transfer._source_training_rows(
        source, target_train, target_valid, "1h", "delay")

    assert set(eligible.journey_id) == {"j1", "j3"}
    assert set(eligible.journey_id).isdisjoint(set(target_valid.journey_id))
    assert eligible.journey_id.eq("j0").sum() == 0


def test_pooled_fit_adds_corridor_and_returns_cv_prediction_schema():
    target = _rows()
    source = _source_for(target)
    records = []
    metrics, predictions = transfer.evaluate_corridor_transfer(
        target, source, prediction_time="1h", task="delay", method="pooled",
        numeric=["marker"], categorical=["station"], params={},
        splits=_single_outer_split, factory=spy_factory(records),
    )

    assert metrics.loc[0, "method"] == "pooled"
    assert metrics.loc[0, "target_train_rows"] == 4
    assert metrics.loc[0, "source_train_rows"] == 4
    assert set(predictions.journey_id) == {"j4", "j5"}
    assert set(predictions.method) == {"pooled"}
    assert "corridor" in next(row for row in records if row["kind"] == "fit")["features"]


def test_severity_probabilities_follow_shared_class_order():
    target = _rows()
    source = _source_for(target)
    metrics, predictions = transfer.evaluate_corridor_transfer(
        target, source, prediction_time="1h", task="severity", method="pooled",
        numeric=["marker"], categorical=["station"], params={},
        splits=_single_outer_split, factory=spy_classifier_factory([]),
    )

    assert np.allclose(predictions[[f"p_{name}" for name in CLASSES]].iloc[0],
                       [0.3, 0.2, 0.4, 0.1])
    assert metrics.loc[0, "method"] == "pooled"


def test_init_model_uses_shared_categories_and_continues_source_booster():
    target = _rows()
    source = _source_for(target)
    records = []
    metrics, predictions = transfer.evaluate_corridor_transfer(
        target, source, prediction_time="1h", task="delay", method="init_model",
        numeric=["marker"], categorical=["station"], params={},
        splits=_single_outer_split, factory=spy_factory(records),
    )

    fits = [row for row in records if row["kind"] == "fit"]
    assert len(fits) == 2
    assert fits[0]["category_levels"] == fits[1]["category_levels"]
    assert fits[1]["init_model"] == {"fit": 0}
    assert metrics.loc[0, "source_train_rows"] == 4
    assert set(predictions.journey_id) == {"j4", "j5"}


def test_stacked_source_models_crossfit_and_exclude_validation_physical_ids():
    target = _rows()
    source = _source_for(target)
    # The validation ID is deliberately put into a different, allowed source
    # week. Its physical journey ID must still keep it out of that source fit.
    source.loc[source.journey_id.eq("j0"), "origin_departure_planned_local"] = target.iloc[2][
        "origin_departure_planned_local"]
    source.loc[source.journey_id.eq("j1"), "origin_departure_planned_local"] = target.iloc[3][
        "origin_departure_planned_local"]
    source.loc[source.journey_id.eq("j4"), "origin_departure_planned_local"] = target.iloc[2][
        "origin_departure_planned_local"]
    independent = _source_for(target)
    independent["journey_id"] = independent.journey_id.str.replace("j", "s", regex=False)
    independent["marker"] += 100
    source = pd.concat([source, independent], ignore_index=True)
    records = []
    metrics, predictions = transfer.evaluate_corridor_transfer(
        target, source, prediction_time="1h", task="delay", method="stacked",
        numeric=["marker"], categorical=["station"], params={},
        splits=_single_outer_split, inner_splitter=_two_inner_splits,
        factory=spy_factory(records),
    )

    source_predictions = [row for row in records if row["kind"] == "predict"
                          and "source_predicted_delay" not in row["features"]]
    assert len(source_predictions) == 3  # two inner source fits and one outer source fit
    assert all(set(row["markers"]).isdisjoint(row["training_markers"])
               for row in source_predictions)
    assert all(4 not in row["training_markers"] for row in source_predictions)
    assert metrics.loc[0, "method"] == "stacked"
    assert metrics.loc[0, "source_inner_fit_rows"] > 0
    assert set(predictions.journey_id) == {"j4", "j5"}
    assert "source_predicted_delay" in next(
        row for row in records if row["kind"] == "fit" and "source_predicted_delay" in row["features"]
    )["features"]


def test_stacked_default_outer_and_inner_week_folds_cover_target_cv_rows():
    n_weeks = 40
    target = _rows(n=6).iloc[:1].copy()
    target = pd.concat([target] * n_weeks, ignore_index=True)
    target["journey_id"] = [f"run-{i}" for i in range(n_weeks)]
    target["origin_departure_planned_local"] = pd.date_range(
        "2025-01-06", periods=n_weeks, freq="7D")
    target["marker"] = np.arange(n_weeks)
    target["fold"] = (np.arange(n_weeks) % N_FOLDS) + 1
    target["severity_class"] = [CLASSES[i % len(CLASSES)] for i in range(n_weeks)]
    target["arrival_delay_min"] = np.arange(n_weeks, dtype=float)
    source = _source_for(target)
    source["run_week"] = source.origin_departure_planned_local
    source["destination_week"] = source.origin_departure_planned_local
    source["outcome_week"] = source.origin_departure_planned_local
    records = []

    metrics, predictions = transfer.evaluate_corridor_transfer(
        target, source, prediction_time="1h", task="delay", method="stacked",
        numeric=["marker"], categorical=["station"], params={},
        factory=spy_factory(records),
    )

    assert len(metrics) == N_FOLDS
    assert len(predictions) == n_weeks
    assert set(predictions.journey_id) == set(target.journey_id)
    source_predictions = [row for row in records if row["kind"] == "predict"
                          and "source_predicted_delay" not in row["features"]]
    assert source_predictions
    assert all(set(row["markers"]).isdisjoint(row["training_markers"])
               for row in source_predictions)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be"):
        transfer.evaluate_corridor_transfer(
            _rows(), _source_for(_rows()), prediction_time="1h", task="delay",
            method="other", numeric=["marker"], categorical=["station"], params={},
        )


def test_real_lightgbm_init_model_smoke_uses_shared_categorical_schema():
    pytest.importorskip("lightgbm")
    params = {
        "n_estimators": 2, "learning_rate": 0.1, "num_leaves": 3,
        "min_child_samples": 1, "subsample": 1.0, "colsample_bytree": 1.0,
        "reg_lambda": 0.0, "reg_alpha": 0.0, "cat_smooth": 0.0,
    }
    metrics, predictions = transfer.evaluate_corridor_transfer(
        _rows(), _source_for(_rows()), prediction_time="1h", task="delay",
        method="init_model", numeric=["marker"], categorical=["station"],
        params=params, splits=_single_outer_split,
    )

    assert np.isfinite(predictions.predicted_delay).all()
    assert metrics.loc[0, "method"] == "init_model"
