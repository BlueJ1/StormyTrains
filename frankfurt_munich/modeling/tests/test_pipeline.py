"""Checks of the cross-validation method itself, not of any model."""
from __future__ import annotations

import json
from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from cv import PROBA_COLUMNS, class_probabilities, cross_validate, fold_splits
from data import DATA, HERE, LEAD, PREDICTION_TIMES, TIME_COLUMN, load_rows
from features import feature_columns
from folds import cv_folds, fold_bounds
from metrics import CLASSES, class_log_loss
from models import make_class_shares, make_linear_regression, make_logistic_regression


@cache
def rows_for(prediction_time):
    return load_rows(prediction_time)


def test_folds_match_cv_folds_json():
    saved = json.loads((HERE.parent / "cv_folds.json").read_text())
    journeys = pd.read_parquet(DATA / "journeys_cv.parquet")
    for bounds, entry, (fold, train, validation) in zip(fold_bounds(), saved, cv_folds(journeys)):
        assert fold == entry["fold"]
        for key in ("train_start", "train_end", "validation_start", "validation_end"):
            assert bounds[key] == pd.Timestamp(entry[key])
        assert train.sum() == entry["train_journeys"]
        assert validation.sum() == entry["validation_journeys"]


@pytest.mark.parametrize("prediction_time", PREDICTION_TIMES)
def test_training_rows_precede_validation_and_respect_gap(prediction_time):
    rows = rows_for(prediction_time)
    for bounds, (fold, train, validation) in zip(fold_bounds(), fold_splits(rows, prediction_time)):
        assert train.any() and validation.any()
        assert not (train & validation).any()
        assert rows.loc[train, TIME_COLUMN].max() < rows.loc[validation, TIME_COLUMN].min()
        cutoff = bounds["validation_start"] - LEAD[prediction_time]
        assert rows.loc[train, "outcome_known_at"].max() < cutoff
        # The gap only removes journeys close to the cut, never older ones.
        dropped = rows.loc[rows[TIME_COLUMN] < bounds["train_end"]].index.difference(rows.index[train])
        assert (rows.loc[dropped, TIME_COLUMN] >= cutoff - pd.Timedelta(hours=12)).all()


def test_rows_are_one_per_journey_and_complete():
    for prediction_time in PREDICTION_TIMES:
        rows = rows_for(prediction_time)
        assert rows["journey_id"].is_unique
        assert rows["outcome_known_at"].notna().all()
        assert rows["train_type"].notna().all()


def test_current_delay_only_from_1h():
    for prediction_time in PREDICTION_TIMES:
        numeric, categorical = feature_columns(prediction_time)
        assert ("current_delay" in numeric) == (prediction_time in ("1h", "20min", "0min"))


def test_holdout_needs_explicit_permission():
    with pytest.raises(PermissionError):
        load_rows("1h", split="holdout")


def test_class_log_loss_matches_sklearn():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(4), size=200)
    y = rng.choice(CLASSES, size=200)
    alphabetical = sorted(CLASSES)
    reordered = proba[:, [CLASSES.index(c) for c in alphabetical]]
    assert class_log_loss(y, proba) == pytest.approx(log_loss(y, reordered, labels=alphabetical))


def test_class_shares_score_is_the_hand_computed_cross_entropy():
    rows = rows_for("1h")
    metrics, _ = cross_validate(rows, "1h", "severity", make_class_shares, [], ["train_type"])
    for (fold, train, validation), score in zip(fold_splits(rows, "1h"), metrics["log_loss"]):
        shares = rows.loc[train, "severity_class"].value_counts(normalize=True)
        expected = -np.log(shares[rows.loc[validation, "severity_class"]]).mean()
        assert score == pytest.approx(expected)


def test_probability_columns_follow_classes():
    class Fixed:
        classes_ = np.array(sorted(CLASSES))  # canceled, moderate, normal_minor, severe

        def predict_proba(self, X):
            return np.tile([0.1, 0.2, 0.3, 0.4], (len(X), 1))

    proba = class_probabilities(Fixed(), [0, 0])
    assert proba[0].tolist() == [0.3, 0.2, 0.4, 0.1]  # normal_minor, moderate, severe, canceled


def test_validation_predictions_line_up_with_their_rows():
    """With the answer given as a feature, the models must be (almost) perfect."""
    rows = rows_for("1h").assign(answer_class=lambda r: r["severity_class"].astype(str),
                                 answer_delay=lambda r: r["arrival_delay_min"])
    metrics, predictions = cross_validate(rows, "1h", "severity", make_logistic_regression,
                                          [], ["answer_class"])
    assert (metrics["log_loss"] < 0.05).all()
    assert (metrics["macro_f1"] == 1).all()
    assert np.allclose(predictions[PROBA_COLUMNS].sum(axis=1), 1)
    metrics, _ = cross_validate(rows, "1h", "delay", make_linear_regression, ["answer_delay"], [])
    assert (metrics["rmse"] < 1e-6).all()


def test_shuffled_labels_do_not_beat_class_shares():
    """With labels unrelated to the features, the full model can't learn anything."""
    rows = rows_for("1h").copy()
    rows["severity_class"] = np.random.default_rng(0).permutation(rows["severity_class"].to_numpy())
    numeric, categorical = feature_columns("1h")
    model, _ = cross_validate(rows, "1h", "severity", make_logistic_regression, numeric, categorical)
    shares, _ = cross_validate(rows, "1h", "severity", make_class_shares, numeric, categorical)
    assert model["log_loss"].mean() > shares["log_loss"].mean() - 0.005
