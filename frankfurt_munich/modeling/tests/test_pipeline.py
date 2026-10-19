"""Checks of the split and the cross-validation method, not of any model."""
from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from cv import PROBA_COLUMNS, class_probabilities, cross_validate
from data import DATA, PREDICTION_TIMES, TIME_COLUMN, load_rows
from features import WEATHER, feature_columns
from folds import N_FOLDS, cv_folds, read_assignment, week_of
from metrics import CLASSES, class_log_loss
from models import make_class_shares, make_linear_regression, make_logistic_regression


@cache
def rows_for(prediction_time):
    return load_rows(prediction_time)


@cache
def journeys(split):
    return pd.read_parquet(DATA / f"journeys_{split}.parquet")


def test_data_files_follow_the_committed_week_assignment():
    """Fails if the data were built with another split than split_weeks.csv."""
    weeks = read_assignment().set_index("week_start")
    for split in ("cv", "holdout"):
        week = week_of(journeys(split)[TIME_COLUMN])
        assert week.isin(weeks.index).all()
        assert week.map(weeks["split"]).eq(split).all()
    cv = journeys("cv")
    assert cv["fold"].astype(int).equals(week_of(cv[TIME_COLUMN]).map(weeks["fold"]).astype(int))


def test_holdout_and_folds_share_no_week():
    cv, holdout = journeys("cv"), journeys("holdout")
    assert not set(week_of(cv[TIME_COLUMN])) & set(week_of(holdout[TIME_COLUMN]))
    assert not set(cv["journey_id"]) & set(holdout["journey_id"])
    assert set(cv["fold"]) == set(range(1, N_FOLDS + 1))
    assert week_of(cv[TIME_COLUMN]).groupby(cv["fold"]).nunique().sum() == week_of(cv[TIME_COLUMN]).nunique()


def test_holdout_is_about_a_fifth_of_the_weeks():
    weeks = read_assignment()
    assert 0.15 < weeks["split"].eq("holdout").mean() < 0.25


def test_folds_have_similar_class_shares():
    cv = journeys("cv")
    shares = pd.crosstab(cv["fold"], cv["severity_class"], normalize="index")
    overall = cv["severity_class"].value_counts(normalize=True)
    assert (shares - overall).abs().max().max() < 0.02  # whole weeks: about 1 percentage point


@pytest.mark.parametrize("prediction_time", PREDICTION_TIMES)
def test_row_folds_match_the_journey_file(prediction_time):
    rows = rows_for(prediction_time)
    assert rows["fold"].equals(rows["journey_id"].map(journeys("cv").set_index("journey_id")["fold"]))
    assert rows["fold"].notna().all()


def test_rows_are_one_per_journey_and_complete():
    for prediction_time in PREDICTION_TIMES:
        rows = rows_for(prediction_time)
        assert rows["journey_id"].is_unique
        assert rows["train_type"].notna().all()


def test_current_delay_only_from_1h():
    for prediction_time in PREDICTION_TIMES:
        numeric, categorical = feature_columns(prediction_time)
        assert ("current_delay" in numeric) == (prediction_time in ("1h", "20min", "0min"))


def test_weather_rows_are_the_prediction_rows():
    """The weather file adds columns only: same rows, same order, same values."""
    for split in ("cv", "holdout"):
        rows = pd.read_parquet(DATA / f"prediction_rows_{split}.parquet")
        weather = pd.read_parquet(DATA / f"prediction_rows_weather_{split}.parquet")
        pd.testing.assert_frame_equal(weather[rows.columns], rows)


@pytest.mark.parametrize("prediction_time", PREDICTION_TIMES)
def test_weather_features_are_complete(prediction_time):
    rows = rows_for(prediction_time)
    assert rows[WEATHER].notna().all().all()
    assert set(rows["snow_on_route_24h"]) <= {0.0, 1.0}


def test_weather_is_not_from_after_the_prediction_time():
    """The weather hour is at or before `predicted_at` (German local time)."""
    for prediction_time in PREDICTION_TIMES:
        rows = rows_for(prediction_time)
        # Clock-change hours: take the earlier possible moment, the stricter check.
        predicted_at = rows["predicted_at"].dt.tz_localize(
            "Europe/Berlin", ambiguous=np.ones(len(rows), dtype=bool),
            nonexistent="shift_backward")
        assert (rows["weather_hour_utc"] <= predicted_at).all()


def test_weather_features_only_when_asked():
    for prediction_time in PREDICTION_TIMES:
        railway, categorical = feature_columns(prediction_time)
        weather, categorical_weather = feature_columns(prediction_time, weather=True)
        assert not set(WEATHER) & set(railway)
        assert weather == railway + WEATHER
        assert categorical_weather == categorical


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
    for (fold, train, validation), score in zip(cv_folds(rows), metrics["log_loss"]):
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


@pytest.mark.parametrize("weather", [False, True])
def test_shuffled_labels_do_not_beat_class_shares(weather):
    """With labels unrelated to the features, the full model can't learn anything."""
    rows = rows_for("1h").copy()
    rows["severity_class"] = np.random.default_rng(0).permutation(rows["severity_class"].to_numpy())
    numeric, categorical = feature_columns("1h", weather)
    model, _ = cross_validate(rows, "1h", "severity", make_logistic_regression, numeric, categorical)
    shares, _ = cross_validate(rows, "1h", "severity", make_class_shares, numeric, categorical)
    assert model["log_loss"].mean() > shares["log_loss"].mean() - 0.005
