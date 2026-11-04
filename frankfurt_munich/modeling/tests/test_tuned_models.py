"""Checks of the tuned models' preprocessing and of the random search, not of their scores."""
from __future__ import annotations

from functools import cache, partial

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from cv import class_probabilities, cross_validate
from data import load_rows
from features import feature_columns
from metrics import CLASSES
from models import (CategoryCodes, IntegerLabels, make_class_shares, make_lightgbm,
                    make_lightgbm_classifier, make_random_forest,
                    make_random_forest_classifier, make_svr, make_training_mean, make_xgboost,
                    make_xgboost_classifier)
from tuning import N_CONFIGS, SPACES, LogUniformInt, run_search, sample_configs, search

# Small configurations, so the tests run in seconds.
SMALL = {
    "random_forest": partial(make_random_forest, max_features=0.5, min_samples_leaf=5,
                             max_depth=None, max_samples=0.8),
    "xgboost": partial(make_xgboost, n_estimators=100, learning_rate=0.1, max_depth=4,
                       min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                       reg_lambda=1.0, reg_alpha=0.0),
    "lightgbm": partial(make_lightgbm, n_estimators=100, learning_rate=0.1, num_leaves=15,
                        min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
                        reg_lambda=0.0, reg_alpha=0.0, cat_smooth=10),
    "svr": partial(make_svr, C=1.0, epsilon=0.1, gamma=0.01),
}
SMALL_CLASSIFIERS = {
    "random_forest_classifier": partial(make_random_forest_classifier, max_features=0.5,
                                        min_samples_leaf=5, max_depth=None, max_samples=0.8),
    "xgboost_classifier": partial(make_xgboost_classifier, n_estimators=100, learning_rate=0.1,
                                  max_depth=4, min_child_weight=1, subsample=1.0,
                                  colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0),
    "lightgbm_classifier": partial(make_lightgbm_classifier, n_estimators=100,
                                   learning_rate=0.1, num_leaves=15, min_child_samples=20,
                                   subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0,
                                   reg_alpha=0.0, cat_smooth=10),
}


@cache
def severity_rows(prediction_time="1h", weather=True):
    return load_rows(prediction_time, weather=weather)


@cache
def delay_rows(prediction_time="1h", weather=True):
    rows = load_rows(prediction_time, weather=weather)
    return rows[rows["arrival_delay_min"].notna()].reset_index(drop=True)


def test_configurations_are_fixed_and_inside_their_space():
    for name, space in SPACES.items():
        configs = sample_configs(space)
        assert len(configs) == N_CONFIGS
        assert configs == sample_configs(space)  # same for railway-only and weather runs
        for config in configs:
            assert set(config) == set(space)
            for key, value in config.items():
                dist = space[key]
                if isinstance(dist, list):
                    assert value in dist
                elif isinstance(dist, LogUniformInt):
                    assert isinstance(value, int) and dist.low <= value <= dist.high
                else:
                    assert dist.support()[0] <= value <= dist.support()[1]


@pytest.mark.parametrize("model", list(SMALL))
def test_unseen_categories_and_missing_current_delay_give_predictions(model):
    rows = delay_rows()
    numeric, categorical = feature_columns("1h", weather=True)
    train, test = rows[rows["fold"] != 1], rows[rows["fold"] == 1].head(50).copy()
    test.iloc[0, test.columns.get_loc("train_type")] = "a train type never seen"
    test.iloc[1, test.columns.get_loc("current_delay")] = np.nan
    fitted = SMALL[model](numeric, categorical).fit(train[numeric + categorical],
                                                    train["arrival_delay_min"])
    assert np.isfinite(fitted.predict(test[numeric + categorical])).all()


def test_category_codes_use_training_levels():
    codes = CategoryCodes(["c"]).fit(pd.DataFrame({"c": [True, False, True]}))
    out = codes.transform(pd.DataFrame({"c": [False, True, None]}))
    assert list(out["c"].cat.categories) == ["False", "True"]
    assert out["c"].isna().tolist() == [False, False, True]


@pytest.mark.parametrize("model", ["random_forest", "lightgbm", "xgboost"])
def test_validation_predictions_line_up_with_their_rows(model):
    """With the answer given as a feature, the trees must be close to perfect."""
    rows = delay_rows().assign(answer=lambda r: r["arrival_delay_min"])
    metrics, predictions = cross_validate(rows, "1h", "delay", SMALL[model], ["answer"], [])
    assert (metrics["rmse"] < 3).all()  # a delay std of about 25 min
    # Trees cannot predict beyond the training delays, so only most rows are exact.
    error = (predictions["predicted_delay"] - predictions["arrival_delay_min"]).abs()
    assert error.quantile(0.99) < 2


@pytest.mark.parametrize("model", ["lightgbm", "random_forest"])
def test_shuffled_delays_do_not_beat_the_training_mean(model):
    """With targets unrelated to the features, nothing can be learned (no hidden leak)."""
    rows = delay_rows().copy()
    rows["arrival_delay_min"] = np.random.default_rng(0).permutation(
        rows["arrival_delay_min"].to_numpy())
    numeric, categorical = feature_columns("1h", weather=True)
    model_scores, _ = cross_validate(rows, "1h", "delay", SMALL[model], numeric, categorical)
    mean_scores, _ = cross_validate(rows, "1h", "delay", make_training_mean, numeric, categorical)
    assert model_scores["rmse"].mean() > mean_scores["rmse"].mean() - 0.05


def test_search_chooses_the_lowest_mean_rmse():
    """Constant predictions: the constant nearest the mean delay must win."""
    def constant(numeric, categorical, *, value):
        return DummyRegressor(strategy="constant", constant=value)

    rows = delay_rows()
    configs = [{"value": 0.0}, {"value": float(rows["arrival_delay_min"].mean())}, {"value": 100.0}]
    table, metrics, predictions = search(rows, "1h", "delay", constant, configs, [], ["train_type"])
    assert table.loc[table["chosen"], "config_id"].tolist() == [1]
    assert table["rmse_mean"].idxmin() == 1
    assert metrics["rmse"].tolist() == table.loc[1, [f"rmse_fold{f}" for f in range(1, 6)]].tolist()
    assert (predictions["config_id"] == 1).all()
    assert np.allclose(predictions["predicted_delay"], configs[1]["value"])


@pytest.mark.parametrize("model", list(SMALL_CLASSIFIERS))
def test_classifiers_give_probabilities_for_every_class(model):
    """Also for an unseen train type and a missing `current_delay`."""
    rows = severity_rows()
    numeric, categorical = feature_columns("1h", weather=True)
    train, test = rows[rows["fold"] != 1], rows[rows["fold"] == 1].head(50).copy()
    test.iloc[0, test.columns.get_loc("train_type")] = "a train type never seen"
    test.iloc[1, test.columns.get_loc("current_delay")] = np.nan
    fitted = SMALL_CLASSIFIERS[model](numeric, categorical).fit(train[numeric + categorical],
                                                                train["severity_class"])
    assert sorted(fitted.classes_) == sorted(CLASSES)
    proba = class_probabilities(fitted, test[numeric + categorical])
    assert np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1)


def test_integer_labels_keep_the_original_labels():
    X = pd.DataFrame({"x": [0, 1, 2, 3] * 5})
    y = np.array(["d", "b", "a", "c"] * 5)
    fitted = IntegerLabels(DummyClassifier(strategy="prior")).fit(X, y)
    assert list(fitted.classes_) == ["a", "b", "c", "d"]
    model = make_xgboost_classifier(
        [], ["x"], n_estimators=20, learning_rate=0.3, max_depth=2, min_child_weight=1,
        subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0)
    # each x has one label, so the most probable class must be that label
    assert (model.fit(X.astype(str), y).predict(X.astype(str)) == y).all()


@pytest.mark.parametrize("model", list(SMALL_CLASSIFIERS))
def test_class_predictions_line_up_with_their_rows(model):
    """With the answer given as a feature, the classifiers must be close to perfect."""
    rows = severity_rows().assign(answer=lambda r: r["severity_class"].astype(str))
    metrics, _ = cross_validate(rows, "1h", "severity", SMALL_CLASSIFIERS[model], [], ["answer"])
    assert (metrics["log_loss"] < 0.05).all()
    assert (metrics["macro_f1"] == 1).all()


@pytest.mark.parametrize("model", ["lightgbm_classifier", "random_forest_classifier"])
def test_shuffled_classes_do_not_beat_the_class_shares(model):
    """With labels unrelated to the features, nothing can be learned (no hidden leak)."""
    rows = severity_rows().copy()
    rows["severity_class"] = np.random.default_rng(0).permutation(
        rows["severity_class"].to_numpy())
    numeric, categorical = feature_columns("1h", weather=True)
    model_scores, _ = cross_validate(rows, "1h", "severity", SMALL_CLASSIFIERS[model],
                                     numeric, categorical)
    share_scores, _ = cross_validate(rows, "1h", "severity", make_class_shares,
                                     numeric, categorical)
    assert model_scores["log_loss"].mean() > share_scores["log_loss"].mean() - 0.005


def test_severity_search_chooses_the_lowest_mean_log_loss():
    """Uniform probabilities against the training class shares: the shares must win."""
    def dummy(numeric, categorical, *, strategy):
        return DummyClassifier(strategy=strategy)

    configs = [{"strategy": "uniform"}, {"strategy": "prior"}]
    table, metrics, predictions = search(severity_rows(), "1h", "severity", dummy, configs,
                                         [], ["train_type"])
    assert table.loc[table["chosen"], "config_id"].tolist() == [1]
    assert table["log_loss_mean"].idxmin() == 1
    assert metrics["log_loss"].tolist() == table.loc[1, [f"log_loss_fold{f}"
                                                         for f in range(1, 6)]].tolist()
    assert (predictions["config_id"] == 1).all()


def test_run_search_writes_the_chosen_configurations(tmp_path):
    """The whole run, at two prediction times, with constant predictions."""
    def constant(numeric, categorical, *, value):
        return DummyRegressor(strategy="constant", constant=value)

    space = {"value": [0.0, 20.0, 100.0]}
    summary = run_search("constant", "delay", constant, space, prediction_times=["1h", "0min"],
                         n_configs=3, results_dir=tmp_path)
    (out,) = (tmp_path / "constant").iterdir()
    table = pd.read_csv(out / "search_results.csv")
    assert len(table) == 6 and table["chosen"].sum() == 2
    assert summary["prediction_time"].tolist() == ["1h", "0min"]
    assert set(pd.read_csv(out / "fold_metrics.csv").columns) == {
        "prediction_time", "fold", "train_rows", "validation_rows", "rmse"}
    assert (out / "config.json").exists() and (out / "oof_predictions.parquet").exists()
