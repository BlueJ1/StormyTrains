import json
from functools import cache, partial

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from compare import paired_differences
from advanced_data import load_advanced_rows
from cv import pooled_cv
from data import ROW_KEY, TIME_COLUMN, load_rows, merge_inputs
from features import RECORDED_ROUTE, feature_columns
from experiment_specs import Run
from folds import cv_folds, inner_folds, week_of
from tuning import fixed_configurations, nested_search


@cache
def rows():
    return load_rows("1h")


def test_recorded_route_is_diagnostic_only():
    base = feature_columns("1h")
    diagnostic = feature_columns("1h", feature_set="recorded_route_diagnostic")
    for default, audit, removed in zip(base, diagnostic, RECORDED_ROUTE):
        assert not set(default) & set(removed)
        assert set(audit) == set(default) | set(removed)
    with pytest.raises(ValueError, match="unknown feature set"):
        feature_columns("1h", feature_set="typo")


def test_inner_folds_cover_outer_training_weeks_once():
    frame = rows()
    for _, outer_train, outer_validation in cv_folds(frame):
        train = frame.loc[outer_train]
        seen = set()
        for _, inner_train, inner_validation in inner_folds(train):
            valid_weeks = set(week_of(train.loc[inner_validation, TIME_COLUMN]))
            assert not valid_weeks & set(week_of(train.loc[inner_train, TIME_COLUMN]))
            assert not valid_weeks & set(week_of(frame.loc[outer_validation, TIME_COLUMN]))
            assert not valid_weeks & seen
            seen.update(valid_weeks)
        assert seen == set(week_of(train[TIME_COLUMN]))


def test_nested_search_cannot_choose_using_outer_labels():
    frame = rows().assign(arrival_delay_min=0.0)
    frame.loc[frame["fold"].eq(1), "arrival_delay_min"] = 100

    def constant(numeric, categorical, *, value):
        return DummyRegressor(strategy="constant", constant=value)

    table, metrics, oof = nested_search(frame, "1h", "delay", constant,
                                        [{"value": 0}, {"value": 100}], [], ["hour"])
    assert table.loc[table["outer_fold"].eq(1) & table["chosen"], "config_id"].item() == 0
    assert metrics.loc[metrics["fold"].eq(1), "rmse"].item() == 100
    assert oof["journey_id"].is_unique and len(oof) == len(frame)


def test_nested_variant_selection_is_independent_of_outer_labels():
    frame = rows().assign(arrival_delay_min=0.0)
    frame.loc[frame["fold"].eq(1), "arrival_delay_min"] = 100

    def constant(numeric, categorical, *, value, offset):
        return DummyRegressor(strategy="constant", constant=value + offset)

    variants = {name: (partial(constant, value=value), [], ["hour"])
                for name, value in [("first", 2), ("second", 7)]}
    table, metrics, _ = nested_search(frame, "1h", "delay", constant, [{"offset": 0}, {"offset": -7}], [],
                                     ["hour"], variants=variants)
    chosen = table.loc[table["outer_fold"].eq(1) & table["chosen"]]
    assert chosen["feature_variant"].item() == "second"
    assert chosen["config_id"].item() == 1
    assert metrics.loc[metrics["fold"].eq(1), "rmse"].item() == 100


def test_pooling_keeps_all_horizons_in_the_same_journey_fold():
    early = load_rows("24h").groupby("fold", group_keys=False).head(300)
    late = rows()[rows()["journey_id"].isin(early["journey_id"])]
    frame = pd.concat([early, late], ignore_index=True)
    metrics, oof = pooled_cv(frame, "delay", lambda *_: DummyRegressor(), [], ["hour"])
    assert len(metrics) == 10
    assert len(oof) == frame["arrival_delay_min"].notna().sum()
    assert oof.groupby("journey_id")["fold"].nunique().eq(1).all()


def test_network_transfer_uses_furthest_stop_after_earlier_stop_correction():
    from network_transfer import inference_context

    frame = pd.DataFrame({
        "journey_id": ["one"], "prediction_time": ["1h"],
        "predicted_at": pd.to_datetime(["2025-12-01 10:00"]),
        "destination_arrival_planned_local": pd.to_datetime(["2025-12-01 12:00"]),
        "planned_stops": [4],
    })
    keys = frame[["journey_id", "prediction_time", "predicted_at"]]
    events = pd.concat([keys, keys], ignore_index=True).assign(
        station_sequence=[2, 3], station_name=["Earlier", "Frankfurt (Main) Hbf"],
        reported_time_utc=pd.to_datetime(["2025-12-01 08:55", "2025-12-01 08:50"], utc=True),
        planned_time_utc=pd.to_datetime(["2025-12-01 08:20", "2025-12-01 08:45"], utc=True),
        event_report_visible=True, event_occurred_by_cutoff=True,
        event_kind="arrival", event_cancelled=False, event_delay_min=[35, 5],
    )

    result = inference_context(frame, events).iloc[0]
    assert result.last_stop_delay == 5
    assert result.remaining_planned_stops == 3
    assert result.remaining_planned_min == 135


def test_neural_reference_names_distinguish_calendar_choices():
    assert Run("7.1", "mlp", calendar="calendar_cyclic").name != Run(
        "7.1", "mlp", calendar="calendar_integer").name


@pytest.mark.parametrize("run,expected", [
    (Run("0.8", "lightgbm", nested=True), 20),
    (Run("6.1", "catboost", nested=True, n_configs=7), 7),
    (Run("7.1", "mlp", nested=True), 8),
])
def test_nested_runner_uses_registered_configuration_budget(monkeypatch, tmp_path, run, expected):
    import run_advanced

    counts = []

    def configs(space, n):
        counts.append(n)
        return []

    monkeypatch.setattr(run_advanced, "sample_configs", configs)
    monkeypatch.setattr(run_advanced, "git_commit", lambda: "test-revision")
    run_advanced.evaluate_run(run, inputs_dir=tmp_path, results_dir=tmp_path, prediction_times=())

    assert counts == [expected]


@pytest.mark.parametrize("arguments,expected", [([], 20), (["--n-configs", "7"], 7)])
def test_search_cli_passes_same_budget_to_both_weather_arms(monkeypatch, arguments, expected):
    import sys
    import run_advanced

    budgets = []
    monkeypatch.setattr(sys, "argv", ["run_advanced", "0.8", "--model", "lightgbm", *arguments])
    monkeypatch.setattr(run_advanced, "evaluate_run", lambda run, **kwargs: budgets.append(run.n_configs))
    monkeypatch.setattr(run_advanced, "compare_to_base", lambda *args: None)

    run_advanced.main()

    assert budgets == [expected, expected]


def test_differences_align_by_fold_and_use_the_registered_threshold():
    reference = pd.DataFrame({"prediction_time": "1h", "fold": range(1, 6),
                              "validation_rows": range(30, 35), "log_loss": 1.0})
    delta = np.array([-0.01, -0.01, -0.01, 0.01, 0])
    candidate = reference.assign(log_loss=1.0 + delta).iloc[::-1]
    comparison = paired_differences(reference, candidate, "log_loss").iloc[0]
    assert comparison["mean"] == pytest.approx(delta.mean())
    assert comparison["folds_improved"] == 3
    assert comparison["reaches_minimum_gain"]
    with pytest.raises(ValueError, match="counts differ"):
        paired_differences(reference, candidate.assign(validation_rows=99), "log_loss")


def test_fixed_configs_require_a_matching_flat_search(tmp_path):
    path = tmp_path / "config.json"
    options = {"task": "delay", "model": "make_lightgbm", "weather": False}
    chosen = {"1h": {"config_id": 4, "params": {"n_estimators": 123}}}
    path.write_text(json.dumps({**options, "chosen": chosen}))
    assert fixed_configurations(path, **options) == chosen
    with pytest.raises(ValueError, match="does not match"):
        fixed_configurations(path, **{**options, "weather": True})
    path.write_text(json.dumps({**options, "chosen": {"1h": {"1": chosen["1h"]}}}))
    with pytest.raises(ValueError, match="flat search"):
        fixed_configurations(path, **options)


def test_advanced_inputs_reject_missing_or_duplicate_rows():
    frame = rows().head(3)
    source = frame[ROW_KEY].assign(feature=[1, 2, 3])
    assert merge_inputs(frame, source.iloc[::-1])["feature"].tolist() == [1, 2, 3]
    for bad in [source.head(1), pd.concat([source, source.head(1)])]:
        with pytest.raises(ValueError, match="uniquely cover"):
            merge_inputs(frame, bad)


def test_advanced_loader_requires_reconstructed_live_data(tmp_path):
    frame = rows()
    timetable = frame[ROW_KEY].assign(
        scheduled_train_type="IC", scheduled_starts_at_origin=False)
    timetable.to_parquet(tmp_path / "timetable_cv.parquet")
    with pytest.raises(FileNotFoundError, match="history_cv"):
        load_advanced_rows("1h", inputs_dir=tmp_path)
    frame[ROW_KEY].assign(current_delay=123.0).to_parquet(tmp_path / "history_cv.parquet")
    result = load_advanced_rows("1h", inputs_dir=tmp_path)
    assert result[ROW_KEY].equals(frame[ROW_KEY])
    assert result["current_delay"].eq(123).all()
    assert result["train_type"].eq("IC").all()


def test_advanced_loader_does_not_open_holdout_by_default(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("loader attempted to read a parquet")

    monkeypatch.setattr(pd, "read_parquet", deny)
    with pytest.raises(PermissionError):
        load_advanced_rows("1h", split="holdout")


def test_network_training_excludes_validation_run_and_report_week(tmp_path, monkeypatch):
    import network_transfer as transfer

    train_week, validation_week = pd.to_datetime(["2025-11-03", "2025-11-10"])
    assignment = pd.DataFrame({"week_start": [train_week, validation_week], "split": "cv"})
    monkeypatch.setattr(transfer, "read_assignment", lambda: assignment)
    pairs = pd.DataFrame({"id": ["allowed-1", "validation-run-1", "late-outcome-1"],
                          "delay_target": [5.0, 100.0, 200.0]})
    pairs[transfer.CONTEXT] = 1.0
    pairs[transfer.WEEKS] = train_week
    pairs.loc[2, "target_report_week"] = validation_week
    path = tmp_path / "pairs.parquet"
    pairs.to_parquet(path)

    model = transfer.fit_network(path, (train_week,), ("validation-run",))

    np.testing.assert_allclose(model.predict(pairs[transfer.CONTEXT]), 5.0)
