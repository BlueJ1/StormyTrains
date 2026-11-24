"""The forecast arms of step 10.9 (decision D17): registration, features and runner wiring."""
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import pytest

from data import ROW_KEY, load_rows
from experiment_models import CATBOOST, LIGHTGBM, LINEAR, XGBOOST, flat_choices
from experiment_specs import ARMS, Run, observed_runs, registered_runs, tuning_reference
from features import feature_columns, feature_groups
from forecast_columns import FORECAST, FORECAST_CHANGE, FORECAST_LAG
from representations import (EXTREMES, REPRESENTATIONS, WEATHER, WEATHER_LAG, weather_representation)
from audit_cv_results import STEPS, active_arms

FORECAST_GROUPS = ("forecast_temperature", "forecast_precipitation", "forecast_wind", "forecast_snow")


def forecast_arms():
    return [run for run in active_arms(STEPS) if run.forecast]


def test_forecast_arms_count_364_runs():
    counts = Counter(run.arm for run in forecast_arms())
    assert counts == {"forecast": 166, "weather_forecast": 198}
    assert sum(counts.values()) == 364


def test_every_kept_weather_arm_gets_both_forecast_variants():
    observed = [run for run in active_arms(STEPS) if run.arm == "weather" and run.step != "3.1"]
    forecasts = {run.name for run in forecast_arms()}
    kept = 0
    for run in observed:
        variants = {run.name.replace("_weather", "_forecast", 1), run.name.replace("_weather", "_weather_forecast", 1)}
        assert variants <= forecasts or not variants & forecasts, run.name
        kept += variants <= forecasts
    # 166 forecast-only arms, 76 of them in 3.1
    assert kept == 166 - 76


def test_registered_names_of_existing_arms_are_unchanged():
    for step in STEPS:
        observed = list(observed_runs(step))
        assert [run for run in registered_runs(step) if not run.forecast] == observed
        assert all(not run.forecast and run.arm in ("railway", "weather") for run in observed)


@pytest.mark.parametrize("run", [
    Run("0.8", XGBOOST[0], True, nested=True),
    Run("0.8", XGBOOST[1], True, variant="auto_groups", nested=True),
    Run("6.1", CATBOOST[1], True, nested=True),
    Run("5.1", LIGHTGBM[0], True, "plus_catboost"),
    Run("5.1", LIGHTGBM[1], True, "plus_neural"),
    Run("5.2", XGBOOST[1], True, "top5_nested"),
    Run("6.2", XGBOOST[1], True, "isotonic"),
    Run("6.2", CATBOOST[1], True, "temperature"),
])
def test_nested_xgboost_catboost_and_their_dependants_get_no_forecast_arms(run):
    names = {arm.name for arm in forecast_arms()}
    from dataclasses import replace
    assert replace(run, forecast=True).name not in names
    assert replace(run, weather=False, forecast=True).name not in names


@pytest.mark.parametrize("run", [
    Run("0.8", LIGHTGBM[0], True, nested=True),
    Run("0.8", LIGHTGBM[1], True, variant="auto_groups", nested=True),
    Run("5.2", XGBOOST[1], True, "top5_flat"),
    Run("5.2", LIGHTGBM[1], True, "top5_nested"),
    Run("6.2", LIGHTGBM[1], True, "isotonic"),
    Run("0.7", XGBOOST[0], True),
])
def test_other_weather_arms_get_both_forecast_arms(run):
    from dataclasses import replace
    names = {arm.name for arm in forecast_arms()}
    assert replace(run, forecast=True).name in names
    assert replace(run, weather=False, forecast=True).name in names


def test_arm_names_and_flat_configuration_source():
    railway, weather = Run("0.7", "lightgbm"), Run("0.7", "lightgbm", True)
    forecast, both = Run("0.7", "lightgbm", forecast=True), Run("0.7", "lightgbm", True, forecast=True)
    assert [run.arm for run in (railway, weather, forecast, both)] == list(ARMS)
    assert forecast.name == "advanced_0_7_lightgbm_forecast"
    assert Run("1.5", "lightgbm", True, "weather_change", forecast=True).name == (
        "advanced_1_5_lightgbm_weather_forecast_weather_change")
    # Both forecast arms reuse the observed-weather flat configuration (D17).
    assert not railway.flat_weather and weather.flat_weather and forecast.flat_weather and both.flat_weather
    assert tuning_reference(Run("6.2", LIGHTGBM[1], variant="isotonic", forecast=True)) == Run(
        "0.8", LIGHTGBM[1], nested=True, forecast=True)


def test_forecast_groups_are_separate_from_observed_groups():
    railway = feature_groups("1h")
    observed = feature_groups("1h", weather=True)
    forecast = feature_groups("1h", forecast=True)
    both = feature_groups("1h", weather=True, forecast=True)
    assert forecast.keys() - railway.keys() == set(FORECAST_GROUPS)
    assert both.keys() == observed.keys() | forecast.keys()
    columns = [c for group in FORECAST_GROUPS for c in forecast[group][0]]
    assert sorted(columns) == sorted(FORECAST)
    assert forecast["forecast_snow"][0] == ("forecast_route_snow_depth_cm",)
    numeric, categorical = feature_columns("1h", forecast=True)
    base_numeric, base_categorical = feature_columns("1h")
    assert numeric == base_numeric + list(FORECAST) and categorical == base_categorical
    numeric, _ = feature_columns("1h", True, forecast=True)
    assert numeric == base_numeric + list(WEATHER) + list(FORECAST)


def test_ablations_follow_each_arms_groups():
    variants = Counter((run.arm, run.model) for run in registered_runs("3.1"))
    for model in (*LINEAR, *LIGHTGBM):
        assert variants["weather", model] == 19
        assert variants["forecast", model] == 19
        assert variants["weather_forecast", model] == 27
    names = {run.variant for run in registered_runs("3.1") if run.arm == "forecast"}
    assert {"add_forecast_snow", "drop_forecast_wind"} <= names and "add_snow" not in names


@pytest.mark.parametrize("weather,forecast,expected", [
    (True, False, {"temperature"}),
    (False, True, {"forecast_temperature"}),
    (True, True, {"temperature", "forecast_temperature"}),
])
def test_weather_subsets_keep_the_arms_groups_of_that_kind(weather, forecast, expected):
    from run_advanced import select_groups
    groups = feature_groups("1h", weather, forecast)
    run = Run("3.4", "lightgbm", weather, "temperature", forecast=forecast)
    assert select_groups(run, groups) == set(feature_groups("1h")) | expected


def toy_weather(n=4):
    values = {column: np.zeros(n) for column in (*WEATHER, *WEATHER_LAG, *FORECAST, *FORECAST_LAG)}
    frame = pd.DataFrame(values)
    frame["forecast_route_temp_max_c"] = [31.0, 30.0, np.nan, 5.0]
    frame["forecast_route_gust_max_ms"] = [21.0, 0, 0, 0]
    frame["forecast_route_precip_max_mm"] = [0, 16.0, 0, 0]
    frame["forecast_route_snow_depth_cm"] = [0, 0, 0, 0.5]
    frame["forecast_origin_temperature_c_lag24h"] = [1.0, 2.0, 3.0, 4.0]
    frame["route_temp_max_24h_c"] = [35.0, 0, 0, 0]
    frame["other"] = 1.0
    return frame


def transform(representation, frame):
    return representation.transformer().fit_transform(frame)


def test_forecast_extremes_use_the_thresholds_of_1_5():
    frame = toy_weather()
    out = transform(weather_representation("weather_extremes", weather=False, forecast=True), frame)
    assert out["forecast_heat"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["forecast_heat"].iloc[2])
    assert out["forecast_gust"].tolist() == [1.0, 0, 0, 0]
    assert out["forecast_heavy_rain"].tolist() == [0, 1.0, 0, 0]
    assert out["forecast_snow"].tolist() == [0, 0, 0, 1.0]
    assert not set(FORECAST) & set(out.columns)
    assert set(WEATHER) <= set(out.columns)  # observed columns untouched (not in this arm's inputs)


def test_combined_representations_replace_both_sources():
    frame = toy_weather()
    extremes = transform(weather_representation("weather_extremes", weather=True, forecast=True), frame)
    assert set(EXTREMES) | {"forecast_heat", "forecast_gust", "forecast_snow", "forecast_heavy_rain"} <= set(extremes)
    assert not (set(WEATHER) | set(FORECAST)) & set(extremes.columns)
    assert extremes["weather_heat"].tolist() == [1.0, 0, 0, 0]
    change = transform(weather_representation("weather_change", weather=True, forecast=True), frame)
    assert set(FORECAST_CHANGE) <= set(change) and not set(FORECAST_LAG) & set(change)
    assert change["forecast_origin_temperature_c_change24h"].tolist() == [-1.0, -2.0, -3.0, -4.0]
    assert change["other"].eq(1).all()


def test_observed_representations_are_unchanged():
    for variant in ("weather_extremes", "weather_change"):
        assert weather_representation(variant) is REPRESENTATIONS[variant]
    representation = weather_representation("weather_change", weather=False, forecast=True)
    assert representation.extra_inputs == FORECAST_LAG and representation.inputs == FORECAST


def test_loader_merges_forecasts_only_for_forecast_arms(tmp_path):
    from advanced_data import load_advanced_rows
    frame = load_rows("24h")
    frame[ROW_KEY].assign(scheduled_train_type="IC", scheduled_starts_at_origin=False).to_parquet(
        tmp_path / "timetable_cv.parquet")
    forecasts = frame[ROW_KEY].assign(**{c: 1.5 for c in (*FORECAST, *FORECAST_LAG)})
    forecasts.to_parquet(tmp_path / "forecast_weather_cv.parquet")
    railway = load_advanced_rows("24h", inputs_dir=tmp_path)
    assert not set(FORECAST) & set(railway.columns)
    result = load_advanced_rows("24h", forecast=True, inputs_dir=tmp_path)
    assert result[list(FORECAST)].eq(1.5).all(axis=None)
    assert not set(FORECAST_LAG) & set(result.columns)
    lagged = load_advanced_rows("24h", forecast=True, inputs_dir=tmp_path, extra_inputs=FORECAST_LAG)
    assert set(FORECAST_LAG) <= set(lagged.columns)


def test_runner_feeds_forecasts_with_the_observed_flat_configuration(monkeypatch, tmp_path):
    import run_advanced
    calls = {}

    def load(prediction_time, **kwargs):
        calls["load"] = kwargs
        return "rows"

    def cross_validate(rows, prediction_time, task, make_model, numeric, categorical):
        calls["model"], calls["numeric"] = make_model, numeric
        raise StopIteration

    monkeypatch.setattr(run_advanced, "git_commit", lambda: "test-revision")
    monkeypatch.setattr(run_advanced, "load_advanced_rows", load)
    monkeypatch.setattr(run_advanced, "cross_validate", cross_validate)
    with pytest.raises(StopIteration):
        run_advanced.evaluate_run(Run("0.7", "lightgbm", forecast=True), inputs_dir=tmp_path,
                                  results_dir=tmp_path, prediction_times=("1h",))
    assert calls["load"]["forecast"] is True and calls["load"]["weather"] is False
    assert set(FORECAST) <= set(calls["numeric"]) and not set(WEATHER) & set(calls["numeric"])
    assert isinstance(calls["model"], partial)
    assert calls["model"].keywords == flat_choices("lightgbm", True)["1h"]["params"]


def test_cli_and_audit_select_arms(monkeypatch):
    import sys
    import run_advanced
    seen = []
    monkeypatch.setattr(sys, "argv", ["run_advanced", "0.7", "--model", "lightgbm",
                                      "--weather-arm", "forecast", "weather_forecast"])
    monkeypatch.setattr(run_advanced, "evaluate_run", lambda run, **kwargs: seen.append(run.name))
    monkeypatch.setattr(run_advanced, "compare_to_base", lambda *args: None)
    run_advanced.main()
    assert sorted(seen) == ["advanced_0_7_lightgbm_forecast", "advanced_0_7_lightgbm_forecast_recorded_route",
                            "advanced_0_7_lightgbm_weather_forecast",
                            "advanced_0_7_lightgbm_weather_forecast_recorded_route"]

    import audit_cv_results
    audited = []
    monkeypatch.setattr(sys, "argv", ["audit", "--commit", "x", "--arm", "forecast", "weather_forecast"])
    monkeypatch.setattr(audit_cv_results, "audit", lambda run, *args: audited.append(run) or (None, []))
    audit_cv_results.main()
    assert len(audited) == 364 and all(run.forecast for run in audited)


def test_audit_accepts_dirty_flag_only_for_the_b17_commit_and_forecast_arms():
    from audit_cv_results import B17_COMMIT, commit_ok
    forecast, observed = Run("0.7", "lightgbm", forecast=True), Run("0.7", "lightgbm", weather=True)
    flagged, other = B17_COMMIT + " (uncommitted changes)", "a" * 40
    assert commit_ok(other, [other], observed)
    assert commit_ok(other, [B17_COMMIT, other], forecast)
    assert not commit_ok(flagged, [B17_COMMIT], forecast)
    assert commit_ok(flagged, [B17_COMMIT], forecast, accept_flagged=True)
    assert not commit_ok(flagged, [B17_COMMIT], observed, accept_flagged=True)
    assert not commit_ok(other + " (uncommitted changes)", [other], forecast, accept_flagged=True)
    assert not commit_ok(flagged, [other], forecast, accept_flagged=True)


def test_audit_defaults_to_the_railway_and_observed_arms(monkeypatch):
    import sys
    import audit_cv_results
    audited = []
    monkeypatch.setattr(sys, "argv", ["audit", "--commit", "x"])
    monkeypatch.setattr(audit_cv_results, "audit", lambda run, *args: audited.append(run) or (None, []))
    audit_cv_results.main()
    assert len(audited) == 354 and not any(run.forecast for run in audited)


def test_git_commit_ignores_results_outside_its_own_folder_but_not_other_changes(tmp_path, monkeypatch):
    import subprocess
    import cv

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "modeling").mkdir()
    (tmp_path / "modeling" / "cv.py").write_text("")
    old = tmp_path / "advanced" / "results" / "old_run"
    old.mkdir(parents=True)
    (old / "config.json").write_text("{}")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    monkeypatch.setattr(cv, "HERE", tmp_path / "modeling")  # git_commit runs from modeling/
    new = tmp_path / "advanced" / "results" / "new_run" / "20261119-000000"
    new.mkdir(parents=True)
    (new / "config.json").write_text("{}")
    (old / "config.json").write_text('{"rerun": true}')
    assert "uncommitted" not in cv.git_commit()
    (tmp_path / "advanced" / "notes.txt").write_text("x")
    assert cv.git_commit().endswith("(uncommitted changes)")


def test_forecast_comparisons_pair_each_forecast_arm_with_its_registered_counterparts():
    from forecast_comparisons import pairs
    found = Counter(contrast for contrast, _, _ in pairs())
    # railway only has no 1.5, 3.2 or 3.4 arm; the 3.1 forecast ablations exist only with forecasts
    assert found == {"forecast_vs_railway": 74, "forecast_vs_weather": 134,
                     "weather_forecast_vs_weather": 166, "weather_forecast_vs_forecast": 166}
    for contrast, run, reference in pairs():
        assert (run.step, run.model, run.variant, run.nested) == (
            reference.step, reference.model, reference.variant, reference.nested)


@pytest.mark.parametrize("weather, forecast, groups", [
    (True, False, ("temperature", "precipitation", "wind", "snow")),
    (False, True, FORECAST_GROUPS),
    (True, True, ("temperature", "precipitation", "wind", "snow", *FORECAST_GROUPS))])
def test_auto_groups_weather_stage_keeps_all_none_or_all_but_one_group(weather, forecast, groups):
    from experiment_specs import weather_group_variants
    run = Run("0.8", "lightgbm", weather, variant="auto_groups", nested=True, forecast=forecast)
    numeric, categorical = feature_columns("1h", weather, feature_set="holidays", forecast=forecast)
    variant = ("factory", numeric, categorical)
    options = weather_group_variants(run, "1h", variant)
    assert list(options)[:2] == ["all_weather", "no_weather"]
    assert options["all_weather"] is variant
    assert set(options) == {"all_weather", "no_weather", *(f"drop_{group}" for group in groups)}
    all_groups = feature_groups("1h", weather, forecast)
    weather_columns = {c for group in groups for c in all_groups[group][0]}
    assert set(options["no_weather"][1]) == set(numeric) - weather_columns
    for group in groups:
        assert set(options[f"drop_{group}"][1]) == set(numeric) - set(all_groups[group][0])
        assert options[f"drop_{group}"][2] == categorical
    assert weather_group_variants(replace_run(run, weather=False, forecast=False), "1h", variant) is None
    assert weather_group_variants(replace_run(run, variant="auto_representations"), "1h", variant) is None


def replace_run(run, **changes):
    from dataclasses import replace
    return replace(run, **changes)


def test_nested_search_weather_stage_reuses_the_kept_screen_and_records_the_choice(monkeypatch):
    import tuning
    from sklearn.dummy import DummyRegressor
    frame = load_rows("1h").groupby("fold", group_keys=False).head(300).assign(arrival_delay_min=3.0)

    def constant(numeric, categorical, *, value=0.0, offset=0.0):
        return DummyRegressor(strategy="constant", constant=value + offset)
    variants = {"base": (partial(constant, value=10.0), [], ["hour"]),
                "holidays": (partial(constant, value=6.0), [], ["hour"])}
    screened = []
    real = tuning.screen_feature_variants
    monkeypatch.setattr(tuning, "screen_feature_variants",
                        lambda rows, pt, task, options, *args: screened.append(list(options)) or
                        real(rows, pt, task, options, *args))

    def weather(variant):
        return {"all_weather": variant, "no_weather": (partial(constant, value=3.0), [], ["hour"]),
                "drop_wind": (partial(constant, value=4.0), [], ["hour"])}
    table, metrics, _ = tuning.nested_search(frame, "1h", "delay", constant, [{"offset": 0.0}], [], ["hour"],
                                             variants=variants, weather_variants=weather)
    chosen = table.loc[table["chosen"]]
    assert set(chosen["feature_variant"]) == {"holidays"} and set(chosen["weather_variant"]) == {"no_weather"}
    assert np.allclose(metrics["rmse"], 0)
    assert screened[1] == ["no_weather", "drop_wind"]  # the kept variant is not screened again

    tie = lambda variant: {"all_weather": variant, "no_weather": variant}
    table, _, _ = tuning.nested_search(frame, "1h", "delay", constant, [{"offset": 0.0}], [], ["hour"],
                                       variants=variants, weather_variants=tie)
    assert set(table.loc[table["chosen"], "weather_variant"]) == {"all_weather"}


def test_forecast_comparisons_mark_diagnostics_same_inputs_and_configurations(tmp_path):
    import json
    from forecast_comparisons import configuration, excluded

    def save(name, **config):
        out = tmp_path / name / "20261120-000000"
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps(config))
        return out
    railway = save("r", features={"1h": {"numeric": ["a"], "categorical": ["c"]}},
                   configuration_source="lightgbm/x/config.json")
    forecast = save("f", features={"1h": {"numeric": ["a", "f"], "categorical": ["c"]}},
                    configuration_source="lightgbm_weather/y/config.json")
    same = save("s", features={"1h": {"numeric": ["a"], "categorical": ["c"]}},
                configuration_source="lightgbm_weather/y/config.json")
    run = Run("3.1", "lightgbm", forecast=True, variant="timetable")
    assert excluded(run, same, railway) == "same inputs"
    assert excluded(run, forecast, railway) == ""
    assert excluded(Run("0.7", "lightgbm", forecast=True, variant="recorded_route", diagnostic=True),
                    forecast, railway) == "recorded-route diagnostic"
    assert configuration(forecast, railway, tmp_path) == "different flat configuration"
    assert configuration(forecast, same, tmp_path) == "same configuration"
    average = save("m", members=["f/20261120-000000"])
    base_average = save("n", members=["r/20261120-000000"])
    assert configuration(average, base_average, tmp_path) == "different flat configuration"
    nested = [save(n, search={"n_configs": 20}, features={}) for n in ("x", "y")]
    assert configuration(*nested, tmp_path) == "tuned per arm"
