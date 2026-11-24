"""Four-arm finalists for the second hold-out evaluation (finishing plan B6, decision D17).

Group decisions of 24 November 2026: only pipelines with all four CV arms are candidates; the
registered 3.3 weather arms keep all their weather groups; the hold-out set compares five pairs.
"""
import json

import pandas as pd
import pytest

import run_final
from data import PREDICTION_TIMES
from features import WEATHER
from forecast_columns import FORECAST
from tuning import SCORE

RAILWAY = {"numeric": ["planned_minutes"], "categorical": ["hour"]}


def arm_features(arm, weather_columns=None):
    columns = sorted(run_final.arm_weather_columns(arm)) if weather_columns is None else weather_columns
    return {"numeric": [*RAILWAY["numeric"], *columns], "categorical": list(RAILWAY["categorical"])}


def save_run(results, name, **config):
    out = results / f"advanced_0_8_{name}" / "20261124-000000"
    out.mkdir(parents=True)
    defaults = {"step": "0.8", "model": "lightgbm", "task": "delay", "evaluation": "nested", "variant": ""}
    (out / "config.json").write_text(json.dumps({**defaults, **config}))
    pd.DataFrame({"prediction_time": ["1h"], "rmse_mean": [1.0]}).to_csv(out / "summary.csv", index=False)


def test_candidates_need_all_four_arms(tmp_path):
    for arm, (weather, forecast) in run_final.ARMS.items():
        save_run(tmp_path, f"lightgbm_{arm}", weather=weather, forecast=forecast,
                 features={"1h": arm_features(arm)})
    for arm in ("railway", "weather"):  # nested XGBoost has no forecast arms (10.9)
        weather, forecast = run_final.ARMS[arm]
        save_run(tmp_path, f"xgboost_{arm}", model="xgboost", weather=weather, forecast=forecast,
                 features={"1h": arm_features(arm)})
    pairs = run_final.matched_candidates(tmp_path)
    assert len(pairs) == 1
    assert {arm: config["model"] for arm, (_, config, _) in pairs[0].items()} == dict.fromkeys(
        run_final.ARMS, "lightgbm")
    assert {arm: run_final.arm_name(config) for arm, (_, config, _) in pairs[0].items()} == {
        arm: arm for arm in run_final.ARMS}


def test_old_runs_without_forecast_key_are_railway_or_weather():
    assert run_final.arm_name({"weather": False}) == "railway"
    assert run_final.arm_name({"weather": True}) == "weather"
    assert run_final.arm_name({"weather": True, "forecast": True}) == "weather_forecast"


def test_weather_stage_search_key_does_not_unmatch_the_pipeline():
    railway = {"search": {"n_configs": 20, "selection": "inner mean rmse", "feature_selection": "x"}}
    weather = {"search": {**railway["search"], "weather_selection": "all, none or one dropped"}}
    assert run_final.matched_search(railway, weather)
    assert not run_final.matched_search(railway, {"search": {**railway["search"], "n_configs": 8}})


def test_matched_features_by_arm():
    assert run_final.matched_features(RAILWAY, arm_features("forecast"), "forecast")
    assert not run_final.matched_features(RAILWAY, arm_features("weather"), "forecast")
    assert run_final.matched_features(RAILWAY, arm_features("weather_forecast"), "weather_forecast")
    one_group = arm_features("weather", [c for c in WEATHER if "temp" in c])
    assert not run_final.matched_features(RAILWAY, one_group, "weather")
    assert run_final.matched_features(RAILWAY, one_group, "weather", subset=True)
    changed_railway = {**arm_features("weather"), "categorical": []}
    assert not run_final.matched_features(RAILWAY, changed_railway, "weather", subset=True)


def registration(edit=None):
    finalists = []
    for task in SCORE:
        for pt in PREDICTION_TIMES:
            for arm, (weather, forecast) in run_final.ARMS.items():
                spec = {"task": task, "prediction_time": pt, "weather": weather, "forecast": forecast,
                        "model": "lightgbm", "step": "0.8", "variant": "all_groups", "calendar": None,
                        "features": arm_features(arm), "tuning_policy": {"method": "five CV week folds",
                                                                          "config_id": len(finalists)}}
                if edit:
                    edit(spec, arm)
                finalists.append(spec)
    return {"finalists": finalists}


def test_registry_has_four_matched_arms_per_slot():
    run_final.validate_registry(registration())
    missing = registration()
    missing["finalists"] = [spec for spec in missing["finalists"] if not spec["forecast"]]
    with pytest.raises(ValueError, match="all four arms"):
        run_final.validate_registry(missing)

    def other_model(spec, arm):
        if arm == "forecast" and spec["prediction_time"] == "1h":
            spec["model"] = "xgboost"
    with pytest.raises(ValueError, match="unmatched forecast comparison"):
        run_final.validate_registry(registration(other_model))


def test_registry_allows_fewer_weather_columns_only_after_the_weather_stage():
    def dropped(spec, arm, record=True):
        if arm == "weather_forecast":
            spec["features"] = arm_features(arm, sorted(FORECAST))
            if record:
                spec["weather_variant"] = "no_weather"  # hypothetical: only forecasts left
    run_final.validate_registry(registration(dropped))
    with pytest.raises(ValueError, match="unmatched weather_forecast comparison"):
        run_final.validate_registry(registration(lambda spec, arm: dropped(spec, arm, record=False)))


def test_registration_runs_the_weather_stage_in_weather_arms_only(tmp_path, monkeypatch):
    (tmp_path / "PLAN.md").write_text("All development steps are complete.")
    monkeypatch.setattr(run_final, "HERE", tmp_path)
    monkeypatch.setattr(run_final, "git_commit", lambda: "test-revision")
    monkeypatch.setattr(run_final, "runtime_versions", lambda: {"python": "test"})
    monkeypatch.setattr(run_final, "evaluation_fingerprints", lambda *args, **kwargs: {"cv": "frozen"})
    monkeypatch.setattr(run_final, "tune_finalist", lambda spec, config, inputs: {
        "params": {}, "tuning_policy": {"method": "test CV"}})
    loaded = []
    monkeypatch.setattr(run_final, "load_advanced_rows",
                        lambda pt, **options: loaded.append((options.get("weather", False),
                                                             options.get("forecast", False))))
    screened = []

    def select(rows, pt, task, variants, configs, splits):
        screened.append(list(variants))
        return "all_groups" if "all_groups" in variants else "drop_temperature" if "drop_temperature" in variants \
            else "drop_forecast_wind"
    monkeypatch.setattr(run_final, "select_feature_variant", select)
    pair = {}
    for arm, (weather, forecast) in run_final.ARMS.items():
        config = {"model": "lightgbm", "task": "delay", "weather": weather, "forecast": forecast,
                  "step": "0.8", "variant": "auto_groups", "features": {"1h": arm_features(arm)}}
        scores = pd.DataFrame({"rmse_mean": [1.0]}, index=pd.Index(["1h"], name="prediction_time"))
        pair[arm] = (tmp_path / "results" / arm, config, scores)
    monkeypatch.setattr(run_final, "choose_finalists", lambda path: iter([("delay", "1h", pair)]))
    registry = tmp_path / "finalists.json"
    run_final.register(tmp_path / "results", tmp_path, registry)

    assert loaded == [(False, False), (True, False), (False, True), (True, True)]
    assert screened[0][0] == "base" and "all_groups" in screened[0]  # railway candidates, railway arm
    assert all(variants[:2] == ["all_weather", "no_weather"] for variants in screened[1:])
    finalists = json.loads(registry.read_text())["finalists"]
    assert [spec.get("weather_variant") for spec in finalists] == [
        None, "drop_temperature", "drop_forecast_wind", "drop_temperature"]
    assert {spec["variant"] for spec in finalists} == {"all_groups"}
    weather = finalists[1]["features"]["numeric"]
    assert not any("temp" in column for column in weather if column in WEATHER)


def test_fingerprints_include_forecast_inputs_when_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(run_final, "source_fingerprints", lambda path: {})
    monkeypatch.setattr(run_final, "fingerprint_files", lambda paths: dict.fromkeys(paths, "x"))
    spec = {"features": RAILWAY, "forecast": True}
    names = run_final.evaluation_fingerprints({"finalists": [spec]}, tmp_path)
    assert "advanced/forecast_weather_holdout.parquet" in names
    names = run_final.evaluation_fingerprints({"finalists": [{**spec, "forecast": False}]}, tmp_path)
    assert not any("forecast" in name for name in names)


def test_holdout_scoring_writes_five_contrasts(tmp_path, monkeypatch):
    finalists = registration()["finalists"]
    for index, spec in enumerate(finalists):
        arm = tmp_path / "_arms" / str(index)
        arm.mkdir(parents=True)
        (arm / "score.json").write_text(json.dumps({"fold": 0, "rmse": 1.0}))
        (arm / "environment.json").write_text(json.dumps({"interpreter": "python"}))
        prediction = pd.DataFrame({"prediction_time": [spec["prediction_time"]],
                                   "arm": [run_final.arm_name(spec)]})
        prediction.to_parquet(arm / "predictions.parquet", index=False)
        pd.DataFrame({"feature": ["a"], "mean_abs_shap": [1.0]}).to_parquet(arm / "shap.parquet", index=False)
        if run_final.arm_name(spec) == "weather":
            pd.DataFrame({"journey": [index]}).to_parquet(arm / "metadata.parquet", index=False)
    pairs = []
    monkeypatch.setattr(run_final, "paired_bootstrap_ci", lambda left, right, task: pairs.append(
        (task, right["arm"].iloc[0], left["arm"].iloc[0])) or pd.DataFrame({"delta": [0.0]}))
    monkeypatch.setattr(run_final, "paired_error_table", lambda *args, **kwargs: pd.DataFrame({"n": [1]}))
    monkeypatch.setattr(run_final, "calibration_reliability", lambda frame: pd.DataFrame({"bin": [0]}))
    monkeypatch.setattr(run_final, "example_risk_trajectories", lambda frame: pd.DataFrame({"journey": [0]}))
    import result_plots
    monkeypatch.setattr(result_plots, "calibration_plot", lambda tables, path: path.write_text(",".join(tables)))
    monkeypatch.setattr(result_plots, "risk_plot", lambda tables, path: path.write_text(",".join(tables)))
    run_final.run_final_arms({"finalists": finalists}, tmp_path / "registry.json", tmp_path, tmp_path, None)

    assert pairs == [(task, *contrast) for task in SCORE for contrast in run_final.CONTRASTS]
    for task in SCORE:
        assert (tmp_path / f"{task}_weather.csv").is_file()  # first-evaluation name kept
        assert (tmp_path / f"{task}_errors.csv").is_file()
        for candidate, reference in run_final.CONTRASTS[1:]:
            assert (tmp_path / f"{task}_{candidate}_vs_{reference}.csv").is_file()
            assert (tmp_path / f"{task}_{candidate}_vs_{reference}_errors.csv").is_file()
        for arm in run_final.ARMS:
            assert (tmp_path / f"{task}_{arm}.parquet").is_file()
    assert (tmp_path / "calibration.svg").read_text() == ",".join(run_final.ARMS)
    scores = pd.read_csv(tmp_path / "scores.csv")
    assert scores.groupby("arm").size().to_dict() == dict.fromkeys(run_final.ARMS, 2 * len(PREDICTION_TIMES))
