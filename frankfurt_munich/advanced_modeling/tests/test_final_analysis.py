from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_analysis  # noqa: E402
from final_analysis import (  # noqa: E402
    HORIZONS,
    PROBA_COLUMNS,
    calibration_reliability,
    example_risk_trajectories,
    paired_bootstrap_ci,
    paired_error_table,
)
from metrics import CLASSES  # noqa: E402


def _severity_predictions(n=8, *, all_horizons=False):
    departures = pd.to_datetime([
        "2025-09-22", "2025-09-23", "2025-09-29", "2025-09-30",
        "2025-10-06", "2025-10-07", "2025-10-13", "2025-10-14",
    ][:n])
    labels = [CLASSES[i % len(CLASSES)] for i in range(n)]
    rows = []
    times = HORIZONS if all_horizons else ("24h",)
    for index in range(n):
        p = np.full(4, 0.1)
        p[CLASSES.index(labels[index])] = 0.7
        for horizon in times:
            rows.append({
                "journey_id": f"journey-{index}",
                "prediction_time": horizon,
                "origin_departure_planned_local": departures[index],
                "severity_class": labels[index],
                **dict(zip(PROBA_COLUMNS, p)),
            })
    return pd.DataFrame(rows)


def _metadata(predictions):
    result = predictions[[
        "journey_id", "prediction_time", "origin_departure_planned_local",
        "severity_class",
    ]].drop_duplicates().copy()
    result["weather_branch"] = ["Nürnberg", "Stuttgart", None, "Nürnberg"] * (len(result) // 4) + ["Nürnberg"] * (len(result) % 4)
    result["weather_heat"] = [1.0, 0.0, np.nan, 1.0] * (len(result) // 4) + [0.0] * (len(result) % 4)
    result["weather_gust"] = 0.0
    result["weather_snow"] = 0.0
    result["weather_heavy_rain"] = 0.0
    return result


def test_paired_week_bootstrap_is_repeatable_and_uses_expected_delta_sign():
    left = _severity_predictions()
    right = left.copy()
    for label, column in zip(CLASSES, PROBA_COLUMNS):
        right[column] = np.where(right.severity_class.eq(label), 0.82, 0.06)
    first = paired_bootstrap_ci(left.sample(frac=1, random_state=9), right,
                                "severity", n_resamples=80, seed=17)
    second = paired_bootstrap_ci(left, right, "severity", n_resamples=80, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "n_weeks"] == 4
    assert first.loc[0, "delta"] < 0
    assert first.loc[0, "ci_upper"] < 0
    assert first.loc[0, "delta_macro_f1"] == pytest.approx(0)


def test_paired_bootstrap_delay_uses_rmse_and_rejects_unpaired_rows():
    left = pd.DataFrame({
        "journey_id": ["a", "b", "c", "d"],
        "prediction_time": ["1h"] * 4,
        "origin_departure_planned_local": pd.to_datetime(
            ["2025-01-06", "2025-01-07", "2025-01-13", "2025-01-14"]),
        "arrival_delay_min": [0.0, 20.0, 60.0, 10.0],
        "predicted_delay": [4.0, 24.0, 64.0, 14.0],
    })
    right = left.copy()
    right["predicted_delay"] = [2.0, 22.0, 62.0, 12.0]
    result = paired_bootstrap_ci(left, right, "delay", n_resamples=50, seed=2)

    assert result.loc[0, "metric"] == "rmse"
    assert result.loc[0, "delta"] < 0
    with pytest.raises(ValueError, match="same keys"):
        paired_bootstrap_ci(left, right.iloc[:-1], "delay", n_resamples=10)


def test_paired_error_tables_include_period_branch_and_unknown_extremes():
    left = _severity_predictions()
    right = left.copy()
    metadata = _metadata(left)
    output = paired_error_table(
        left, right, "severity", metadata=metadata,
        groups=("class", "period", "weather_branch", "weather_heat"),
    )

    assert set(output["grouping"]) == {"class", "period", "weather_branch", "weather_heat"}
    class_rows = output[output.grouping.eq("class")]
    assert set(class_rows.group) == set(CLASSES)
    assert class_rows.delta_macro_f1.isna().all()
    period_rows = output[output.grouping.eq("period")]
    assert set(period_rows.group) == {"before_2025_10", "from_2025_10"}
    assert "unknown" in set(output.loc[output.grouping.eq("weather_branch"), "group"])
    assert "unknown" in set(output.loc[output.grouping.eq("weather_heat"), "group"])


def test_delay_error_classes_use_metadata_or_exact_thresholds():
    rows = pd.DataFrame({
        "journey_id": ["a", "b", "c", "d"],
        "prediction_time": ["0min"] * 4,
        "origin_departure_planned_local": pd.to_datetime(
            ["2025-01-06", "2025-01-07", "2025-01-13", "2025-01-14"]),
        "arrival_delay_min": [19.9, 20.0, 59.9, 60.0],
        "predicted_delay": [18.0, 21.0, 58.0, 61.0],
    })
    by_threshold = paired_error_table(rows, rows, "delay", groups=("class",))
    assert by_threshold.set_index("group").n_rows.to_dict() == {
        "normal_minor": 1, "moderate": 2, "severe": 1,
    }

    metadata = rows[["journey_id", "severity_class"]] if "severity_class" in rows else rows[["journey_id"]].copy()
    metadata["severity_class"] = ["moderate", "moderate", "severe", "severe"]
    by_metadata = paired_error_table(rows, rows, "delay", metadata=metadata, groups=("class",))
    assert by_metadata.set_index("group").n_rows.to_dict() == {"moderate": 2, "severe": 2}


def test_calibration_returns_reliability_counts_without_a_new_score():
    rows = _severity_predictions(4)
    reliability = calibration_reliability(rows, n_bins=5)

    assert len(reliability) == len(CLASSES) * 5
    assert reliability.groupby("class").n_rows.sum().to_dict() == {label: 4 for label in CLASSES}
    assert {"n_rows", "mean_predicted_probability", "observed_frequency"} <= set(reliability.columns)
    assert not {"brier", "ece", "calibration_error"} & set(reliability.columns)


def test_risk_examples_are_first_complete_journey_per_realized_class():
    rows = _severity_predictions(4, all_horizons=True)
    incomplete = rows[(rows.journey_id == "journey-0") & (rows.prediction_time != "0min")].copy()
    incomplete["journey_id"] = "journey-incomplete"
    incomplete["origin_departure_planned_local"] = pd.Timestamp("2025-01-01")
    rows = pd.concat([rows, incomplete], ignore_index=True)
    # Same-date tie: journey-0 sorts before journey-00 by ID.
    tied = rows[rows.journey_id.eq("journey-0")].copy()
    tied["journey_id"] = "journey-00"
    rows = pd.concat([rows, tied], ignore_index=True)

    result = example_risk_trajectories(rows)
    assert len(result) == len(CLASSES) * len(HORIZONS)
    chosen = result.groupby("severity_class").journey_id.first().to_dict()
    assert chosen[CLASSES[0]] == "journey-0"
    assert result.groupby("journey_id").prediction_time.apply(list).iloc[0] == list(HORIZONS)
    np.testing.assert_allclose(result.p_severe_or_canceled,
                               result.p_severe + result.p_canceled)


def test_module_import_does_not_read_any_dataset(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("final analysis import must not read data")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    importlib.reload(final_analysis)


def test_worker_rejects_missing_lock_or_mismatched_manifest(tmp_path, monkeypatch):
    import run_final

    def forbidden(*args, **kwargs):
        raise AssertionError("worker opened modeling rows before checking the evaluation")

    monkeypatch.setattr(run_final, "committed_registry", lambda path: {"finalists": [{}]})
    monkeypatch.setattr(run_final, "load_advanced_rows", forbidden)
    registry = tmp_path / "finalists.json"
    registry.write_text('{"finalists":[{}]}')
    output = tmp_path / "evaluation"
    output.mkdir()
    with pytest.raises(PermissionError, match="no active"):
        run_final.score_registered_arm(registry, tmp_path, output, 0)

    (output / ".evaluation.lock").touch()
    (output / "registration.json").write_text('{"finalists":[{"different":true}]}')
    with pytest.raises(PermissionError, match="different registration"):
        run_final.score_registered_arm(registry, tmp_path, output, 0)


def test_failed_final_evaluation_rejects_changed_provenance_on_resume(tmp_path, monkeypatch):
    import run_final

    development = {"source/modeling/run_final.py": "original"}
    registration = {"finalists": [{"model": "lightgbm"}], "development_fingerprints": development}
    registry = tmp_path / "finalists.json"
    registry.write_text(json.dumps(registration))
    monkeypatch.setattr(run_final, "committed_registry", lambda path: registration)
    fingerprint = {**development, "advanced/weather_holdout.parquet": "original"}
    monkeypatch.setattr(run_final, "evaluation_fingerprints",
                        lambda *args, splits=("cv", "holdout"): dict(development if splits == ("cv",) else fingerprint))
    calls = []

    def fail(*args):
        calls.append(True)
        raise RuntimeError("interrupted arm")

    monkeypatch.setattr(run_final, "run_final_arms", fail)
    with pytest.raises(RuntimeError, match="interrupted arm"):
        run_final.final_evaluation(registry, tmp_path, tmp_path)
    output = tmp_path / "advanced_holdout"
    assert not (output / ".evaluation.lock").exists()
    assert json.loads((output / "provenance.json").read_text()) == fingerprint
    fingerprint["advanced/weather_holdout.parquet"] = "changed"
    with pytest.raises(PermissionError, match="code or inputs changed"):
        run_final.final_evaluation(registry, tmp_path, tmp_path)
    assert len(calls) == 1
    (output / "provenance.json").unlink()
    (output / "_arms").mkdir()
    with pytest.raises(PermissionError, match="arms have no provenance"):
        run_final.final_evaluation(registry, tmp_path, tmp_path)
    assert len(calls) == 1


def test_holdout_gate_rejects_source_changes_after_registration(tmp_path, monkeypatch):
    import holdout_gate

    project = tmp_path / "frankfurt_munich"
    modeling = project / "modeling"
    modeling.mkdir(parents=True)
    source = modeling / "pipeline.py"
    source.write_text("registered source")
    path = modeling / "finalists.json"
    registration = {"finalists": [{}], "development_fingerprints": holdout_gate.source_fingerprints(project)}
    path.write_text(json.dumps(registration))
    committed = path.read_bytes()
    monkeypatch.setattr(holdout_gate, "HERE", project)
    monkeypatch.setattr(holdout_gate.subprocess, "run", lambda *args, **kwargs:
                        holdout_gate.subprocess.CompletedProcess(args, 0, stdout=committed))

    assert holdout_gate.committed_registration(path) == registration
    source.write_text("changed source")
    with pytest.raises(PermissionError, match="source changed after registration"):
        holdout_gate.committed_registration(path)


def test_first_final_evaluation_rejects_changed_cv_inputs(tmp_path, monkeypatch):
    import run_final

    key = "data/prediction_rows_cv.parquet"
    registration = {"finalists": [{"model": "lightgbm"}], "development_fingerprints": {key: "original"}}
    registry = tmp_path / "finalists.json"
    registry.write_text(json.dumps(registration))
    monkeypatch.setattr(run_final, "committed_registry", lambda path: registration)

    def fingerprints(*args, splits=("cv", "holdout")):
        assert splits == ("cv",)
        return {key: "changed"}

    monkeypatch.setattr(run_final, "evaluation_fingerprints", fingerprints)
    with pytest.raises(PermissionError, match="development code or inputs changed after registration"):
        run_final.final_evaluation(registry, tmp_path, tmp_path)
    assert not (tmp_path / "advanced_holdout").exists()


def test_evaluation_fingerprints_cover_source_and_both_data_splits(tmp_path, monkeypatch):
    import run_final

    source = tmp_path / "modeling"
    source.mkdir()
    (source / "pipeline.py").write_text("original source")
    data = tmp_path / "data"
    inputs = data / "advanced"
    inputs.mkdir(parents=True)
    for split in ("cv", "holdout"):
        for name in ("journeys", "prediction_rows"):
            (data / f"{name}_{split}.parquet").write_bytes(b"canonical rows")
        for name in ("timetable", "history", "weather", "reliability"):
            (inputs / f"{name}_{split}.parquet").write_bytes(b"derived inputs")
    monkeypatch.setattr(run_final, "HERE", source)
    monkeypatch.setattr(run_final, "DATA", data)
    reliability = next(iter(run_final.FEATURE_FILES))
    registration = {"finalists": [{"features": {"numeric": list(reliability), "categorical": []}}]}
    original = run_final.evaluation_fingerprints(registration, inputs)
    development = run_final.evaluation_fingerprints(registration, inputs, splits=("cv",))
    assert "advanced/reliability_holdout.parquet" in original
    assert development == {key: value for key, value in original.items() if not key.endswith("_holdout.parquet")}
    (source / "pipeline.py").write_text("updated source")
    (data / "prediction_rows_cv.parquet").write_bytes(b"updated rows")
    changed = run_final.evaluation_fingerprints(registration, inputs)
    assert {key for key in original if original[key] != changed[key]} == {
        "source/modeling/pipeline.py", "data/prediction_rows_cv.parquet",
    }


def test_parallel_registration_preserves_arm_order_and_serial_result(tmp_path, monkeypatch):
    import run_final

    (tmp_path / "PLAN.md").write_text("All development steps are complete.")
    monkeypatch.setattr(run_final, "HERE", tmp_path)
    monkeypatch.setattr(run_final, "git_commit", lambda: "test-revision")
    monkeypatch.setattr(run_final, "runtime_versions", lambda: {"python": "test"})
    monkeypatch.setattr(run_final, "evaluation_fingerprints", lambda *args, **kwargs: {"cv": "frozen"})
    results = tmp_path / "results"
    pairs = []
    for task, model, metric in (("delay", "lightgbm", "rmse_mean"),
                                 ("severity", "lightgbm_classifier", "log_loss_mean")):
        pair = {}
        for weather in (False, True):
            config = {"model": model, "task": task, "weather": weather, "step": "0.8",
                      "features": {"24h": {"numeric": ["example"], "categorical": []}}}
            scores = pd.DataFrame({metric: [1.0]}, index=pd.Index(["24h"], name="prediction_time"))
            pair[weather] = (results / f"{model}_{weather}", config, scores)
        pairs.append((task, "24h", pair))
    monkeypatch.setattr(run_final, "choose_finalists", lambda path: iter(pairs))
    monkeypatch.setattr(run_final, "tune_finalist", lambda spec, config, inputs: {
        "params": {"weather": spec["weather"]}, "tuning_policy": {"method": "test CV"}})

    def forbidden(*args, **kwargs):
        raise AssertionError("registration attempted to open modeling rows")

    monkeypatch.setattr(run_final, "load_advanced_rows", forbidden)
    workers = []

    class OrderedPool:
        def __init__(self, max_workers):
            workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, records):
            return map(function, records)

    monkeypatch.setattr(run_final, "ProcessPoolExecutor", OrderedPool)
    serial, parallel = tmp_path / "serial.json", tmp_path / "parallel.json"
    run_final.register(results, tmp_path, serial)
    run_final.register(results, tmp_path, parallel, jobs=2)
    assert workers == [2]
    assert serial.read_bytes() == parallel.read_bytes()
    registered = json.loads(parallel.read_text())
    assert registered["development_fingerprints"] == {"cv": "frozen"}
    finalists = registered["finalists"]
    assert [(spec["task"], spec["weather"]) for spec in finalists] == [
        ("delay", False), ("delay", True), ("severity", False), ("severity", True),
    ]


def test_registered_tuning_retains_shared_feature_policy(monkeypatch):
    import run_final

    monkeypatch.setattr(run_final, "tune_finalist", lambda *args: {
        "params": {"chosen": 1}, "tuning_policy": {"method": "five CV week folds"}})
    result = run_final.tune_registered_finalist(
        ({"task": "delay"}, {}, "auto_groups on the railway CV arm"), inputs_dir=Path("unused"))
    assert result["tuning_policy"] == {
        "method": "five CV week folds", "feature_selection": "auto_groups on the railway CV arm",
    }


@pytest.mark.parametrize("source_budget,expected", [(50, 20), (20, 20), (8, 8)])
def test_final_tuning_respects_reduced_budget(monkeypatch, source_budget, expected):
    import run_final

    counts = []

    def configs(space, n):
        counts.append(n)
        return [{"choice": i} for i in range(n)]

    monkeypatch.setattr(run_final, "sample_configs", configs)
    monkeypatch.setattr(run_final, "load_advanced_rows", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(run_final, "registered_factory", lambda spec: (None, [], []))
    monkeypatch.setattr(run_final, "search", lambda *args: (
        pd.DataFrame({"config_id": [2], "chosen": [True]}), None, None))
    spec = {"task": "delay", "prediction_time": "24h", "weather": False,
            "step": "0.8", "model": "lightgbm"}

    result = run_final.tune_finalist(spec, {"search": {"n_configs": source_budget}}, Path("unused"))

    assert counts == [expected]
    assert result["tuning_policy"]["n_configs"] == expected
    assert result["params"] == {"choice": 2}
