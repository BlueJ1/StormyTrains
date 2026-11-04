from __future__ import annotations

import importlib
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
