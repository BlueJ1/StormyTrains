from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import holdout_gate

from scheduled_timetable import (
    EXACT_PLAN_REQUESTS_FROM_UTC,
    ROW_KEY,
    build_timetable_features,
    build_timetable_split,
    plan_availability_utc,
    replacement_announcements_asof,
    service_keys_asof,
)


def test_plan_availability_masks_legacy_hour_labels_by_default():
    stamps = pd.Series(pd.to_datetime([
        "2025-11-02 23:00:00",
        "2025-11-03 03:32:22.619260",
    ], format="mixed"))

    available = plan_availability_utc(stamps)

    assert pd.isna(available.iloc[0])
    assert available.iloc[1] == EXACT_PLAN_REQUESTS_FROM_UTC
    sensitivity = plan_availability_utc(stamps, legacy_hour_label_offset=pd.Timedelta(minutes=45))
    assert sensitivity.iloc[0] == pd.Timestamp("2025-11-02 23:45:00", tz="UTC")


def test_timetable_features_use_latest_asof_snapshot_and_numpy_path():
    rows = pd.DataFrame({
        "journey_id": ["j1"], "prediction_time": ["0min"],
        "predicted_at": [pd.Timestamp("2025-11-03 10:00")],
    })
    context = pd.DataFrame({
        "journey_id": ["j1"], "current_origin_id": ["ride-2"],
        "scheduled_origin_id": ["ride-2"], "is_replacement_train": [False],
    })
    plans = pd.DataFrame([
        {
            "id": "ride-2", "snapshot_timestamp": pd.Timestamp("2025-11-03 08:00"),
            "available_at_utc": pd.Timestamp("2025-11-03 08:00", tz="UTC"),
            "station_name": "Frankfurt (Main) Hbf", "train_type": "ICE", "train_number": "1",
            "arrival_planned_time": pd.Timestamp("2025-11-03 09:30"),
            "departure_planned_time": pd.Timestamp("2025-11-03 09:31"),
            "arrival_planned_path": None,
            "departure_planned_path": np.array(["Stuttgart Hbf", "Ulm Hbf", "München Hbf"], dtype=object),
        },
        {
            "id": "ride-1", "snapshot_timestamp": pd.Timestamp("2025-11-03 08:00"),
            "available_at_utc": pd.Timestamp("2025-11-03 08:00", tz="UTC"),
            "station_name": "Mannheim Hbf", "train_type": "ICE", "train_number": "1",
            "arrival_planned_time": pd.Timestamp("2025-11-03 08:00"),
            "departure_planned_time": pd.Timestamp("2025-11-03 08:01"),
            "arrival_planned_path": None, "departure_planned_path": None,
        },
        {
            "id": "ride-2", "snapshot_timestamp": pd.Timestamp("2025-11-03 10:00"),
            "available_at_utc": pd.Timestamp("2025-11-03 10:00", tz="UTC"),
            "station_name": "Frankfurt (Main) Hbf", "train_type": "ICE", "train_number": "1",
            "arrival_planned_time": pd.Timestamp("2025-11-03 09:30"),
            "departure_planned_time": pd.Timestamp("2025-11-03 09:31"),
            "arrival_planned_path": None,
            "departure_planned_path": np.array(["Würzburg Hbf", "Nürnberg Hbf", "München Hbf"], dtype=object),
        },
    ])

    result = build_timetable_features(rows, context, plans)

    assert result.loc[0, "scheduled_train_type"] == "ICE"
    assert result.loc[0, "scheduled_starts_at_origin"] == False
    assert result.loc[0, "weather_branch"] == "stuttgart"
    assert result.loc[0, "planned_stops"] == 4
    assert result.loc[0, "route_id"] == (
        "Frankfurt (Main) Hbf | Stuttgart Hbf | Ulm Hbf | München Hbf"
    )
    assert result.loc[0, "upstream_planned_min"] == 89.0


def test_timetable_features_mask_when_no_exact_snapshot_is_available():
    rows = pd.DataFrame({
        "journey_id": ["j1"], "prediction_time": ["24h"],
        "predicted_at": [pd.Timestamp("2025-11-03 10:00")],
    })
    context = pd.DataFrame({
        "journey_id": ["j1"], "current_origin_id": ["ride-1"],
        "scheduled_origin_id": ["ride-1"], "is_replacement_train": [False],
    })
    legacy = pd.DataFrame([{
        "id": "ride-1", "snapshot_timestamp": pd.Timestamp("2025-11-02 23:00"),
        "available_at_utc": pd.NaT, "station_name": "Frankfurt (Main) Hbf",
        "train_type": "ICE", "train_number": "1",
        "arrival_planned_time": pd.NaT, "departure_planned_time": pd.Timestamp("2025-11-03 11:00"),
        "arrival_planned_path": None, "departure_planned_path": ["München Hbf"],
    }])

    result = build_timetable_features(rows, context, legacy)

    assert result[TIMETABLE_FIELDS].isna().all().all()


TIMETABLE_FIELDS = [
    "scheduled_train_type", "scheduled_starts_at_origin", "weather_branch",
    "planned_stops", "route_id", "upstream_planned_min",
]


def test_replacement_announcement_is_keyed_by_prediction_and_ordinary_is_false():
    rows = pd.DataFrame({
        "journey_id": ["ordinary", "replacement", "replacement"],
        "prediction_time": ["24h", "24h", "0min"],
        "predicted_at": pd.to_datetime([
            "2024-07-01 09:00", "2024-07-01 09:00", "2024-07-02 12:00",
        ]),
    })
    context = pd.DataFrame({
        "journey_id": ["ordinary", "replacement"],
        "current_origin_id": ["ordinary-1", "replacement-4"],
        "scheduled_origin_id": ["ordinary-1", "scheduled-2"],
        "scheduled_train_type": ["ICE", "ICE"],
        "scheduled_train_number": ["10", "595"],
        "replacement_train_type": [pd.NA, "ICE"],
        "replacement_train_number": [pd.NA, "2947"],
        "is_replacement_train": [False, True],
    })
    changes = pd.DataFrame([{
        "id": "replacement-4", "snapshot_timestamp": pd.Timestamp("2024-07-02 09:00"),
        "replaced_train_type": "ICE", "replaced_train_number": "595",
    }])

    announcements = replacement_announcements_asof(rows, context, changes)

    assert announcements.loc[0, "is_replacement"] == False
    assert announcements.loc[0, "replacement_announced"] == False
    assert announcements.loc[1, "is_replacement"] == True
    assert announcements.loc[1, "replacement_announced"] == False
    assert announcements.loc[2, "replacement_announced"] == True
    assert announcements.loc[2, "replacement_announced_at_utc"] == pd.Timestamp(
        "2024-07-02 09:45", tz="UTC"
    )

    keys = service_keys_asof(rows, pd.DataFrame(), context, announcements)
    assert keys.loc[1, "history_train_number"] == "595"
    assert keys.loc[2, "history_train_number"] == "2947"
    assert keys.loc[0, "service_key_is_replacement"] == False
    assert keys.loc[2, "service_key_is_replacement"] == True


def test_candidate_service_keys_keep_target_and_candidate_ids_distinct():
    rows = pd.DataFrame({
        "journey_id": ["target-a", "target-b"],
        "prediction_time": ["1h", "1h"],
        "predicted_at": pd.to_datetime(["2025-12-01 10:00", "2025-12-01 10:00"]),
        "candidate_journey_id": ["candidate-r", "candidate-r"],
    })
    context = pd.DataFrame({
        "journey_id": ["candidate-r"],
        "current_origin_id": ["candidate-r-origin"],
        "scheduled_origin_id": ["scheduled-origin"],
        "scheduled_train_type": ["ICE"],
        "scheduled_train_number": ["100"],
        "replacement_train_type": ["ICE"],
        "replacement_train_number": ["900"],
        "is_replacement_train": [True],
    })
    changes = pd.DataFrame([{
        "id": "candidate-r-origin",
        "snapshot_timestamp": pd.Timestamp("2025-12-01 08:00"),
        "replaced_train_type": "ICE", "replaced_train_number": "100",
    }])
    row_keys = ROW_KEY + ["candidate_journey_id"]

    announcements = replacement_announcements_asof(
        rows, context, changes, entity_col="candidate_journey_id", row_keys=row_keys
    )
    keys = service_keys_asof(
        rows, pd.DataFrame(), context, announcements,
        entity_col="candidate_journey_id", row_keys=row_keys,
    )

    assert announcements["replacement_announced"].tolist() == [True, True]
    assert keys["journey_id"].tolist() == ["target-a", "target-b"]
    assert keys["candidate_journey_id"].tolist() == ["candidate-r", "candidate-r"]
    assert keys["history_train_number"].tolist() == ["900", "900"]


def test_holdout_gate_failure_happens_before_any_input_read(monkeypatch, tmp_path):
    def reject_registration(*args, **kwargs):
        raise PermissionError("synthetic registration failure")

    def forbidden_read(*args, **kwargs):
        raise AssertionError("an input was read before the gate")

    monkeypatch.setattr(holdout_gate, "committed_registration", reject_registration)
    monkeypatch.setattr(pd, "read_parquet", forbidden_read)

    with pytest.raises(PermissionError, match="synthetic registration failure"):
        build_timetable_split("holdout", output_dir=tmp_path, plan_dir=tmp_path)


def test_gated_synthetic_holdout_build_projects_only_metadata(monkeypatch, tmp_path):
    reads = []
    journey = pd.DataFrame([{
        "journey_id": "synthetic-journey", "train_type": "ICE", "train_number": "123",
        "origin_departure_planned_local": pd.Timestamp("2024-07-01 10:00"),
    }])
    rows = pd.DataFrame([{
        "journey_id": "synthetic-journey", "prediction_time": "24h",
        "predicted_at": pd.Timestamp("2024-06-30 10:00"),
    }])

    def read_parquet(path, columns=None, **kwargs):
        reads.append((Path(path).name, tuple(columns or ())))
        if Path(path).name == "journeys_holdout.parquet":
            assert tuple(columns) == (
                "journey_id", "train_type", "train_number", "origin_departure_planned_local"
            )
            return journey.copy()
        if Path(path).name == "prediction_rows_holdout.parquet":
            assert tuple(columns) == tuple(ROW_KEY)
            return rows.copy()
        raise AssertionError(f"unexpected parquet read: {path}")

    stops = pd.DataFrame([{
        "id": "synthetic-ride-1", "journey_id": "synthetic-journey",
        "station_name": "Frankfurt (Main) Hbf", "train_type": "ICE", "train_number": "123",
        "is_replacement_train": False, "replaced_train_type": pd.NA,
        "replaced_train_number": pd.NA, "train_line_ride_id": "synthetic-ride",
        "train_line_station_num": 1, "stop_index": 0,
        "arrival_planned_time": pd.NaT, "departure_planned_time": pd.Timestamp("2024-07-01 10:00"),
    }])
    monkeypatch.setattr(holdout_gate, "committed_registration", lambda path: {"synthetic": True})
    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    monkeypatch.setattr("scheduled_timetable._cv_stops", lambda ids: stops.copy())
    monkeypatch.setattr(
        "scheduled_timetable._cv_upstream_ids",
        lambda ids: pd.DataFrame(columns=["id", "journey_id", "train_line_station_num", "station_name"]),
    )
    monkeypatch.setattr(
        "scheduled_timetable._find_replacement_origin_ids",
        lambda *args, **kwargs: pd.DataFrame(columns=["journey_id", "scheduled_origin_id", "candidate_id_count"]),
    )
    monkeypatch.setattr(
        "scheduled_timetable._read_change_references",
        lambda *args, **kwargs: pd.DataFrame(columns=["id", "snapshot_timestamp", "replaced_train_type", "replaced_train_number"]),
    )

    timetable, audit = build_timetable_split(
        "holdout", output_dir=tmp_path, plan_dir=tmp_path, change_dir=tmp_path
    )

    assert reads == [
        ("journeys_holdout.parquet", ("journey_id", "train_type", "train_number", "origin_departure_planned_local")),
        ("prediction_rows_holdout.parquet", tuple(ROW_KEY)),
    ]
    assert len(timetable) == 1
    assert audit["holdout_gate_passed"] is True
    assert audit["holdout_inputs_projected_without_targets"] is True
    assert (tmp_path / "timetable_holdout.parquet").is_file()
    assert (tmp_path / "replacement_announcements_holdout.parquet").is_file()
