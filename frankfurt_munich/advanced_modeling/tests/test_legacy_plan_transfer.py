from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_corridor_inputs  # noqa: E402
import network_transfer  # noqa: E402


def test_auxiliary_rows_use_first_legacy_schedule_even_after_cutoff(monkeypatch):
    monkeypatch.setattr(build_corridor_inputs, "HORIZONS", {"24h": pd.Timedelta(hours=24)})
    monkeypatch.setattr(build_corridor_inputs, "CURRENT_DELAY_TIMES", ())
    departure = pd.Timestamp("2025-01-03 10:00:00")
    origin_id = "run-2501031000-1"
    rows = pd.DataFrame([{
        "journey_id": "run-2501031000", "id_origin": origin_id,
        "id_destination": "run-2501031000-2", "corridor": "berlin",
        build_corridor_inputs.TIME_COLUMN: departure,
        build_corridor_inputs.DESTINATION: departure + pd.Timedelta(hours=4),
    }])
    plans = pd.DataFrame([
        {"id": origin_id, "snapshot_timestamp": "2025-01-03T11:00:00Z",
         "train_type": "ICE"},
        {"id": origin_id, "snapshot_timestamp": "2025-01-03T11:15:00Z",
         "train_type": "IC"},
    ])

    result = build_corridor_inputs.prediction_rows(
        rows,
        pd.DataFrame(columns=["journey_id", "id", "train_line_station_num", "station_name"]),
        pd.DataFrame(), plans)

    assert len(result) == 1
    assert result.loc[0, "train_type"] == "ICE"
    assert result.loc[0, "starts_at_origin"]
    assert pd.isna(result.loc[0, "current_delay"])


def test_corridor_builder_reads_and_prepares_legacy_month_plans(monkeypatch, tmp_path):
    monthly_dir, plan_dir, change_dir = (tmp_path / name for name in ("monthly", "plans", "changes"))
    for directory in (monthly_dir, plan_dir, change_dir):
        directory.mkdir()
    archive = monthly_dir / "data-2024-01.parquet"
    archive.touch()
    (plan_dir / archive.name).touch()
    (change_dir / archive.name).touch()

    journey = pd.DataFrame([{
        "corridor": "berlin", "journey_id": "run-2401010900",
        "id_origin": "run-2401010900-1", "id_destination": "run-2401010900-2",
        "origin_sequence": 1,
        build_corridor_inputs.TIME_COLUMN: pd.Timestamp("2024-01-01 09:00"),
        build_corridor_inputs.DESTINATION: pd.Timestamp("2024-01-01 13:00"),
    }])
    monkeypatch.setattr(build_corridor_inputs, "read_assignment", lambda: pd.DataFrame({
        "split": ["cv"], "week_start": [pd.Timestamp("2024-01-01")],
    }))
    monkeypatch.setattr(build_corridor_inputs, "source_segments", lambda *args: journey.copy())
    monkeypatch.setattr(build_corridor_inputs, "source_journeys", lambda paths, segments, weeks: segments.copy())
    monkeypatch.setattr(build_corridor_inputs, "read_upstream", lambda *args: pd.DataFrame(
        columns=["journey_id", "id", "train_line_station_num"]))
    calls = []

    def read_ids(ids, *, kind, **kwargs):
        calls.append((kind, tuple(sorted(ids))))
        if kind == "change":
            return pd.DataFrame()
        return pd.DataFrame([
            {"id": "run-2401010900-1", "snapshot_timestamp": "2024-01-02T11:00:00Z",
             "train_type": "ICE"},
        ])

    monkeypatch.setattr(build_corridor_inputs, "_read_snapshot_ids_in_months", read_ids)
    seen = []

    def prediction_rows(segment, stops, changes, plans):
        seen.append(plans)
        return pd.DataFrame([{
            "corridor": "berlin", "journey_id": "run-2401010900", "prediction_time": "24h",
            "predicted_at": pd.Timestamp("2023-12-31 09:00"), "current_delay": np.nan,
        }])

    monkeypatch.setattr(build_corridor_inputs, "prediction_rows", prediction_rows)

    def write_stub(self, path, **kwargs):
        Path(path).write_bytes(b"synthetic parquet placeholder")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_stub)
    output = tmp_path / "advanced" / "corridor_training_cv.parquet"

    build_corridor_inputs.build(output, plan_dir, change_dir, monthly_dir)

    assert any(kind == "plan" for kind, _ in calls)
    assert len(seen) == 1
    assert seen[0].available_at_utc.iloc[0] == pd.Timestamp("1900-01-01", tz="UTC")
    assert output.is_file()


def _network_fixture(day: str, *, exact_plans: bool):
    date = pd.Timestamp(day)
    run = f"train-{date.strftime('%y%m%d0900')}"
    source_id, target_id = f"{run}-1", f"{run}-2"
    monthly = pd.DataFrame([
        {"id": source_id, "train_type": "ICE",
         "arrival_planned_time": date + pd.Timedelta(hours=9),
         "arrival_change_time": pd.NaT, "arrival_is_canceled": False,
         "is_additional_stop": False, "has_plan_record": True},
        {"id": target_id, "train_type": "ICE",
         "arrival_planned_time": date + pd.Timedelta(hours=10),
         "arrival_change_time": date + pd.Timedelta(hours=10, minutes=30),
         "arrival_is_canceled": False, "is_additional_stop": False,
         "has_plan_record": True},
    ])
    snapshot = (pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=7, minutes=45)
                if exact_plans else pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=7))
    changes = pd.DataFrame([{
        "id": source_id, "snapshot_timestamp": snapshot,
        "arrival_change_time": date + pd.Timedelta(hours=8, minutes=30),
        "arrival_change_status": None, "arrival_cancellation_time": pd.NaT,
    }])
    plans = [
        {"id": source_id, "snapshot_timestamp": "2025-11-02T08:00:00Z",
         "arrival_planned_time": date + pd.Timedelta(hours=9)},
        {"id": target_id, "snapshot_timestamp": "2025-11-02T08:00:00Z",
         "arrival_planned_time": date + pd.Timedelta(hours=10)},
    ]
    if exact_plans:
        plans.extend([
            {"id": source_id, "snapshot_timestamp": f"{day}T07:30:00Z",
             "arrival_planned_time": date + pd.Timedelta(hours=8, minutes=50)},
            {"id": target_id, "snapshot_timestamp": f"{day}T07:30:00Z",
             "arrival_planned_time": date + pd.Timedelta(hours=10, minutes=20)},
            {"id": source_id, "snapshot_timestamp": f"{day}T08:00:00Z",
             "arrival_planned_time": date + pd.Timedelta(hours=9, minutes=10)},
            {"id": target_id, "snapshot_timestamp": f"{day}T08:00:00Z",
             "arrival_planned_time": date + pd.Timedelta(hours=11)},
        ])
    else:
        # Both legacy fetches are after the source report cutoff; only the first
        # static timetable is eligible under the selected legacy policy.
        plans.extend([
            {"id": source_id, "snapshot_timestamp": f"{day}T08:10:00Z",
             "arrival_planned_time": date + pd.Timedelta(hours=9, minutes=30)},
            {"id": target_id, "snapshot_timestamp": f"{day}T08:10:00Z",
             "arrival_planned_time": date + pd.Timedelta(hours=10, minutes=30)},
        ])
    return monthly, changes, pd.DataFrame(plans), source_id, target_id


def _paired_month(monkeypatch, tmp_path, day: str, *, exact_plans: bool):
    monthly, changes, plans, source_id, target_id = _network_fixture(day, exact_plans=exact_plans)
    change_dir, plan_dir = tmp_path / "changes", tmp_path / "plans"
    monkeypatch.setattr(network_transfer.pq, "read_table",
                        lambda *args, **kwargs: SimpleNamespace(to_pandas=lambda: monthly.copy()))

    def archive_rows(directory, month, ids, columns):
        frame = changes if directory == change_dir else plans
        return frame.loc[frame.id.isin(set(ids)), columns].copy()

    monkeypatch.setattr(network_transfer, "archive_rows", archive_rows)
    result = network_transfer.paired_month(
        tmp_path / "data-2025-11.parquet", plan_dir, change_dir,
        [pd.Timestamp(day).normalize() - pd.Timedelta(days=pd.Timestamp(day).dayofweek)],
    )
    return result, source_id, target_id


def test_network_pairs_accept_legacy_timetable_fetched_after_cutoff(monkeypatch, tmp_path):
    result, source_id, target_id = _paired_month(
        monkeypatch, tmp_path, "2025-11-02", exact_plans=False)

    assert len(result) == 1
    assert (result.loc[0, "id"], result.loc[0, "target_id"]) == (source_id, target_id)
    assert result.loc[0, "last_stop_delay"] == -30
    assert result.loc[0, "remaining_planned_min"] == 60


def test_network_pairs_use_latest_exact_plan_at_cutoff(monkeypatch, tmp_path):
    result, _, _ = _paired_month(monkeypatch, tmp_path, "2025-11-10", exact_plans=True)

    assert len(result) == 1
    assert result.loc[0, "last_stop_delay"] == -20
    assert result.loc[0, "remaining_planned_min"] == 90


def test_network_build_defaults_to_all_source_months(monkeypatch, tmp_path):
    class MonthlySource:
        def glob(self, pattern):
            return [Path("data-2024-08.parquet"), Path("data-2025-12.parquet")]

    seen = []
    monkeypatch.setattr(network_transfer, "SOURCE", MonthlySource())
    monkeypatch.setattr(network_transfer, "DATA", tmp_path)
    monkeypatch.setattr(network_transfer, "read_assignment", lambda: pd.DataFrame({
        "split": ["cv"], "week_start": [pd.Timestamp("2024-08-05")],
    }))
    monkeypatch.setattr(network_transfer, "paired_month", lambda path, *args:
                        seen.append(path.name) or pd.DataFrame({"id": [path.name], "target_id": ["target"]}))
    written = []

    def write_stub(self, path, **kwargs):
        path = Path(path)
        path.write_bytes(b"synthetic parquet placeholder")
        written.append(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_stub)
    monkeypatch.setattr(sys, "argv", ["network_transfer.py"])

    network_transfer.main()

    assert seen == ["data-2024-08.parquet", "data-2025-12.parquet"]
    assert len(written) == 1
