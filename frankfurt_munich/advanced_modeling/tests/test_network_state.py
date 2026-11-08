from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network_state import (  # noqa: E402
    DESTINATION,
    ORIGIN,
    _event_intervals,
    _run_scope_segments,
    _scope_deltas,
    _shares_at_cutoffs,
    _timeline_states,
    build_network_features,
)
from history_features import NETWORK_STATE_COLUMNS, ROW_KEY  # noqa: E402


PANEL = {"Hanau Hbf", ORIGIN, DESTINATION}


def rows_at(*local_times):
    return pd.DataFrame({
        "journey_id": [f"target-{i}" for i in range(len(local_times))],
        "prediction_time": ["0min"] * len(local_times),
        "predicted_at": pd.to_datetime(local_times),
    })


def plan_row(stop_id, station, *, timestamp="2025-11-04 08:00:00", train_type="ICE",
             arrival="2025-11-04 10:00:00", departure="2025-11-04 10:01:00"):
    return {
        "id": stop_id, "station_name": station,
        "snapshot_timestamp": pd.Timestamp(timestamp), "train_type": train_type,
        "train_number": "100", "arrival_planned_time": pd.Timestamp(arrival),
        "departure_planned_time": pd.Timestamp(departure),
    }


def change_row(stop_id, station, *, timestamp, arrival=None, cancellation=None,
               status=None, train_type="ICE", departure=None, departure_cancel=None,
               departure_status=None):
    return {
        "id": stop_id, "station_name": station,
        "snapshot_timestamp": pd.Timestamp(timestamp), "train_type": train_type,
        "train_number": "100", "arrival_change_time": pd.NaT if arrival is None else pd.Timestamp(arrival),
        "arrival_cancellation_time": pd.NaT if cancellation is None else pd.Timestamp(cancellation),
        "arrival_change_status": status,
        "departure_change_time": pd.NaT if departure is None else pd.Timestamp(departure),
        "departure_cancellation_time": pd.NaT if departure_cancel is None else pd.Timestamp(departure_cancel),
        "departure_change_status": departure_status,
    }


def features(rows, changes, plans):
    states = _timeline_states(pd.DataFrame(changes), pd.DataFrame(plans))
    intervals = _event_intervals(states, PANEL)
    segments = _run_scope_segments(intervals)
    return _shares_at_cutoffs(rows, _scope_deltas([segments]))


def test_delay_activation_expiration_and_next_snapshot_update_are_cutoff_safe():
    stop = "run-a-2511041000-1"
    changes = [
        change_row(stop, "Hanau Hbf", timestamp="2025-11-04 08:30:00",
                   arrival="2025-11-04 10:25:00"),
        # A later report revises the event below the disruption threshold.
        change_row(stop, "Hanau Hbf", timestamp="2025-11-04 09:40:00",
                   arrival="2025-11-04 10:10:00"),
    ]
    plans = [plan_row(stop, "Hanau Hbf")]
    rows = rows_at("2025-11-04 10:24", "2025-11-04 10:25",
                   "2025-11-04 10:39", "2025-11-04 10:40",
                   "2025-11-04 11:09", "2025-11-04 11:10")

    result = features(rows, changes, plans)

    share = NETWORK_STATE_COLUMNS[0]
    assert pd.isna(result.loc[0, share])  # Event time has not arrived.
    assert result.loc[1, share] == 1.0   # Inclusive activation at the event time.
    assert result.loc[2, share] == 1.0
    assert result.loc[3, share] == 0.0   # The new state starts at its receipt time.
    assert result.loc[4, share] == 0.0
    assert pd.isna(result.loc[5, share])  # Half-open end at event time + one hour.


def test_plan_update_recomputes_event_delay_and_ends_the_previous_state():
    stop = "run-p-2511041000-1"
    changes = [change_row(stop, "Hanau Hbf", timestamp="2025-11-04 08:30:00",
                          arrival="2025-11-04 10:25:00")]
    plans = [
        plan_row(stop, "Hanau Hbf", timestamp="2025-11-04 08:00:00",
                 arrival="2025-11-04 10:00:00"),
        plan_row(stop, "Hanau Hbf", timestamp="2025-11-04 09:40:00",
                 arrival="2025-11-04 10:20:00"),
    ]
    rows = rows_at("2025-11-04 10:39", "2025-11-04 10:40")

    result = features(rows, changes, plans)

    share = NETWORK_STATE_COLUMNS[0]
    assert result.loc[0, share] == 1.0
    assert result.loc[1, share] == 0.0


def test_cancellation_counts_until_a_withdrawal_snapshot():
    stop = "run-c-2511041000-1"
    changes = [
        change_row(stop, ORIGIN, timestamp="2025-11-04 08:40:00",
                   cancellation="2025-11-04 09:20:00", status="c"),
        change_row(stop, ORIGIN, timestamp="2025-11-04 09:00:00",
                   cancellation="2025-11-04 09:20:00", status="p"),
    ]
    rows = rows_at("2025-11-04 09:50", "2025-11-04 10:00")

    result = features(rows, changes, [])

    frankfurt = NETWORK_STATE_COLUMNS[1]
    assert result.loc[0, frankfurt] == 1.0
    assert pd.isna(result.loc[1, frankfurt])


def test_plan_only_row_does_not_establish_outcome_and_run_is_deduplicated():
    ids = ["run-a-2511041000-1", "run-a-2511041000-2", "run-b-2511041000-1"]
    changes = [
        change_row(ids[0], "Hanau Hbf", timestamp="2025-11-04 08:30:00",
                   arrival="2025-11-04 10:25:00"),
        change_row(ids[1], "Hanau Hbf", timestamp="2025-11-04 08:30:00",
                   arrival="2025-11-04 10:30:00"),
        change_row(ids[2], "Hanau Hbf", timestamp="2025-11-04 08:30:00",
                   arrival="2025-11-04 10:00:00"),
    ]
    plans = [
        plan_row(ids[0], "Hanau Hbf", arrival="2025-11-04 10:00:00"),
        plan_row(ids[1], "Hanau Hbf", arrival="2025-11-04 10:00:00"),
        plan_row(ids[2], "Hanau Hbf", arrival="2025-11-04 10:00:00"),
        plan_row("run-old-2511041000-1", "Hanau Hbf",
                 timestamp="2025-11-02 08:00:00", arrival="2025-11-04 10:00:00"),
    ]
    rows = rows_at("2025-11-04 10:45")

    result = features(rows, changes, plans)

    share = NETWORK_STATE_COLUMNS[0]
    assert result.loc[0, share] == pytest.approx(0.5)
    assert pd.isna(result.loc[0, NETWORK_STATE_COLUMNS[1]])
    assert pd.isna(result.loc[0, NETWORK_STATE_COLUMNS[2]])


def test_legacy_plan_is_static_and_can_establish_delay():
    stop = "run-old-2511041000-1"
    changes = [change_row(stop, "Hanau Hbf", timestamp="2025-11-04 08:30:00",
                          arrival="2025-11-04 10:25:00")]
    legacy = [plan_row(stop, "Hanau Hbf", timestamp="2025-11-02 08:00:00")]
    rows = rows_at("2025-11-04 10:45")

    result = features(rows, changes, legacy)

    assert result.loc[0, NETWORK_STATE_COLUMNS[0]] == 1.0


def test_earliest_legacy_plan_is_static_until_an_exact_update_arrives():
    stop = "run-old-2511041000-1"
    changes = [change_row(stop, "Hanau Hbf", timestamp="2025-11-04 08:30:00",
                          arrival="2025-11-04 10:25:00")]
    plans = [
        plan_row(stop, "Hanau Hbf", timestamp="2025-11-02 08:00:00",
                 arrival="2025-11-04 10:00:00"),
        # Later legacy target-hour labels do not replace the earliest static plan.
        plan_row(stop, "Hanau Hbf", timestamp="2025-11-02 09:00:00",
                 arrival="2025-11-04 10:20:00"),
        plan_row(stop, "Hanau Hbf", timestamp="2025-11-04 09:40:00",
                 arrival="2025-11-04 10:20:00"),
    ]
    rows = rows_at("2025-11-04 10:39", "2025-11-04 10:40")

    result = features(rows, changes, plans)

    share = NETWORK_STATE_COLUMNS[0]
    assert result.loc[0, share] == 1.0
    assert result.loc[1, share] == 0.0


def test_no_unintended_prediction_columns_and_unique_row_alignment():
    empty = pd.DataFrame(columns=["id", "snapshot_timestamp"])
    rows = rows_at("2025-11-04 10:45", "2025-11-04 11:00")

    result = features(rows, empty, empty)

    assert result[ROW_KEY].equals(rows[ROW_KEY])
    assert list(result.columns) == [*ROW_KEY, *NETWORK_STATE_COLUMNS]
    assert result[NETWORK_STATE_COLUMNS].isna().all().all()


def test_missing_plan_archive_fails_instead_of_building_cancellation_only_features(tmp_path):
    change_dir = tmp_path / "change"
    change_dir.mkdir()
    (change_dir / "data-2025-11.parquet").touch()
    rows = rows_at("2025-11-04 10:45")

    with pytest.raises(FileNotFoundError, match="no monthly plan snapshots"):
        build_network_features(rows, change_dir=change_dir, plan_dir=tmp_path / "missing",
                               stations=sorted(PANEL))


def test_holdout_registration_gate_runs_before_any_prediction_file_read(monkeypatch, tmp_path):
    import holdout_gate
    import network_state

    def reject(_path):
        raise PermissionError("registration is not committed")

    monkeypatch.setattr(holdout_gate, "committed_registration", reject)
    monkeypatch.setattr(network_state, "_load_prediction_rows",
                        lambda *_args: pytest.fail("prediction rows read before gate"))
    monkeypatch.setattr(sys, "argv", ["network_state.py", "--split", "holdout",
                                       "--data-dir", str(tmp_path)])

    with pytest.raises(PermissionError, match="registration"):
        network_state.main()
