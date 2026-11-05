from __future__ import annotations

import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import history_features
from history_features import (
    build_delay_dynamics,
    build_history_dynamics_split,
    build_upstream_events_asof,
    mask_unlinked_replacement_rows,
    cutoff_instants,
    latest_snapshot_asof,
    request_upper_bound_utc,
    ROW_KEY,
)
from recent_reliability import (_candidate_outcomes_asof, _candidate_pairs_for_target_batch,
                                build_recent_reliability, build_reliability_file)


def test_legacy_snapshot_request_uses_latest_possible_block_time():
    stamps = pd.Series(pd.to_datetime([
        "2025-11-02 23:00:00",
        "2025-11-03 03:32:22",
    ], format="mixed"))

    upper = request_upper_bound_utc(stamps)

    assert upper.iloc[0] == pd.Timestamp("2025-11-02 23:45:00", tz="UTC")
    assert upper.iloc[1] == pd.Timestamp("2025-11-03 03:32:22", tz="UTC")


def test_latest_snapshot_asof_ignores_later_observation():
    queries = pd.DataFrame({
        "_query_id": [0], "id": ["stop-1"],
        "cutoff_utc": [pd.Timestamp("2025-01-01 08:00:00", tz="UTC")],
    })
    snapshots = pd.DataFrame({
        "id": ["stop-1", "stop-1"],
        "snapshot_timestamp": pd.to_datetime([
            "2025-01-01 07:00:00", "2025-01-01 08:10:00",
        ]),
        "arrival_change_time": pd.to_datetime([
            "2025-01-01 08:20:00", "2025-01-01 08:55:00",
        ]),
    })

    result = latest_snapshot_asof(queries, snapshots)

    assert result.loc[0, "arrival_change_time"] == pd.Timestamp("2025-01-01 08:20:00")


def test_cutoff_uses_earlier_occurrence_of_repeated_local_hour():
    local = pd.Series(pd.to_datetime(["2025-10-26 02:30:00"]))

    assert cutoff_instants(local).iloc[0] == pd.Timestamp("2025-10-26 00:30:00", tz="UTC")


def test_delay_dynamics_uses_only_snapshot_and_plan_state_at_cutoff():
    cutoff = pd.Timestamp("2025-12-01 09:00:00")  # 08:00 UTC
    rows = pd.DataFrame([{
        "journey_id": "journey-1", "prediction_time": "1h",
        "predicted_at": cutoff,
    }])
    # These monthly changed times are deliberately late final values. The builder
    # must use the snapshot archive instead.
    upstream = pd.DataFrame([
        {"journey_id": "journey-1", "id": "stop-1", "station_name": "Hanau",
         "train_line_station_num": 3,
             "arrival_change_time": pd.Timestamp("2025-12-01 08:55:00")},
        {"journey_id": "journey-1", "id": "frankfurt", "station_name": "Frankfurt (Main) Hbf",
         "train_line_station_num": 4,
             "arrival_change_time": pd.Timestamp("2025-12-01 10:20:00")},
    ])
    changes = pd.DataFrame([
        {"id": "stop-1", "snapshot_timestamp": pd.Timestamp("2025-12-01 07:00:00"),
         "arrival_change_time": pd.Timestamp("2025-12-01 08:20:00"),
         "arrival_cancellation_time": pd.NaT, "arrival_change_status": None,
         "departure_change_time": pd.NaT, "departure_cancellation_time": pd.NaT,
         "departure_change_status": None},
        # This update reports an earlier event time, but it arrived after cutoff.
        {"id": "stop-1", "snapshot_timestamp": pd.Timestamp("2025-12-01 08:10:00"),
         "arrival_change_time": pd.Timestamp("2025-12-01 08:55:00"),
         "arrival_cancellation_time": pd.NaT, "arrival_change_status": None,
         "departure_change_time": pd.NaT, "departure_cancellation_time": pd.NaT,
         "departure_change_status": None},
        {"id": "frankfurt", "snapshot_timestamp": pd.Timestamp("2025-12-01 07:00:00"),
         "arrival_change_time": pd.NaT, "arrival_cancellation_time": pd.NaT,
         "arrival_change_status": None, "departure_change_time": pd.NaT,
         "departure_cancellation_time": pd.NaT, "departure_change_status": None},
    ])
    plans = pd.DataFrame([
        {"id": "stop-1", "station_name": "Hanau",
         "snapshot_timestamp": pd.Timestamp("2025-12-01 06:50:00"),
         "arrival_planned_time": pd.Timestamp("2025-12-01 08:10:00"),
         "departure_planned_time": pd.Timestamp("2025-12-01 08:30:00")},
        # This final timetable revision is also too late for the target cutoff.
        {"id": "stop-1", "station_name": "Hanau",
         "snapshot_timestamp": pd.Timestamp("2025-12-01 08:10:00"),
         "arrival_planned_time": pd.Timestamp("2025-12-01 08:40:00"),
         "departure_planned_time": pd.Timestamp("2025-12-01 08:50:00")},
        {"id": "frankfurt", "station_name": "Frankfurt (Main) Hbf",
         "snapshot_timestamp": pd.Timestamp("2025-12-01 06:50:00"),
         "arrival_planned_time": pd.Timestamp("2025-12-01 09:55:00"),
         "departure_planned_time": pd.Timestamp("2025-12-01 10:00:00")},
    ])

    events = build_upstream_events_asof(rows, upstream, changes, plans)
    arrival = events.loc[
        events.event_id.eq("stop-1") & events.event_kind.eq("arrival")
    ].iloc[0]
    result = build_delay_dynamics(rows, upstream, changes, plans).iloc[0]

    assert arrival["reported_time"] == pd.Timestamp("2025-12-01 08:20:00")
    assert arrival["event_delay_min"] == pytest.approx(10.0)
    assert arrival["event_report_visible"]
    assert arrival["event_occurred_by_cutoff"]
    assert arrival["change_observed_at_utc"] == pd.Timestamp("2025-12-01 07:00:00", tz="UTC")
    assert result["current_delay"] == pytest.approx(30.0)  # waits through planned departure
    assert result["upstream_stops_reported"] == 1
    assert result["minutes_since_last_report"] == pytest.approx(60.0)
    assert result["planned_minutes_remaining_before_frankfurt"] == pytest.approx(110.0)
    assert pd.isna(result["current_delay_change_min"])


def test_delay_dynamics_preserves_negative_delay_while_waiting_at_a_stop():
    rows = pd.DataFrame([{
        "journey_id": "journey-early", "prediction_time": "1h",
        "predicted_at": pd.Timestamp("2025-12-01 09:00:00"),
    }])
    upstream = pd.DataFrame([
        {"journey_id": "journey-early", "id": "early-stop-1",
         "station_name": "Hanau", "train_line_station_num": 3},
        {"journey_id": "journey-early", "id": "early-stop-2",
         "station_name": "Frankfurt (Main) Hbf", "train_line_station_num": 4},
    ])
    changes = pd.DataFrame([
        {"id": "early-stop-1", "station_name": "Hanau",
         "snapshot_timestamp": pd.Timestamp("2025-12-01 07:00:00"),
         "arrival_change_time": pd.Timestamp("2025-12-01 08:40:00"),
         "arrival_cancellation_time": pd.NaT, "arrival_change_status": None,
         "departure_change_time": pd.NaT, "departure_cancellation_time": pd.NaT,
         "departure_change_status": None},
        {"id": "early-stop-2", "station_name": "Frankfurt (Main) Hbf",
         "snapshot_timestamp": pd.Timestamp("2025-12-01 07:00:00"),
         "arrival_change_time": pd.NaT, "arrival_cancellation_time": pd.NaT,
         "arrival_change_status": None, "departure_change_time": pd.NaT,
         "departure_cancellation_time": pd.NaT, "departure_change_status": None},
    ])
    plans = pd.DataFrame([
        {"id": "early-stop-1", "station_name": "Hanau",
         "snapshot_timestamp": pd.Timestamp("2025-12-01 07:00:00"),
         "arrival_planned_time": pd.Timestamp("2025-12-01 08:50:00"),
         "departure_planned_time": pd.Timestamp("2025-12-01 09:30:00")},
        {"id": "early-stop-2", "station_name": "Frankfurt (Main) Hbf",
         "snapshot_timestamp": pd.Timestamp("2025-12-01 07:00:00"),
         "arrival_planned_time": pd.Timestamp("2025-12-01 09:55:00"),
         "departure_planned_time": pd.Timestamp("2025-12-01 10:00:00")},
    ])

    result = build_delay_dynamics(rows, upstream, changes, plans).iloc[0]

    assert result["current_delay"] == pytest.approx(-10.0)


def test_delay_dynamics_rejects_non_cv_rows_when_split_is_present():
    rows = pd.DataFrame([{
        "journey_id": "journey-1", "prediction_time": "1h",
        "predicted_at": pd.Timestamp("2025-01-01 09:00:00"), "split": "holdout",
    }])
    empty = pd.DataFrame()

    with pytest.raises(PermissionError, match="CV"):
        build_delay_dynamics(rows, empty, empty, empty)


def test_recent_reliability_uses_only_known_prior_outcomes_in_each_window():
    rows = pd.DataFrame([{
        "journey_id": "target", "prediction_time": "1h",
        "predicted_at": pd.Timestamp("2025-01-10 12:00:00"),
        "history_train_type": "ICE", "history_train_number": "100",
    }])
    target_key = {
        "journey_id": "target", "prediction_time": "1h",
        "predicted_at": pd.Timestamp("2025-01-10 12:00:00"),
    }
    def state(journey, departure, train, delay, cancelled=False, known=True,
              outcome_time=None):
        departure_utc = pd.Timestamp(departure, tz="UTC")
        return {
            **target_key, "candidate_journey_id": journey,
            "history_train_type": "ICE", "history_train_number": train,
            "origin_departure_utc": departure_utc,
            "outcome_time_utc": (pd.Timestamp(outcome_time, tz="UTC")
                                 if outcome_time else departure_utc + pd.Timedelta(hours=3)),
            "outcome_known": known, "cancelled": cancelled,
            "arrival_delay_min": delay,
        }

    outcomes = pd.DataFrame([
        state("same-1", "2025-01-09 12:00", "100", 65),
        state("same-2", "2025-01-05 10:00", "100", 20),
        state("same-3", "2025-01-01 12:00", "100", 60),
        state("other-cancel", "2025-01-10 10:00", "200", float("nan"), True,
              outcome_time="2025-01-10 10:30"),
        state("unknown", "2025-01-10 10:30", "100", 100, known=False),
        state("target", "2025-01-10 10:45", "100", 100),
    ])

    result = build_recent_reliability(rows, outcomes).iloc[0]

    assert result["service_severe_or_cancelled_rate_7d"] == pytest.approx(0.5)
    assert result["service_mean_delay_min_7d"] == pytest.approx(42.5)
    assert result["service_severe_or_cancelled_rate_28d"] == pytest.approx(2 / 3)
    assert result["service_mean_delay_min_28d"] == pytest.approx(145 / 3)
    assert result["corridor_severe_or_cancelled_rate_24h"] == pytest.approx(1.0)
    assert result["corridor_mean_delay_min_24h"] == pytest.approx(65.0)
    assert result["corridor_severe_or_cancelled_rate_3h"] == pytest.approx(1.0)
    assert pd.isna(result["corridor_mean_delay_min_3h"])


def test_unlinked_replacement_is_excluded_from_service_but_kept_for_corridor():
    target = {
        "journey_id": "target", "prediction_time": "1h",
        "predicted_at": pd.Timestamp("2025-01-10 12:00:00"),
        "history_train_type": "ICE", "history_train_number": "100",
    }
    cutoff = cutoff_instants(pd.Series([target["predicted_at"]])).iloc[0]
    outcomes = pd.DataFrame([
        {
            **{key: target[key] for key in ROW_KEY},
            "candidate_journey_id": "unlinked-replacement",
            "history_train_type": "ICE", "history_train_number": "100",
            "origin_departure_utc": cutoff - pd.Timedelta(hours=3),
            "outcome_time_utc": cutoff - pd.Timedelta(hours=1),
            "outcome_known": True, "cancelled": False, "arrival_delay_min": 90.0,
            "is_replacement": True, "replacement_announced": False,
        },
        {
            **{key: target[key] for key in ROW_KEY},
            "candidate_journey_id": "ordinary",
            "history_train_type": "ICE", "history_train_number": "100",
            "origin_departure_utc": cutoff - pd.Timedelta(hours=2),
            "outcome_time_utc": cutoff - pd.Timedelta(minutes=30),
            "outcome_known": True, "cancelled": False, "arrival_delay_min": 10.0,
        },
    ])

    result = build_recent_reliability(pd.DataFrame([target]), outcomes).iloc[0]

    assert result["service_severe_or_cancelled_rate_7d"] == pytest.approx(0.0)
    assert result["service_mean_delay_min_7d"] == pytest.approx(10.0)
    assert result["corridor_severe_or_cancelled_rate_24h"] == pytest.approx(0.5)
    assert result["corridor_mean_delay_min_24h"] == pytest.approx(50.0)


def test_strict_events_exclude_legacy_plan_but_keep_visible_change_reports():
    rows = pd.DataFrame({"journey_id": ["j"], "prediction_time": ["1h"],
                         "predicted_at": [pd.Timestamp("2025-01-01 09:00")]})
    stops = pd.DataFrame({"journey_id": ["j"], "id": ["j-1"],
                          "station_name": ["Hanau"], "train_line_station_num": [1]})
    changes = pd.DataFrame({"id": ["j-1"], "station_name": ["Hanau"],
                            "snapshot_timestamp": [pd.Timestamp("2025-01-01 07:00")],
                            "arrival_change_time": [pd.Timestamp("2025-01-01 08:20")]})
    plans = pd.DataFrame({"id": ["j-1"], "snapshot_timestamp": [pd.Timestamp("2025-01-01 06:00")],
                          "arrival_planned_time": [pd.Timestamp("2025-01-01 08:10")],
                          "departure_planned_time": [pd.Timestamp("2025-01-01 08:30")]})

    default_events = build_upstream_events_asof(rows, stops, changes, plans)
    default_arrival = default_events.loc[default_events.event_kind.eq("arrival")].iloc[0]
    assert default_arrival["event_delay_min"] == pytest.approx(10.0)
    events = build_upstream_events_asof(rows, stops, changes, plans, plan_legacy_policy="exclude")
    arrival = events.loc[events.event_kind.eq("arrival")].iloc[0]

    assert pd.isna(arrival["planned_time"])
    assert pd.isna(arrival["stop_departure_planned_time"])
    assert pd.isna(arrival["event_delay_min"])
    assert arrival["station_name"] == "Hanau"
    assert arrival["event_report_visible"]
    assert arrival["event_occurred_by_cutoff"]


def test_replacement_dynamics_and_sequence_wait_for_timestamped_link():
    rows = pd.DataFrame([
        {"journey_id": "ordinary", "prediction_time": "1h",
         "predicted_at": pd.Timestamp("2025-12-01 10:00:00")},
        {"journey_id": "replacement-before", "prediction_time": "1h",
         "predicted_at": pd.Timestamp("2025-12-01 10:00:00")},
        {"journey_id": "replacement-after", "prediction_time": "1h",
         "predicted_at": pd.Timestamp("2025-12-01 11:00:00")},
    ])
    features = rows.copy()
    for column in ("current_delay", "current_delay_change_min", "upstream_stops_reported",
                   "minutes_since_last_report", "planned_minutes_remaining_before_frankfurt"):
        features[column] = 7.0
    events = pd.concat([
        rows.assign(event_id="event-a"), rows.assign(event_id="event-b"),
    ], ignore_index=True)
    announcements = rows.copy()
    announcements["is_replacement"] = [False, True, True]
    announcements["replacement_announced"] = [False, False, True]
    announcements["replacement_announced_at_utc"] = [
        pd.NaT, pd.NaT, pd.Timestamp("2025-12-01 09:30:00", tz="UTC"),
    ]

    masked_features, masked_events = mask_unlinked_replacement_rows(
        features, events, announcements)

    before = masked_features.loc[masked_features.journey_id.eq("replacement-before")]
    after = masked_features.loc[masked_features.journey_id.eq("replacement-after")]
    assert before["current_delay"].isna().all()
    assert after["current_delay"].eq(7.0).all()
    assert masked_features.loc[masked_features.journey_id.eq("ordinary"),
                               "current_delay"].eq(7.0).all()
    assert not masked_events.journey_id.eq("replacement-before").any()
    assert set(masked_events.journey_id) == {"ordinary", "replacement-after"}


def test_reliability_candidate_pairs_are_limited_to_service_28d_and_corridor_24h():
    cutoff = pd.Timestamp("2025-12-01 12:00:00", tz="UTC")
    targets = pd.DataFrame([{
        "journey_id": "target", "prediction_time": "0min",
        "predicted_at": pd.Timestamp("2025-12-01 13:00:00"),
        "_cutoff_utc": cutoff, "history_train_type": "ICE",
        "history_train_number": "100",
    }])
    candidates = pd.DataFrame([
        {"journey_id": "same-service", "run_start_utc": cutoff - pd.Timedelta(days=10),
         "scheduled_train_type": "ICE", "scheduled_train_number": "100",
         "replacement_train_type": pd.NA, "replacement_train_number": pd.NA},
        {"journey_id": "corridor-recent", "run_start_utc": cutoff - pd.Timedelta(hours=8),
         "scheduled_train_type": "ICE", "scheduled_train_number": "200",
         "replacement_train_type": pd.NA, "replacement_train_number": pd.NA},
        {"journey_id": "old-same-service", "run_start_utc": cutoff - pd.Timedelta(days=40),
         "scheduled_train_type": "ICE", "scheduled_train_number": "100",
         "replacement_train_type": pd.NA, "replacement_train_number": pd.NA},
        {"journey_id": "future", "run_start_utc": cutoff + pd.Timedelta(hours=1),
         "scheduled_train_type": "ICE", "scheduled_train_number": "100",
         "replacement_train_type": pd.NA, "replacement_train_number": pd.NA},
        {"journey_id": "target", "run_start_utc": cutoff - pd.Timedelta(hours=2),
         "scheduled_train_type": "ICE", "scheduled_train_number": "100",
         "replacement_train_type": pd.NA, "replacement_train_number": pd.NA},
    ])

    pairs = _candidate_pairs_for_target_batch(targets, candidates)

    assert set(pairs.candidate_journey_id) == {"same-service", "corridor-recent"}


def test_candidate_outcomes_use_latest_plan_and_change_snapshots_at_target_cutoff():
    pairs = pd.DataFrame([{
        "journey_id": "target", "prediction_time": "0min",
        "predicted_at": pd.Timestamp("2025-12-01 12:00:00"),
        "candidate_journey_id": "candidate",
    }])
    candidates = pd.DataFrame([{
        "journey_id": "candidate", "origin_id": "origin", "destination_id": "destination",
    }])
    plans = pd.DataFrame([
        {"id": "origin", "snapshot_timestamp": pd.Timestamp("2025-12-01 09:00:00"),
         "departure_planned_time": pd.Timestamp("2025-12-01 08:10:00"),
         "arrival_planned_time": pd.NaT, "train_type": "ICE", "train_number": "100"},
        {"id": "destination", "snapshot_timestamp": pd.Timestamp("2025-12-01 09:00:00"),
         "departure_planned_time": pd.NaT,
         "arrival_planned_time": pd.Timestamp("2025-12-01 09:40:00"),
         "train_type": "ICE", "train_number": "100"},
        {"id": "destination", "snapshot_timestamp": pd.Timestamp("2025-12-01 12:00:00"),
         "departure_planned_time": pd.NaT,
         "arrival_planned_time": pd.Timestamp("2025-12-01 10:00:00"),
         "train_type": "ICE", "train_number": "100"},
    ])
    changes = pd.DataFrame([
        {"id": "origin", "snapshot_timestamp": pd.Timestamp("2025-12-01 09:00:00"),
         "departure_change_time": pd.Timestamp("2025-12-01 08:20:00"),
         "departure_cancellation_time": pd.NaT, "departure_change_status": None,
         "arrival_change_time": pd.NaT, "arrival_cancellation_time": pd.NaT,
         "arrival_change_status": None},
        {"id": "destination", "snapshot_timestamp": pd.Timestamp("2025-12-01 09:00:00"),
         "arrival_change_time": pd.Timestamp("2025-12-01 09:50:00"),
         "arrival_cancellation_time": pd.NaT, "arrival_change_status": None,
         "departure_change_time": pd.NaT, "departure_cancellation_time": pd.NaT,
         "departure_change_status": None},
        # Both later rows would change the event to after-cutoff state.
        {"id": "destination", "snapshot_timestamp": pd.Timestamp("2025-12-01 12:00:00"),
         "arrival_change_time": pd.Timestamp("2025-12-01 10:30:00"),
         "arrival_cancellation_time": pd.NaT, "arrival_change_status": None,
         "departure_change_time": pd.NaT, "departure_cancellation_time": pd.NaT,
         "departure_change_status": None},
    ])

    state = _candidate_outcomes_asof(pairs, candidates, changes, plans).iloc[0]

    assert state["origin_departure_utc"] == pd.Timestamp("2025-12-01 07:10:00", tz="UTC")
    assert state["outcome_known"]
    assert not state["cancelled"]
    assert state["arrival_delay_min"] == pytest.approx(10.0)
    assert state["outcome_time_utc"] == pd.Timestamp("2025-12-01 08:50:00", tz="UTC")


def test_history_builder_fails_fast_without_replacement_announcement_cache(tmp_path, monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("snapshot/data read happened before checking the cache")

    monkeypatch.setattr(history_features.pd, "read_parquet", unexpected_read)

    with pytest.raises(FileNotFoundError, match="replacement announcement cache"):
        build_history_dynamics_split(
            "cv", data_dir=tmp_path, output_dir=tmp_path / "advanced",
            plan_dir=tmp_path / "plan", change_dir=tmp_path / "change",
            require_full_archive=False,
        )


def test_holdout_history_builder_gates_before_reading_any_input(tmp_path, monkeypatch):
    import holdout_gate

    def reject_registry(path):
        raise PermissionError("registration not committed")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("holdout data was read before the gate")

    monkeypatch.setattr(holdout_gate, "committed_registration", reject_registry)
    monkeypatch.setattr(history_features.pd, "read_parquet", unexpected_read)

    with pytest.raises(PermissionError, match="registration not committed"):
        build_history_dynamics_split(
            "holdout", data_dir=tmp_path, output_dir=tmp_path / "advanced",
            plan_dir=tmp_path / "plan", change_dir=tmp_path / "change",
            require_full_archive=False,
        )


def test_gated_holdout_dynamics_use_the_shared_upstream_file(tmp_path, monkeypatch):
    import holdout_gate

    output = tmp_path / "advanced"
    output.mkdir()
    (output / "replacement_announcements_holdout.parquet").touch()
    reads = []

    def read(path, *, columns=None):
        reads.append((path.name, columns))
        if path.name == "replacement_announcements_holdout.parquet":
            return pd.DataFrame()
        if path.name == "prediction_rows_holdout.parquet":
            return pd.DataFrame({"journey_id": ["holdout-example"]})
        raise InterruptedError("stop before reading any archives")

    monkeypatch.setattr(holdout_gate, "committed_registration", lambda path: {})
    monkeypatch.setattr(history_features.pd, "read_parquet", read)
    with pytest.raises(InterruptedError, match="stop before reading any archives"):
        build_history_dynamics_split("holdout", data_dir=tmp_path, require_full_archive=False)

    assert reads[1] == ("prediction_rows_holdout.parquet", history_features.ROW_KEY)
    assert reads[2] == ("upstream_runs.parquet",
                        ["journey_id", "id", "train_line_station_num", "station_name"])


def test_holdout_reliability_builder_gates_before_reading_any_input(tmp_path, monkeypatch):
    import holdout_gate

    def reject_registry(path):
        raise PermissionError("registration not committed")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("holdout data was read before the gate")

    monkeypatch.setattr(holdout_gate, "committed_registration", reject_registry)
    monkeypatch.setattr(history_features.pd, "read_parquet", unexpected_read)

    with pytest.raises(PermissionError, match="registration not committed"):
        build_reliability_file(
            split="holdout", data_dir=tmp_path, output=tmp_path / "reliability.parquet",
            plan_dir=tmp_path / "plan", change_dir=tmp_path / "change",
            registry_path=tmp_path / "finalists.json",
        )


def _stub_reliability_builder_inputs(
    tmp_path, monkeypatch, *, target_train_number="123", scheduled_train_number="123",
    replacement_train_number=None, is_replacement_train=False,
):
    import recent_reliability
    import scheduled_timetable

    parquet_reader = pd.read_parquet
    data_dir = tmp_path / "data"
    advanced_dir = data_dir / "advanced"
    advanced_dir.mkdir(parents=True)
    (advanced_dir / "replacement_context_cv.parquet").touch()
    (advanced_dir / "replacement_announcements_cv.parquet").touch()

    target_rows = pd.DataFrame([{
        "journey_id": "target-2502010000",
        "prediction_time": "24h",
        "predicted_at": pd.Timestamp("2025-01-31 23:30:00"),
    }])
    candidate_id = "candidate-2501312200"
    candidate_context = pd.DataFrame([{
        "journey_id": candidate_id,
        "current_origin_id": f"{candidate_id}-0",
        "scheduled_train_type": "ICE", "scheduled_train_number": scheduled_train_number,
        "replacement_train_type": "ICE" if replacement_train_number is not None else pd.NA,
        "replacement_train_number": replacement_train_number if replacement_train_number is not None else pd.NA,
        "is_replacement_train": is_replacement_train,
    }])
    candidate_stops = pd.DataFrame([
        {"id": f"{candidate_id}-0", "journey_id": candidate_id,
         "station_name": "Frankfurt (Main) Hbf", "train_line_station_num": 0},
        {"id": f"{candidate_id}-1", "journey_id": candidate_id,
         "station_name": "München Hbf", "train_line_station_num": 1},
    ])

    def read_parquet(path, columns=None):
        name = Path(path).name
        if name == "prediction_rows_cv.parquet":
            return target_rows.copy()
        if name == "replacement_context_cv.parquet":
            return candidate_context.copy()
        if name == "replacement_announcements_cv.parquet":
            return pd.DataFrame()
        if name == "stops.parquet":
            return candidate_stops.copy()
        raise AssertionError(f"unexpected Parquet read before archive interception: {name}")

    monkeypatch.setattr(recent_reliability.pd, "read_parquet", read_parquet)
    monkeypatch.setattr(
        scheduled_timetable, "service_keys_asof",
        lambda rows, journeys, context, announcements, **kwargs: rows.assign(
            history_train_type="ICE",
            history_train_number=(replacement_train_number or scheduled_train_number)
            if "candidate_journey_id" in rows else target_train_number,
        ),
    )
    return recent_reliability, scheduled_timetable, data_dir, parquet_reader


def test_reliability_builder_reads_next_month_for_month_end_targets(tmp_path, monkeypatch):
    recent_reliability, _, data_dir, _ = _stub_reliability_builder_inputs(
        tmp_path, monkeypatch,
    )
    observed = {}

    def intercept_archive_read(ids, *, kind, months, change_dir, plan_dir):
        observed["kind"] = kind
        observed["months"] = {str(month) for month in months}
        raise InterruptedError("archive read intercepted after source month selection")

    monkeypatch.setattr(recent_reliability, "_read_snapshot_ids_in_months", intercept_archive_read)

    with pytest.raises(InterruptedError, match="source month selection"):
        build_reliability_file(
            data_dir=data_dir, plan_dir=tmp_path / "plan", change_dir=tmp_path / "change",
            output=tmp_path / "reliability.parquet", require_full_archive=False,
        )

    assert observed["kind"] == "plan"
    assert "2025-02" in observed["months"]


def test_reliability_builder_passes_cutoff_announcement_into_service_gate(tmp_path, monkeypatch):
    recent_reliability, scheduled_timetable, data_dir, parquet_reader = _stub_reliability_builder_inputs(
        tmp_path, monkeypatch, target_train_number="900", scheduled_train_number="100",
        replacement_train_number="900", is_replacement_train=True,
    )

    monkeypatch.setattr(
        recent_reliability, "_read_snapshot_ids_in_months",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    def announcements_asof(pairs, context, changes, *, row_keys, **kwargs):
        result = pairs[row_keys].copy()
        result["is_replacement"] = True
        result["replacement_announced"] = True
        return result

    monkeypatch.setattr(scheduled_timetable, "replacement_announcements_asof", announcements_asof)

    def candidate_outcomes(pairs, candidates, changes, plans):
        result = pairs[ROW_KEY + ["candidate_journey_id"]].copy()
        cutoff = cutoff_instants(result["predicted_at"])
        result["origin_departure_utc"] = cutoff - pd.Timedelta(hours=3)
        result["outcome_time_utc"] = cutoff - pd.Timedelta(hours=1)
        result["outcome_known"] = True
        result["cancelled"] = False
        result["arrival_delay_min"] = 90.0
        result["origin_train_type_asof"] = pd.NA
        result["origin_train_number_asof"] = pd.NA
        return result

    monkeypatch.setattr(recent_reliability, "_candidate_outcomes_asof", candidate_outcomes)

    output = build_reliability_file(
        data_dir=data_dir, plan_dir=tmp_path / "plan", change_dir=tmp_path / "change",
        output=tmp_path / "reliability.parquet", require_full_archive=False,
    )
    result = parquet_reader(output).iloc[0]

    assert result["service_severe_or_cancelled_rate_7d"] == pytest.approx(1.0)
    assert result["service_mean_delay_min_7d"] == pytest.approx(90.0)
    assert result["corridor_mean_delay_min_24h"] == pytest.approx(90.0)


def test_legacy_plan_is_static_before_its_label_and_exact_updates_stay_asof():
    from history_features import latest_plan_asof

    queries = pd.DataFrame({
        "_query_id": [0, 1], "id": ["j-1", "j-1"],
        "cutoff_utc": pd.to_datetime(["2025-01-01", "2025-12-01"], utc=True),
    })
    plans = pd.DataFrame({
        "id": ["j-1"] * 3,
        "snapshot_timestamp": pd.to_datetime(["2025-01-03", "2025-12-01", "2025-01-02"], utc=True),
        "marker": [20, 30, 10],
    })
    selected = latest_plan_asof(queries, plans)
    assert selected.marker.tolist() == [10, 30]
    assert selected.loc[0, "snapshot_timestamp"] > selected.loc[0, "cutoff_utc"]
    assert selected.loc[0, "_observed_utc"] == pd.Timestamp("1900-01-01", tz="UTC")
