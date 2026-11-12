"""The dataset-build rules corrected on 12 November 2026 (finishing plan, B7):
no prediction row while a cancellation is in force (R6), no Frankfurt departure
in current_delay (R7), cancellation_announced is the first issue time (R8)."""
import numpy as np
import pandas as pd

import build_dataset as bd
import snapshots as sn


def ts(text):
    return pd.Timestamp(text)


def snapshots(rows):
    """Snapshot rows (id, UTC stamp, side, issue time, status) in the form
    read_snapshots returns."""
    frame = pd.DataFrame([{column: None for column in sn.COLUMNS} for _ in rows])
    for i, (stop_id, stamp, side, issue, status) in enumerate(rows):
        frame.loc[i, "id"] = stop_id
        frame.loc[i, "snapshot_timestamp"] = ts(stamp)
        frame.loc[i, f"{side}_cancellation_time"] = None if issue is None else ts(issue)
        frame.loc[i, f"{side}_change_status"] = status
    frame["snapshot_timestamp"] = pd.to_datetime(frame.snapshot_timestamp)
    for column in sn.COLUMNS:
        if column.endswith("_time"):
            frame[column] = pd.to_datetime(frame[column])
    return sn.with_request_times(frame)


def in_force(events, stop_id, times):
    ids = pd.Series(stop_id, index=range(len(times)))
    return sn.cancellation_in_force(events, ids, pd.Series([ts(t) for t in times])).tolist()


def test_cancellation_is_in_force_from_its_issue_until_the_withdrawal_is_shown():
    # January: local time is UTC + 1 hour. Exact request stamps.
    snaps = snapshots([
        ("a", "2026-01-10 08:30:10", "departure", "2026-01-10 08:00", "c"),
        ("a", "2026-01-10 11:00:10", "departure", "2026-01-10 08:00", "p"),
        ("a", "2026-01-10 14:00:10", "departure", "2026-01-10 08:00", "p")])
    events = sn.cancellation_events(snaps, ["a"], "departure")
    assert events.in_force.tolist() == [True, False]
    assert events.time.tolist() == [ts("2026-01-10 08:00"), ts("2026-01-10 12:00:10")]
    assert in_force(events, "a", ["2026-01-10 07:59", "2026-01-10 08:00", "2026-01-10 12:00",
                                  "2026-01-10 12:01", "2026-01-11 09:00"]) == [False, True, True, False, False]
    # A stop without any cancellation, and the other side of the same stop.
    assert in_force(events, "b", ["2026-01-10 09:00"]) == [False]
    assert sn.cancellation_events(snaps, ["a"], "arrival").empty


def test_withdrawal_first_shown_in_a_block_label_snapshot_counts_45_minutes_later():
    # July 2025: local time is UTC + 2 hours; the stamp is the start of the fetch block.
    snaps = snapshots([("a", "2025-07-01 06:00:00", "arrival", "2025-07-01 05:10", "p")])
    events = sn.cancellation_events(snaps, ["a"], "arrival")
    assert events.time.tolist() == [ts("2025-07-01 05:10"), ts("2025-07-01 08:45")]
    assert in_force(events, "a", ["2025-07-01 08:44", "2025-07-01 08:45"]) == [True, False]


def test_cancellation_shown_again_after_a_withdrawal_holds_from_that_snapshot():
    snaps = snapshots([
        ("a", "2026-03-02 21:00:10", "departure", "2026-03-02 16:43", "p"),
        ("a", "2026-03-03 03:50:10", "departure", "2026-03-02 16:43", "c"),
        ("a", "2026-03-03 15:20:10", "departure", "2026-03-03 03:19", "c")])
    events = sn.cancellation_events(snaps, ["a"], "departure")
    assert in_force(events, "a", ["2026-03-02 17:00", "2026-03-02 23:00", "2026-03-03 03:00",
                                  "2026-03-03 03:19", "2026-03-03 05:00"]) == [True, False, False, True, True]


def journeys_and_stops():
    journeys = pd.DataFrame({
        "journey_id": ["ran", "withdrawn", "cancelled"],
        "origin_departure_planned_local": [ts("2026-01-10 12:00")] * 3,
        "journey_canceled": [False, False, True],
        "origin_departure_canceled": [False, False, True],
        "destination_arrival_canceled": [False, False, True],
    }).set_index("journey_id")
    stops = pd.DataFrame({
        "journey_id": ["ran", "ran", "withdrawn", "withdrawn", "cancelled", "cancelled"],
        "id": ["ran-0", "ran-1", "withdrawn-0", "withdrawn-1", "cancelled-0", "cancelled-1"],
        "stop_index": [0, 1] * 3, "final_destination": [False, True] * 3})
    return journeys, stops


def test_no_prediction_row_while_a_cancellation_is_in_force(monkeypatch):
    journeys, stops = journeys_and_stops()
    snaps = snapshots([
        # Cancelled the evening before, withdrawal first shown at 10:30:10 local time.
        ("withdrawn-0", "2026-01-09 20:00:10", "departure", "2026-01-09 19:00", "c"),
        ("withdrawn-0", "2026-01-10 09:30:10", "departure", "2026-01-09 19:00", "p"),
        # Cancelled at 07:00 on the day; the latest snapshot carries a later issue time.
        ("cancelled-0", "2026-01-10 07:30:10", "departure", "2026-01-10 07:00", "c"),
        ("cancelled-0", "2026-01-10 10:30:10", "departure", "2026-01-10 09:40", "c"),
        ("cancelled-1", "2026-01-10 10:30:10", "arrival", "2026-01-10 09:40", "c")])
    monkeypatch.setattr(bd, "read_snapshots", lambda ids: snaps[snaps.id.isin(ids)])
    out, in_force_at, audit = bd.cancellation_times(journeys, stops)

    # R8: the first issue time, not the one in the latest snapshot.
    assert out.cancellation_announced["cancelled"] == ts("2026-01-10 07:00")
    assert out.cancellation_announced[["ran", "withdrawn"]].isna().all()
    assert out.cancellation_withdrawn.to_dict() == {"ran": False, "withdrawn": True, "cancelled": False}
    assert audit["canceled_with_a_later_issue_time_in_the_latest_snapshot"] == 1

    # R6: prediction times are 12:00 the day before, 06:00, 11:00, 11:40 and 12:00.
    assert in_force_at.loc["ran"].tolist() == [False] * 5
    assert in_force_at.loc["withdrawn"].tolist() == [False, True, False, False, False]
    assert in_force_at.loc["cancelled"].tolist() == [False, False, True, True, True]

    frame = journeys.join(out).reset_index().assign(
        train_number="1", arrival_delay_min=[3.0, 8.0, np.nan], severity_class=["under_20", "under_20", "canceled"],
        severe_or_canceled=[False, False, True],
        **{f"current_delay_{name}": np.nan for name in bd.HORIZONS})
    rows, left_out = bd.prediction_rows(frame, in_force_at)
    assert left_out == {"24h": 0, "6h": 1, "1h": 1, "20min": 1, "0min": 1}
    kept = rows.groupby("journey_id").prediction_time.apply(lambda s: sorted(s.astype(str))).to_dict()
    assert kept["ran"] == sorted(bd.HORIZONS)
    assert kept["withdrawn"] == ["0min", "1h", "20min", "24h"]
    assert kept["cancelled"] == ["24h", "6h"]


def test_current_delay_does_not_use_the_departure_from_frankfurt():
    origin_num = pd.Series({"starts-here": 1, "from-upstream": 2})
    runs = pd.DataFrame({
        "journey_id": ["starts-here", "from-upstream", "from-upstream"],
        "train_line_station_num": [1, 1, 2],
        "station_name": ["Frankfurt (Main) Hbf", "Mainz Hbf", "Frankfurt (Main) Hbf"],
        "arrival_planned_time": [pd.NaT, pd.NaT, ts("2026-01-10 11:50")],
        "arrival_change_time": [pd.NaT, pd.NaT, ts("2026-01-10 11:54")],
        "arrival_is_canceled": [False, False, False],
        # Both trains left Frankfurt on time, at 12:00.
        "departure_planned_time": [ts("2026-01-10 12:00"), ts("2026-01-10 11:20"), ts("2026-01-10 12:00")],
        "departure_change_time": [ts("2026-01-10 12:00"), ts("2026-01-10 11:27"), ts("2026-01-10 12:00")],
        "departure_is_canceled": [False, False, False]})
    events = bd.events_from(runs, origin_num)
    assert not (events.station_name.eq("Frankfurt (Main) Hbf") & events.kind.eq(1)).any()

    at_departure = pd.Series(ts("2026-01-10 12:00"), index=origin_num.index)
    delay = bd.current_delay_at(events, at_departure)
    assert np.isnan(delay["starts-here"])       # nothing is known before the train leaves
    assert delay["from-upstream"] == 4.0        # its arrival at Frankfurt
    earlier = bd.current_delay_at(events, at_departure - pd.Timedelta(minutes=20))
    assert earlier["from-upstream"] == 7.0      # its departure from Mainz
