"""Build the Frankfurt (Main) Hbf -> München Hbf modelling dataset.

Journeys, targets and severity classes follow corridor_analysis/extract.py,
whose functions are reused here for the single pair. On top of that, this
script

1. keeps one journey per departure (SHARED DEPARTURES below);
2. adds `current_delay` at each prediction time (CURRENT DELAY below);
3. adds when a cancellation was announced (KNOWN CANCELLATIONS below);
4. splits the journeys by planned Frankfurt departure (see folds.py) into
   data/journeys_cv.parquet (July 2024 to September 2025) and
   data/journeys_holdout.parquet (October 2025 to September 2026), and writes
   the cross-validation fold boundaries to cv_folds.json;
5. writes the prediction rows, one per journey and prediction time, to
   data/prediction_rows_cv.parquet and data/prediction_rows_holdout.parquet.

It also writes data/stops.parquet (the stops from Frankfurt to München),
data/upstream_runs.parquet (each run's stops up to Frankfurt) and
build_audit.json (the counts quoted in README.md).

Prediction times are 24 h, 6 h, 1 h, 20 min and 0 min before the planned
Frankfurt departure. Only stations in data/stations_present_in_all_months.csv
are used, as in the journey extraction, so the 3 November 2025 coverage
expansion adds no stations.

SHARED DEPARTURES. Some departures appear as two or more journeys under
different train numbers with the same planned Frankfurt departure and München
arrival, often a cancelled train and the replacement train that ran instead.
One journey is kept per departure, from the passenger's point of view: a
journey that completed if there is one, then one not flagged as a replacement
train, then the lowest train number. `shared_departure` is the number of
journeys that were listed for the departure.

CURRENT DELAY. current_delay at prediction time t is the latest delay known at
t, from the train's own run up to and including its arrival at Frankfurt:

- the latest non-cancelled arrival or departure whose reported time is at or
  before t;
- if that event is an arrival (the train is standing at a stop and has not left
  by t), the delay accumulated by t: the larger of the arrival delay and the
  time since the planned departure from that stop;
- missing if no event of the run has happened by t, for example 24 h ahead or
  for trains that begin their run at Frankfurt.

Negative values (early trains) are kept. Missing values are kept; how to fill
them is decided per model.

KNOWN CANCELLATIONS. No prediction is made for a journey whose cancellation is
already announced at the prediction time. `cancellation_announced` is the time
DB issued the cancellation of the cancelled Frankfurt departure or München
arrival, the earlier of the two, from the live-update snapshots in
data/monthly_processed_data_change (the monthly files have no issue time). A
journey has no prediction row at prediction times at or after that moment. A
cancellation that was later withdrawn, so that the train ran, does not count:
those journeys keep every prediction row and are marked
`cancellation_withdrawn`.

Run from the project root: .venv/bin/python frankfurt_munich/build_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "corridor_analysis"))
import extract  # noqa: E402  corridor_analysis/extract.py

from folds import CV_START, HOLDOUT_END, HOLDOUT_START, cv_folds, fold_bounds  # noqa: E402
from snapshots import event_summary, read_snapshots  # noqa: E402

INPUT_DIR = ROOT / "data" / "monthly_processed_data"
PANEL_FILE = ROOT / "data" / "stations_present_in_all_months.csv"
OUTPUT_DIR = HERE / "data"
ORIGIN, DESTINATION = "Frankfurt (Main) Hbf", "München Hbf"
HORIZONS = {"24h": pd.Timedelta(hours=24), "6h": pd.Timedelta(hours=6),
            "1h": pd.Timedelta(hours=1), "20min": pd.Timedelta(minutes=20),
            "0min": pd.Timedelta(0)}
RUN_COLUMNS = [
    "id", "station_name", "train_type", "train_line_ride_id", "train_line_station_num",
    "arrival_planned_time", "arrival_change_time", "departure_planned_time",
    "departure_change_time", "arrival_is_canceled", "departure_is_canceled",
    "is_replacement_train",
]
PREDICTION_ROW_COLUMNS = [
    "journey_id", "origin_departure_planned_local", "train_number", "cancellation_withdrawn",
    "arrival_delay_min", "journey_canceled", "severity_class", "severe_or_canceled",
]


def extract_journeys(paths: list[Path], panel: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The corridor_analysis extraction, for Frankfurt -> München only."""
    endpoints = extract.read_endpoints(paths, [ORIGIN, DESTINATION])
    seg, audit = extract.choose_segments(endpoints, ORIGIN, DESTINATION)
    stops = extract.read_segments(paths, {"pair": seg})["pair"]
    seg = seg.set_index("journey_id")
    gaps = (seg.train_line_station_num_destination - seg.train_line_station_num_origin + 1
            - stops.groupby("journey_id").size().reindex(seg.index).fillna(0))
    outside = ~stops.station_name.isin(panel)
    audit.update({"journeys": int(len(seg)), "stops_removed_outside_panel": int(outside.sum())})
    stops = stops[~outside].drop(columns=["origin_sequence", "destination_sequence"])
    journeys = extract.journey_table(stops, gaps, ORIGIN, DESTINATION)
    return journeys, stops, audit


def drop_shared_departures(journeys: pd.DataFrame, stops: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one journey per planned departure and arrival (see SHARED DEPARTURES)."""
    key = ["origin_departure_planned_local", "destination_arrival_planned_local"]
    j = journeys.assign(
        is_replacement_train=journeys.journey_id.map(stops.groupby("journey_id").is_replacement_train.any()),
        number=pd.to_numeric(journeys.train_number, errors="coerce"))
    j["shared_departure"] = j.groupby(key).journey_id.transform("size")
    canceled = j.groupby(key).journey_canceled.transform("sum")
    shared = j.shared_departure.gt(1)
    mixed = canceled.gt(0) & canceled.lt(j.shared_departure)
    keep = j.sort_values(key + ["journey_canceled", "is_replacement_train", "number"]).groupby(key).head(1).index
    kept = j.loc[keep]
    audit = {"journeys_in_shared_departures": int(shared.sum()),
             "groups": int(j[shared].groupby(key).ngroups),
             "groups_all_completed": int(j[shared & canceled.eq(0)].groupby(key).ngroups),
             "groups_all_canceled": int(j[shared & canceled.eq(j.shared_departure)].groupby(key).ngroups),
             "groups_canceled_and_completed": int(j[mixed].groupby(key).ngroups),
             "kept_replacement_trains_from_canceled_and_completed":
                 int((kept.is_replacement_train & mixed.loc[keep]).sum()),
             "journeys_removed": int(len(j) - len(keep))}
    out = kept.drop(columns="number").sort_values("origin_departure_planned_local")
    return out.reset_index(drop=True), audit


def read_runs(origin_num: pd.Series) -> pd.DataFrame:
    """Every recorded stop of each journey's run up to and including Frankfurt."""
    rides = sorted(origin_num.index.str.rsplit("-", n=1).str[0].unique())
    parts = []
    for path in sorted(INPUT_DIR.glob("data-????-??.parquet")):
        frame = pq.read_table(path, columns=RUN_COLUMNS, filters=[
            ("train_type", "in", ["ICE", "IC"]), ("train_line_ride_id", "in", rides)]).to_pandas()
        frame["journey_id"] = frame["id"].str.rsplit("-", n=1).str[0]
        frame = frame[frame.journey_id.isin(origin_num.index)]
        parts.append(frame[frame.train_line_station_num.le(frame.journey_id.map(origin_num))])
    runs = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["id"], keep="first")
    return runs.sort_values(["journey_id", "train_line_station_num"]).reset_index(drop=True)


def events_from(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per non-cancelled arrival or departure of the runs."""
    keep = ["journey_id", "train_line_station_num", "station_name"]
    arr = runs[keep].assign(
        kind=0, planned=runs.arrival_planned_time, reported=runs.arrival_change_time,
        canceled=runs.arrival_is_canceled.fillna(False).astype(bool),
        next_dep_planned=runs.departure_planned_time)
    dep = runs[keep].assign(
        kind=1, planned=runs.departure_planned_time, reported=runs.departure_change_time,
        canceled=runs.departure_is_canceled.fillna(False).astype(bool), next_dep_planned=pd.NaT)
    ev = pd.concat([arr, dep], ignore_index=True)
    ev = ev[ev.planned.notna() & ev.reported.notna() & ~ev.canceled]
    ev["delay"] = (ev.reported - ev.planned).dt.total_seconds() / 60
    return ev.sort_values(["journey_id", "reported", "train_line_station_num", "kind"])


def current_delay_at(events: pd.DataFrame, when: pd.Series) -> pd.Series:
    """Latest delay known at `when` (indexed by journey_id); see CURRENT DELAY."""
    ev = events.assign(t=events.journey_id.map(when))
    ev = ev[ev.reported.le(ev.t)]
    last = ev.groupby("journey_id", sort=False).tail(1).set_index("journey_id")
    waiting = (last.t - last.next_dep_planned).dt.total_seconds() / 60
    standing = last.kind.eq(0)
    value = last.delay.where(~standing, np.fmax(last.delay, waiting.fillna(-np.inf)))
    return value.reindex(when.index)


def cancellation_times(journeys: pd.DataFrame, stops: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`cancellation_announced` and `cancellation_withdrawn` (see KNOWN CANCELLATIONS),
    indexed by journey_id."""
    canceled = journeys.journey_canceled.astype(bool)
    ends = {"departure": (stops[stops.stop_index.eq(0)], journeys.origin_departure_canceled),
            "arrival": (stops[stops.final_destination], journeys.destination_arrival_canceled)}
    snaps = read_snapshots(set(pd.concat([s.id for s, _ in ends.values()])))
    announced, ever = [], []
    for side, (end_stops, end_canceled) in ends.items():
        summary = event_summary(snaps, end_stops.set_index("journey_id").id.reindex(journeys.index), side)
        # The issue time of the cancellation in force. If the latest snapshot only
        # carries the cancelled status, the first time a cancellation was seen.
        issued = summary.latest_cancellation_time.fillna(summary.first_cancellation_time)
        announced.append(issued.where(end_canceled.astype(bool)))
        ever.append(summary.first_cancellation_time.notna())
    out = pd.DataFrame({"cancellation_announced": pd.concat(announced, axis=1).min(axis=1),
                        "cancellation_withdrawn": (ever[0] | ever[1]) & ~canceled})
    audit = {"last_snapshot_utc": str(snaps.snapshot_timestamp.max()),
             "canceled": int(canceled.sum()),
             "canceled_without_announcement_time": int((canceled & out.cancellation_announced.isna()).sum()),
             "withdrawn": int(out.cancellation_withdrawn.sum())}
    return out, audit


def prediction_rows(journeys: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per journey and prediction time, without known cancellations."""
    rows, left_out = [], {}
    for name, ahead in HORIZONS.items():
        at = journeys.origin_departure_planned_local - ahead
        known = journeys.cancellation_announced.le(at)
        left_out[name] = int(known.sum())
        rows.append(journeys.loc[~known, PREDICTION_ROW_COLUMNS].assign(
            prediction_time=name, predicted_at=at[~known], current_delay=journeys[f"current_delay_{name}"][~known]))
    out = pd.concat(rows, ignore_index=True)
    out["prediction_time"] = pd.Categorical(out.prediction_time, categories=list(HORIZONS), ordered=True)
    return out.sort_values(["origin_departure_planned_local", "journey_id", "prediction_time"]), left_out


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_parquet(OUTPUT_DIR / name, index=False, compression="zstd")


def main() -> None:
    paths = sorted(INPUT_DIR.glob("data-????-??.parquet"))
    panel = set(pd.read_csv(PANEL_FILE).station_name)

    # Journeys, one per departure
    journeys, stops, extraction_audit = extract_journeys(paths, panel)
    journeys, shared_audit = drop_shared_departures(journeys, stops)
    stops = stops[stops.journey_id.isin(journeys.journey_id)]
    j = journeys.set_index("journey_id")

    # current_delay, from each run's stops up to Frankfurt
    origin_num = stops[stops.stop_index.eq(0)].set_index("journey_id").train_line_station_num
    assert origin_num.index.is_unique and set(origin_num.index) == set(j.index)
    runs = read_runs(origin_num)
    off_panel = ~runs.station_name.isin(panel)
    runs_audit = {"run_stops_read": int(len(runs)), "run_stops_off_panel": int(off_panel.sum())}
    runs = runs[~off_panel]
    events = events_from(runs)
    j["starts_at_origin"] = origin_num.reindex(j.index).eq(1)
    for name, ahead in HORIZONS.items():
        j[f"current_delay_{name}"] = current_delay_at(events, j.origin_departure_planned_local - ahead)

    # When cancellations were announced
    cancellations, cancel_audit = cancellation_times(j, stops)
    j = j.join(cancellations).reset_index()

    # Split, prediction rows and folds
    when = j.origin_departure_planned_local
    splits = {"cv": j[when.ge(CV_START) & when.lt(HOLDOUT_START)],
              "holdout": j[when.ge(HOLDOUT_START) & when.lt(HOLDOUT_END)]}
    assert sum(len(s) for s in splits.values()) == len(j)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    left_out = {}
    for split, frame in splits.items():
        write(frame, f"journeys_{split}.parquet")
        rows, left_out[split] = prediction_rows(frame)
        write(rows, f"prediction_rows_{split}.parquet")
    write(stops, "stops.parquet")
    write(runs, "upstream_runs.parquet")

    cv = splits["cv"]
    folds = [{k: v if k == "fold" else str(v) for k, v in bounds.items()}
             | {"train_journeys": int(train.sum()), "validation_journeys": int(val.sum())}
             for (_, train, val), bounds in zip(cv_folds(cv), fold_bounds())]
    (HERE / "cv_folds.json").write_text(json.dumps(folds, indent=2))

    audit = {"source_files": [p.name for p in paths], "extraction": extraction_audit,
             "shared_departures": shared_audit, "journeys": int(len(j)),
             **{f"{split}_journeys": int(len(frame)) for split, frame in splits.items()},
             **{f"{split}_{name}": str(getattr(frame.origin_departure_planned_local, end)())
                for split, frame in splits.items() for name, end in (("first", "min"), ("last", "max"))},
             **runs_audit, "cancellations": cancel_audit, "left_out_known_cancellation": left_out,
             "current_delay_known": {c: int(j[c].notna().sum()) for c in j if c.startswith("current_delay_")}}
    (HERE / "build_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(audit | {"folds": folds}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
