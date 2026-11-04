"""Build cutoff-specific CV training rows for the auxiliary corridors in step 8.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from advanced_data import ADVANCED
from build_dataset import HORIZONS, INPUT_DIR, PANEL_FILE, extract
from corridors import CORRIDORS
from data import ROW_KEY, TIME_COLUMN, merge_inputs
from features import CURRENT_DELAY_TIMES
from folds import read_assignment, week_of
from history_features import (CHANGE_DIR, PLAN_DIR, PLAN_EXACT_UTC_FROM,
                              _cancellation_in_force, _journey_run_start_utc, _read_snapshot_ids_in_months,
                              build_upstream_events_asof, cutoff_instants,
                              delay_dynamics_from_events, latest_snapshot_asof)

SOURCE_CORRIDORS = ("berlin", "hamburg", "hamburg_frankfurt")
DESTINATION = "destination_arrival_planned_local"
STOP_COLUMNS = ["id", "station_name", "train_line_station_num"]
OUTCOME_COLUMNS = ["id", "departure_is_canceled", "arrival_is_canceled",
                   "arrival_change_time", "is_replacement_train", "is_additional_stop"]


def source_segments(paths, weeks):
    """Select CV runs using planned times and run IDs before reading outcomes."""
    ends = {CORRIDORS[c][end] for c in SOURCE_CORRIDORS for end in ("origin", "destination")}
    endpoints = extract.read_endpoints(paths, sorted(ends))
    segments = []
    for name in SOURCE_CORRIDORS:
        corridor = CORRIDORS[name]
        pairs, _ = extract.choose_segments(endpoints, corridor["origin"], corridor["destination"])
        planned = pairs.rename(columns={"departure_planned_time_origin": TIME_COLUMN,
                                        "arrival_planned_time_destination": DESTINATION,
                                        "train_line_station_num_origin": "origin_sequence"})
        planned = planned[["journey_id", "id_origin", "id_destination", "origin_sequence",
                           TIME_COLUMN, DESTINATION]].assign(corridor=name)
        start = _journey_run_start_utc(planned.journey_id).dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
        planned = planned.assign(run_week=week_of(start), destination_week=week_of(planned[DESTINATION]))
        duration = planned[DESTINATION] - planned[TIME_COLUMN]
        keep = (week_of(planned[TIME_COLUMN]).isin(weeks) & planned.run_week.isin(weeks)
                & planned.destination_week.isin(weeks) & duration.gt(pd.Timedelta(0)))
        segments.append(planned.loc[keep])
    return pd.concat(segments, ignore_index=True)


def source_journeys(paths, segments, weeks):
    ids = sorted(set(segments.id_origin) | set(segments.id_destination))
    outcomes = pd.concat([
        pq.read_table(path, columns=OUTCOME_COLUMNS, filters=[("id", "in", ids)],
                      use_threads=False).to_pandas() for path in paths
    ], ignore_index=True).drop_duplicates("id").set_index("id")
    origin, destination = (outcomes.loc[segments[col]].reset_index(drop=True)
                           for col in ("id_origin", "id_destination"))
    cancelled = origin.departure_is_canceled.fillna(False) | destination.arrival_is_canceled.fillna(False)
    delay = (destination.arrival_change_time - segments[DESTINATION]).dt.total_seconds() / 60
    observed_at = destination.arrival_change_time.where(~cancelled, segments[DESTINATION])
    journeys = segments.assign(journey_canceled=cancelled, arrival_delay_min=delay.mask(cancelled),
                               outcome_week=week_of(observed_at), severity_class=extract.severity(delay, cancelled))
    ordinary = ~(origin.is_replacement_train.fillna(False) | destination.is_replacement_train.fillna(False)
                 | origin.is_additional_stop.fillna(False) | destination.is_additional_stop.fillna(False))
    same_departure = journeys.duplicated(["corridor", TIME_COLUMN, DESTINATION], keep=False)
    return journeys.loc[ordinary & ~same_departure & journeys.outcome_week.isin(weeks)
                        & journeys.severity_class.notna()].reset_index(drop=True)


def read_upstream(paths, journeys):
    """Collect stop IDs and sequence numbers, leaving reported times in the archive."""
    origin = journeys.groupby("journey_id").origin_sequence.max()
    rides = sorted(origin.index.str.rsplit("-", n=1).str[0].unique())
    panel = pd.read_csv(PANEL_FILE, usecols=["station_name"]).station_name.tolist()
    parts = []
    for path in paths:
        stops = pq.read_table(path, columns=STOP_COLUMNS, use_threads=False,
                              filters=[("train_line_ride_id", "in", rides),
                                       ("station_name", "in", panel)]).to_pandas()
        stops = extract.add_journey_id(stops)
        keep = stops.train_line_station_num.le(stops.journey_id.map(origin))
        parts.append(stops.loc[keep])
    return pd.concat(parts, ignore_index=True).drop_duplicates("id")


def prediction_rows(journeys, upstream, changes, plans):
    rows = pd.concat([journeys.assign(prediction_time=name, predicted_at=journeys[TIME_COLUMN] - ahead)
                      for name, ahead in HORIZONS.items()], ignore_index=True)
    departure = rows[TIME_COLUMN].dt
    rows = rows.assign(hour=departure.hour, weekday=departure.dayofweek, month=departure.month,
                       planned_travel_min=(rows[DESTINATION] - rows[TIME_COLUMN]).dt.total_seconds() / 60)
    queries = rows[["id_origin", "id_destination"]].assign(
        _query_id=np.arange(len(rows)), cutoff_utc=cutoff_instants(rows.predicted_at))
    planned = latest_snapshot_asof(queries.rename(columns={"id_origin": "id"}), plans,
                                   request_bound=False)
    known = planned.get("_observed_utc", pd.Series(pd.NaT, index=rows.index)).notna()
    rows["train_type"] = planned.get("train_type", pd.Series(pd.NA, index=rows.index))
    rows["starts_at_origin"] = queries.id_origin.str.rsplit("-", n=1).str[-1].eq("1").where(known)
    for column in ("train_type", "starts_at_origin"):
        rows[column] = rows[column].astype("string").fillna("unknown")

    cancelled = pd.Series(False, index=rows.index)
    if not changes.empty:
        for endpoint, kind in (("origin", "departure"), ("destination", "arrival")):
            state = latest_snapshot_asof(queries.rename(columns={f"id_{endpoint}": "id"}), changes)
            cancelled |= _cancellation_in_force(state[f"{kind}_cancellation_time"],
                                                state[f"{kind}_change_status"], state.cutoff_utc)
    rows = rows.loc[~cancelled].reset_index(drop=True)
    late = rows.loc[rows.prediction_time.isin(CURRENT_DELAY_TIMES)]
    events = build_upstream_events_asof(late, upstream, changes, plans)
    delay = delay_dynamics_from_events(late, events)[[*ROW_KEY, "current_delay"]]
    late = merge_inputs(late, delay)
    early = rows.loc[~rows.prediction_time.isin(CURRENT_DELAY_TIMES)].assign(current_delay=np.nan)
    return pd.concat([early, late], ignore_index=True).drop(columns=["id_origin", "id_destination"])


def build(output, plan_dir, change_dir, monthly_dir):
    paths = sorted(monthly_dir.glob("data-????-??.parquet"))
    if not paths:
        raise FileNotFoundError(f"no monthly railway files in {monthly_dir}")
    first_exact_month = PLAN_EXACT_UTC_FROM.tz_localize(None).to_period("M")
    for path in paths:
        month = pd.Period(path.stem.removeprefix("data-"), freq="M")
        for directory in ((change_dir, plan_dir) if month >= first_exact_month else (change_dir,)):
            if not (directory / path.name).is_file():
                raise FileNotFoundError(f"missing auxiliary snapshot archive {directory / path.name}")
    weeks = read_assignment().query("split == 'cv'").week_start
    journeys = source_journeys(paths, source_segments(paths, weeks), weeks)
    upstream = read_upstream(paths, journeys)
    frames = []
    # Month batches bound the number of candidate stop events expanded at once.
    for month, month_rows in journeys.groupby(journeys[TIME_COLUMN].dt.to_period("M")):
        ids = set(upstream.loc[upstream.journey_id.isin(month_rows.journey_id), "id"])
        ids |= set(month_rows.id_origin) | set(month_rows.id_destination)
        options = {"months": (month - 1, month, month + 1), "plan_dir": plan_dir, "change_dir": change_dir}
        changes = _read_snapshot_ids_in_months(ids, kind="change", **options)
        plans = (_read_snapshot_ids_in_months(ids, kind="plan", **options) if month >= first_exact_month
                 else pd.DataFrame())
        if not plans.empty:
            plans = plans.loc[pd.to_datetime(plans.snapshot_timestamp, utc=True).ge(PLAN_EXACT_UTC_FROM)]
        for _, segment in month_rows.groupby("corridor", sort=False):
            stops = upstream.loc[upstream.journey_id.isin(segment.journey_id)]
            origins = segment.set_index("journey_id").origin_sequence
            stops = stops.loc[stops.train_line_station_num.le(stops.journey_id.map(origins))]
            frames.append(prediction_rows(segment, stops, changes, plans))
        print(f"{month}: {len(month_rows)} auxiliary journeys", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    if rows.duplicated(["corridor", *ROW_KEY]).any():
        raise ValueError("duplicate prediction keys in an auxiliary corridor")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    rows.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    print(json.dumps({"rows": len(rows), "by_corridor": rows.corridor.value_counts().to_dict(),
                      "live_by_horizon": rows.groupby("prediction_time").current_delay.count().to_dict()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=PLAN_DIR)
    parser.add_argument("--change-dir", type=Path, default=CHANGE_DIR)
    parser.add_argument("--monthly-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=ADVANCED / "corridor_training_cv.parquet")
    args = parser.parse_args()
    build(args.output, args.plan_dir, args.change_dir, args.monthly_dir)
