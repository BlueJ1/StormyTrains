"""Extract dated ICE/IC journeys for the study corridors in corridors.py.

Railway data only. The rules are the same as in
berlin_munich_weather/extract_journeys.py, applied to each origin:

- A journey is one dated run (the source `id` without its stop number) that
  stops at the origin before the destination, with the same ride id and train
  type at both ends. Runs whose destination time is not later than the origin
  time are excluded.
- Only stops at stations in data/stations_present_in_all_months.csv are kept,
  so the station set does not change when the source expanded its coverage on
  3 November 2025. Gaps in the stop numbering are counted before that filter.
- Timestamps stay as naive German-local wall times.

The journey table carries the same targets as berlin_munich_weather/summarize.py:
arrival delay at the destination (from arrival times, because the source
delay_in_min is the departure delay where the train continues), a cancellation
flag (origin departure or destination arrival canceled) and the severity class.
The destination columns are named neutrally (destination_arrival_*,
arrival_delay_min); for the Berlin corridor they equal the munich_* columns of
berlin_munich_weather/data/journeys.parquet.

Run from the project root: .venv/bin/python corridor_analysis/extract.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from corridors import CORRIDORS

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "monthly_processed_data"
PANEL_FILE = ROOT / "data" / "stations_present_in_all_months.csv"
OUTPUT_DIR = HERE / "data"
TRAIN_TYPES = ["ICE", "IC"]
ENDPOINT_COLUMNS = [
    "id", "station_name", "eva", "train_type", "train_line_ride_id",
    "train_line_station_num", "departure_planned_time", "arrival_planned_time",
]


def add_journey_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame["journey_id"] = frame["id"].str.rsplit("-", n=1).str[0]
    return frame


def read_endpoints(paths: list[Path], stations: list[str]) -> pd.DataFrame:
    parts = []
    for path in paths:
        table = pq.read_table(path, columns=ENDPOINT_COLUMNS, filters=[
            ("train_type", "in", TRAIN_TYPES), ("station_name", "in", stations)])
        parts.append(add_journey_id(table.to_pandas()))
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["id"], keep="first")


def choose_segments(endpoints: pd.DataFrame, origin: str, destination: str) -> tuple[pd.DataFrame, dict]:
    origins = endpoints[endpoints.station_name.eq(origin)]
    destinations = endpoints[endpoints.station_name.eq(destination)]
    pairs = origins.merge(destinations, on="journey_id", suffixes=("_origin", "_destination"))
    pairs = pairs[
        pairs.train_line_station_num_origin.lt(pairs.train_line_station_num_destination)
        & pairs.train_line_ride_id_origin.eq(pairs.train_line_ride_id_destination)
        & pairs.train_type_origin.eq(pairs.train_type_destination)
    ].sort_values(["journey_id", "train_line_station_num_origin",
                   "train_line_station_num_destination"])
    audit = {"journeys_with_multiple_endpoint_pairs":
             int(pairs.groupby("journey_id").size().gt(1).sum())}
    pairs = pairs.drop_duplicates(subset=["journey_id"], keep="first")
    origin_time = pairs.departure_planned_time_origin.fillna(pairs.arrival_planned_time_origin)
    dest_time = pairs.arrival_planned_time_destination.fillna(pairs.departure_planned_time_destination)
    reversed_order = dest_time.le(origin_time)
    audit["sequence_only_candidate_journeys"] = int(len(pairs))
    audit["excluded_reverse_endpoint_candidates"] = int(reversed_order.sum())
    return pairs[~reversed_order].copy(), audit


def read_segments(paths: list[Path], segments: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """One pass over the monthly files, cutting out every corridor's segments."""
    wanted = {key: seg.set_index("journey_id")[["train_line_station_num_origin",
                                                "train_line_station_num_destination"]]
              for key, seg in segments.items()}
    parts = {key: [] for key in segments}
    for path in paths:
        frame = add_journey_id(pq.read_table(
            path, filters=[("train_type", "in", TRAIN_TYPES)]).to_pandas())
        for key, want in wanted.items():
            part = frame[frame.journey_id.isin(want.index)].join(want, on="journey_id")
            part = part[part.train_line_station_num.ge(part.train_line_station_num_origin)
                        & part.train_line_station_num.le(part.train_line_station_num_destination)]
            parts[key].append(part)
        print(path.name, flush=True)
    out = {}
    for key, frames in parts.items():
        stops = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"], keep="first")
        stops = stops.rename(columns={"train_line_station_num_origin": "origin_sequence",
                                      "train_line_station_num_destination": "destination_sequence"})
        stops["stop_index"] = stops.train_line_station_num - stops.origin_sequence
        stops["final_destination"] = stops.train_line_station_num.eq(stops.destination_sequence)
        stops["event_kind"] = np.where(stops.final_destination, "arrival", "departure")
        stops["event_planned_time"] = stops.departure_planned_time.where(
            ~stops.final_destination, stops.arrival_planned_time)
        stops["event_reported_time"] = stops.departure_change_time.where(
            ~stops.final_destination, stops.arrival_change_time)
        stops["selected_event_canceled"] = stops.departure_is_canceled.where(
            ~stops.final_destination, stops.arrival_is_canceled)
        out[key] = stops.sort_values(["journey_id", "stop_index", "id"])
    return out


def severity(delay: pd.Series, canceled: pd.Series) -> pd.Series:
    """Normal/minor below 20 min, moderate 20 to below 60, severe 60 or more."""
    classes = pd.cut(delay, [-np.inf, 20, 60, np.inf], right=False,
                     labels=["normal_minor", "moderate", "severe"]).astype("string")
    return classes.mask(canceled, "canceled")


def journey_table(stops: pd.DataFrame, gaps: pd.Series, origin: str, destination: str) -> pd.DataFrame:
    ordered = stops.sort_values(["journey_id", "stop_index", "id"])
    first = ordered.groupby("journey_id", sort=False).head(1).set_index("journey_id")
    last = ordered.groupby("journey_id", sort=False).tail(1).set_index("journey_id")
    assert first.station_name.eq(origin).all() and first.stop_index.eq(0).all()
    assert last.station_name.eq(destination).all() and last.final_destination.all()
    assert stops.groupby("journey_id").final_destination.sum().eq(1).all()
    summary = pd.DataFrame({
        "train_type": first.train_type, "train_number": first.train_number,
        "origin_departure_planned_local": first.departure_planned_time,
        "origin_departure_reported_local": first.departure_change_time,
        "destination_arrival_planned_local": last.arrival_planned_time,
        "destination_arrival_reported_local": last.arrival_change_time,
        "origin_departure_canceled": first.departure_is_canceled,
        "destination_arrival_canceled": last.arrival_is_canceled,
    })
    groups = stops.groupby("journey_id")
    summary["observed_stops"] = groups.size()
    summary["any_selected_stop_canceled"] = groups.selected_event_canceled.any()
    summary["recorded_route"] = ordered.groupby("journey_id").station_name.agg(" | ".join)
    summary["unobserved_sequence_positions"] = gaps.reindex(summary.index).fillna(0).astype(int)
    summary["journey_canceled"] = (summary.origin_departure_canceled.fillna(False).astype(bool)
                                   | summary.destination_arrival_canceled.fillna(False).astype(bool))
    delay = (last.arrival_change_time - last.arrival_planned_time).dt.total_seconds() / 60
    summary["arrival_delay_min"] = delay.mask(summary.journey_canceled)
    summary["severity_class"] = severity(summary.arrival_delay_min, summary.journey_canceled)
    summary["severe_or_canceled"] = summary.severity_class.isin(["severe", "canceled"])
    assert summary.severity_class.notna().all()
    return summary.sort_values("origin_departure_planned_local").reset_index()


def main() -> None:
    paths = sorted(INPUT_DIR.glob("data-????-??.parquet"))
    if not paths:
        raise FileNotFoundError(f"No monthly Parquet files in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = set(pd.read_csv(PANEL_FILE).station_name)
    ends = sorted({c[k] for c in CORRIDORS.values() for k in ("origin", "destination")})
    if not set(ends) <= panel:
        raise ValueError(f"Every origin and destination must be in {PANEL_FILE}")
    endpoints = read_endpoints(paths, ends)
    segments, audits = {}, {}
    for key, corridor in CORRIDORS.items():
        segments[key], audits[key] = choose_segments(endpoints, corridor["origin"], corridor["destination"])
    stops_by_corridor = read_segments(paths, segments)
    journeys, stops_all = [], []
    for key, corridor in CORRIDORS.items():
        stops = stops_by_corridor[key]
        seg = segments[key].set_index("journey_id")
        # Stop-number gaps are stops the source did not record; measured before
        # the panel filter, which removes stations outside the stable panel.
        gaps = (seg.train_line_station_num_destination - seg.train_line_station_num_origin + 1
                - stops.groupby("journey_id").size().reindex(seg.index).fillna(0))
        outside = ~stops.station_name.isin(panel)
        audits[key].update({
            "origin": corridor["origin"], "destination": corridor["destination"],
            "journeys": int(len(seg)),
            "ICE_journeys": int(seg.train_type_origin.eq("ICE").sum()),
            "IC_journeys": int(seg.train_type_origin.eq("IC").sum()),
            "stops_before_panel_filter": int(len(stops)),
            "stops_removed_outside_panel": int(outside.sum()),
            "stations_removed_outside_panel": sorted(stops.loc[outside, "station_name"].unique()),
            "journeys_missing_sequence": int(gaps.gt(0).sum()),
        })
        stops = stops[~outside].copy()
        table = journey_table(stops, gaps, corridor["origin"], corridor["destination"])
        audits[key].update({
            "observed_stops": int(len(stops)),
            "journeys_canceled": int(table.journey_canceled.sum()),
            "severity_class_counts": table.severity_class.value_counts().to_dict(),
        })
        table.insert(0, "corridor", key)
        stops.insert(0, "corridor", key)
        journeys.append(table)
        stops_all.append(stops.drop(columns=["origin_sequence", "destination_sequence"]))
    pd.concat(journeys, ignore_index=True).to_parquet(
        OUTPUT_DIR / "journeys.parquet", index=False, compression="zstd")
    pd.concat(stops_all, ignore_index=True).to_parquet(
        OUTPUT_DIR / "stops.parquet", index=False, compression="zstd")
    audits["source_files"] = [str(p.relative_to(ROOT)) for p in paths]
    audits["panel_file"] = str(PANEL_FILE.relative_to(ROOT))
    (HERE / "extraction_audit.json").write_text(
        json.dumps(audits, indent=2, default=str, ensure_ascii=False))
    print(json.dumps({k: v for k, v in audits.items() if k in CORRIDORS}, indent=2,
                     default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
