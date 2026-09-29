"""Extract observed ICE/IC stops on dated Berlin Hbf to München Hbf runs.

The source `id` ends in a stop sequence number. Its prefix contains the
service's dated run identifier; `train_line_ride_id` alone repeats by date.
Source timestamps are retained as naive German-local wall times.
Only stops at stations in the stable panel (stations present in every monthly
file) are kept, so the recorded stop set does not change when the source
expanded its station coverage on 3 November 2025.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "monthly_processed_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
PANEL_FILE = ROOT / "data" / "stations_present_in_all_months.csv"
ORIGIN = "Berlin Hauptbahnhof"
DESTINATION = "München Hbf"
TRAIN_TYPES = ["ICE", "IC"]
ENDPOINT_COLUMNS = [
    "id", "station_name", "eva", "train_type", "train_line_ride_id",
    "train_line_station_num", "departure_planned_time", "arrival_planned_time",
]


def add_journey_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame["journey_id"] = frame["id"].str.rsplit("-", n=1).str[0]
    return frame


def read_endpoints(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for path in paths:
        table = pq.read_table(
            path,
            columns=ENDPOINT_COLUMNS,
            filters=[("train_type", "in", TRAIN_TYPES),
                     ("station_name", "in", [ORIGIN, DESTINATION])],
        )
        part = add_journey_id(table.to_pandas())
        part["source_month"] = path.stem.removeprefix("data-")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def choose_segments(endpoints: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    endpoint_duplicate_ids = int(endpoints.duplicated(subset=["id"]).sum())
    endpoint_conflicting_ids = int(
        endpoints.drop(columns=["source_month"]).drop_duplicates()
        .groupby("id").size().gt(1).sum()
    )
    endpoints = endpoints.drop_duplicates(subset=["id"], keep="first")
    origins = endpoints[endpoints.station_name.eq(ORIGIN)].copy()
    destinations = endpoints[endpoints.station_name.eq(DESTINATION)].copy()
    pairs = origins.merge(destinations, on="journey_id", suffixes=("_origin", "_destination"))
    pairs = pairs[
        pairs.train_line_station_num_origin.lt(pairs.train_line_station_num_destination)
        & pairs.train_line_ride_id_origin.eq(pairs.train_line_ride_id_destination)
        & pairs.train_type_origin.eq(pairs.train_type_destination)
    ].copy()
    pairs.sort_values(
        ["journey_id", "train_line_station_num_origin", "train_line_station_num_destination"],
        inplace=True,
    )
    pair_counts = pairs.groupby("journey_id").size()
    audit = {
        "endpoint_duplicate_ids": endpoint_duplicate_ids,
        "endpoint_conflicting_ids": endpoint_conflicting_ids,
        "endpoint_rows": len(endpoints),
        "origin_rows": len(origins),
        "destination_rows": len(destinations),
        "valid_endpoint_pairs": len(pairs),
        "journeys_with_multiple_endpoint_pairs": int(pair_counts.gt(1).sum()),
    }
    selected = pairs.drop_duplicates(subset=["journey_id"], keep="first")
    audit["planned_destination_before_origin"] = int((
        selected.arrival_planned_time_destination
        < selected.departure_planned_time_origin
    ).sum())
    audit["planned_endpoint_time_missing"] = int((
        selected.arrival_planned_time_destination.isna()
        | selected.departure_planned_time_origin.isna()
    ).sum())
    return selected, audit


def read_segments(paths: list[Path], segments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    wanted = segments.set_index("journey_id")[[
        "train_line_station_num_origin", "train_line_station_num_destination"
    ]]
    wanted_ids = set(wanted.index)
    parts = []
    duplicate_source_rows = 0
    for path in paths:
        table = pq.read_table(path, filters=[("train_type", "in", TRAIN_TYPES)])
        frame = add_journey_id(table.to_pandas())
        frame = frame[frame.journey_id.isin(wanted_ids)].copy()
        if frame.empty:
            continue
        frame = frame.join(wanted, on="journey_id")
        frame = frame[
            frame.train_line_station_num.ge(frame.train_line_station_num_origin)
            & frame.train_line_station_num.le(frame.train_line_station_num_destination)
        ].copy()
        frame["source_month"] = path.stem.removeprefix("data-")
        parts.append(frame)
    if not parts:
        raise ValueError("No segment stops found")
    out = pd.concat(parts, ignore_index=True)
    duplicate_source_rows = int(out.duplicated(subset=["id"]).sum())
    conflicting_source_ids = int(
        out.drop(columns=["source_month", "train_line_station_num_origin",
                          "train_line_station_num_destination"]).drop_duplicates()
        .groupby("id").size().gt(1).sum()
    )
    out = out.drop_duplicates(subset=["id"], keep="first").copy()
    out.rename(columns={
        "train_line_station_num_origin": "origin_sequence",
        "train_line_station_num_destination": "destination_sequence",
    }, inplace=True)
    out["stop_index"] = out.train_line_station_num - out.origin_sequence
    out["final_destination"] = out.train_line_station_num.eq(out.destination_sequence)
    out["event_planned_time"] = out.departure_planned_time.where(
        ~out.final_destination, out.arrival_planned_time
    )
    out["event_reported_time"] = out.departure_change_time.where(
        ~out.final_destination, out.arrival_change_time
    )
    out.sort_values(["journey_id", "stop_index", "id"], inplace=True)
    return out, {
        "duplicate_source_rows": duplicate_source_rows,
        "conflicting_source_ids": conflicting_source_ids,
    }


def main() -> None:
    paths = sorted(INPUT_DIR.glob("data-????-??.parquet"))
    if not paths:
        raise FileNotFoundError(f"No monthly Parquet files in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = set(pd.read_csv(PANEL_FILE).station_name)
    if not {ORIGIN, DESTINATION} <= panel:
        raise ValueError(f"{ORIGIN} and {DESTINATION} must be in {PANEL_FILE}")
    endpoints = read_endpoints(paths)
    segments, audit = choose_segments(endpoints)
    if segments.empty:
        raise ValueError("No Berlin Hbf to München Hbf ICE/IC journeys found")
    berlin_order_time = segments.departure_planned_time_origin.fillna(
        segments.arrival_planned_time_origin
    )
    munich_order_time = segments.arrival_planned_time_destination.fillna(
        segments.departure_planned_time_destination
    )
    reversed_order = munich_order_time.le(berlin_order_time)
    excluded = segments[reversed_order].copy()
    excluded["exclusion_reason"] = "Munich endpoint time is not later than Berlin endpoint time"
    audit["sequence_only_candidate_journeys"] = int(len(segments))
    audit["excluded_reverse_endpoint_candidates"] = int(len(excluded))
    audit["excluded_candidate_ids"] = excluded.journey_id.tolist()
    segments = segments[~reversed_order].copy()
    excluded.to_csv(OUTPUT_DIR / "train_excluded_candidates.csv", index=False)
    stops, stop_audit = read_segments(paths, segments)
    audit.update(stop_audit)
    start_times = segments.set_index("journey_id").departure_planned_time_origin
    end_times = segments.set_index("journey_id").arrival_planned_time_destination
    stops["outside_endpoint_planned_window"] = (
        stops.event_planned_time.lt(stops.journey_id.map(start_times))
        | stops.event_planned_time.gt(stops.journey_id.map(end_times))
    )
    by_journey = stops.groupby("journey_id", sort=False)
    coverage = by_journey.agg(
        first_observed_sequence=("train_line_station_num", "min"),
        last_observed_sequence=("train_line_station_num", "max"),
        observed_stops=("id", "size"),
        origin_sequence=("origin_sequence", "first"),
        destination_sequence=("destination_sequence", "first"),
    )
    coverage["missing_sequence_count"] = (
        coverage.destination_sequence - coverage.origin_sequence + 1 - coverage.observed_stops
    )
    missing = coverage[coverage.missing_sequence_count.gt(0)]
    # Gaps above are measured before the panel filter: they are stops the source
    # did not record. Stops removed by the panel filter are counted separately.
    outside_panel = ~stops.station_name.isin(panel)
    audit.update({
        "panel_file": str(PANEL_FILE.relative_to(ROOT)),
        "stops_before_panel_filter": int(len(stops)),
        "stops_removed_outside_panel": int(outside_panel.sum()),
        "stations_removed_outside_panel": sorted(stops.loc[outside_panel, "station_name"].unique()),
    })
    stops = stops[~outside_panel].copy()
    endpoint_counts = stops.assign(
        is_origin=stops.station_name.eq(ORIGIN)
        & stops.train_line_station_num.eq(stops.origin_sequence),
        is_destination=stops.station_name.eq(DESTINATION)
        & stops.train_line_station_num.eq(stops.destination_sequence),
    ).groupby("journey_id")[["is_origin", "is_destination"]].sum()
    audit.update({
        "source_files": [str(path.relative_to(ROOT)) for path in paths],
        "journeys": int(len(segments)),
        "observed_stops": int(len(stops)),
        "ICE_journeys": int(segments.train_type_origin.eq("ICE").sum()),
        "IC_journeys": int(segments.train_type_origin.eq("IC").sum()),
        "journeys_missing_sequence": int(len(missing)),
        "missing_sequence_rows": int(missing.missing_sequence_count.sum()),
        "journeys_missing_origin_row": int(coverage.first_observed_sequence.ne(coverage.origin_sequence).sum()),
        "journeys_missing_destination_row": int(coverage.last_observed_sequence.ne(coverage.destination_sequence).sum()),
        "intermediate_sequence_100_rows": int((
            stops.train_line_station_num.eq(100) & ~stops.final_destination
        ).sum()),
        "stops_outside_endpoint_planned_window": int(stops.outside_endpoint_planned_window.sum()),
        "journeys_without_exact_origin": int(endpoint_counts.is_origin.ne(1).sum()),
        "journeys_without_exact_destination": int(endpoint_counts.is_destination.ne(1).sum()),
        "min_selected_event_time": str(stops.event_planned_time.min()),
        "max_selected_event_time": str(stops.event_planned_time.max()),
        "reported_destination_before_origin": int((
            stops[stops.train_line_station_num.eq(stops.destination_sequence)]
            .set_index("journey_id").event_reported_time
            < stops[stops.train_line_station_num.eq(stops.origin_sequence)]
            .set_index("journey_id").event_reported_time
        ).sum()),
        "stops_missing_selected_planned_time": int(stops.event_planned_time.isna().sum()),
        "stops_missing_selected_reported_time": int(stops.event_reported_time.isna().sum()),
        "stops_arrival_canceled": int(stops.arrival_is_canceled.fillna(False).sum()),
        "stops_departure_canceled": int(stops.departure_is_canceled.fillna(False).sum()),
        "stations": stops[["station_name", "eva"]].drop_duplicates().sort_values(
            ["station_name", "eva"]
        ).to_dict("records"),
        "missing_sequence_detail": "data/train_missing_sequence_coverage.csv",
        "stop_index_meaning": "Original stop sequence minus Berlin sequence; gaps remain visible",
        "timestamp_basis": "Source naive German-local wall time; no conversion in extraction",
    })
    schema = pa.Schema.from_pandas(stops, preserve_index=False)
    pq.write_table(pa.Table.from_pandas(stops, schema=schema, preserve_index=False),
                   OUTPUT_DIR / "train_stops.parquet", compression="zstd")
    segments.to_csv(OUTPUT_DIR / "train_journeys.csv", index=False)
    missing.reset_index().to_csv(OUTPUT_DIR / "train_missing_sequence_coverage.csv", index=False)
    with (OUTPUT_DIR.parent / "extraction_audit.json").open("w") as file:
        json.dump(audit, file, indent=2, default=str, ensure_ascii=False)
    print(json.dumps({key: val for key, val in audit.items() if key not in (
        "stations", "source_files")}, indent=2))
    print("stations", len(audit["stations"]))


if __name__ == "__main__":
    main()
