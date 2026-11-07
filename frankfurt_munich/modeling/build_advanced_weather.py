"""Build weather inputs from the archived timetable, without reading targets."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from advanced_data import ADVANCED
from data import DATA, ROW_KEY, TIME_COLUMN, merge_inputs
from features import WEATHER
from holdout_gate import REGISTRY, committed_registration
from representations import WEATHER_LAG
from weather_features import DESTINATION, ORIGIN, ROUTES, features, station_values


def weather_from_timetable(rows, timetable, hourly, depth):
    rows = merge_inputs(rows, timetable[[*ROW_KEY, "weather_branch"]])
    unrecognized = set(rows["weather_branch"].dropna()) - ROUTES.keys()
    if unrecognized:
        raise ValueError(f"unrecognized timetable weather branches: {sorted(unrecognized)}")
    routes = {**ROUTES, "unknown": [ORIGIN, DESTINATION]}
    branch = rows["weather_branch"].astype("string").fillna("unknown")
    route = rows[ROW_KEY[:2]].assign(eva=branch.map(routes)).explode("eva")
    current = features(rows, route, hourly, depth)
    earlier = features(rows, route, hourly, depth, lag_hours=24)
    route_columns = [column for column in WEATHER if column.startswith("route_")
                     or column == "snow_on_route_24h"]
    for frame in (current, earlier):
        frame.loc[branch.eq("unknown"), route_columns] = None
    weather = current.drop(columns=ROW_KEY[:2])
    weather[list(WEATHER_LAG)] = earlier[WEATHER]
    return rows[ROW_KEY].join(weather)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=ADVANCED)
    parser.add_argument("--split", choices=("cv", "holdout"), default="cv")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    if args.split == "holdout":
        committed_registration(args.registry)
    rows = pd.read_parquet(DATA / f"prediction_rows_{args.split}.parquet", columns=[*ROW_KEY, TIME_COLUMN])
    timetable = pd.read_parquet(args.inputs_dir / f"timetable_{args.split}.parquet",
                                columns=[*ROW_KEY, "weather_branch"])
    hourly, depth, _ = station_values()
    output = weather_from_timetable(rows, timetable, hourly, depth)
    output.to_parquet(args.inputs_dir / f"weather_{args.split}.parquet", index=False, compression="zstd")
    print(output.groupby("prediction_time", observed=True)[WEATHER].count().to_string())


if __name__ == "__main__":
    main()
