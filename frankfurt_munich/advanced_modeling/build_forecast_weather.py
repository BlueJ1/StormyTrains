"""Build GFS forecast weather inputs for the hours of the planned journey, without reading targets.

Step 10.8 of PLAN.md (finishing plan B13-B16, decision Q13). The forecasts
supplement the DWD observations of build_advanced_weather.py; they do not
replace them.

RUN. Each row uses the latest GFS run that started at or before its cutoff
(`weather_features.cutoff_utc`). A run counts as available from its start
time (B13, decided 17 November 2026): a system in use would receive forecasts
in near real time. NOAA's own files reach its AWS bucket about 3.8 h after the
start (median of 1,100 sampled runs), so this is optimistic by that much.

HOURS. The planned journey: from the hour of the planned Frankfurt departure,
rounded down, to the hour of the planned München arrival, rounded up, in UTC.
Temperature, wind, gust and snow depth are values at the hour, so they use
every hour of that window. GFS precipitation at hour H fell in the hour
before H, so precipitation uses the window's hours after the first. The run
starts on the hour and at or before the departure, so precipitation never
needs lead 0, where it is missing. Leads over 48 h are not in the extract;
such hours count as missing.

ROUTE. The fixed station list of the journey's weather branch (from the
timetable inputs, as for the DWD weather), each station on its nearest GFS
cell (`forecast_weather_GFS/gfs_stations.csv`). Without a branch only the two
endpoint temperatures are kept, as in build_advanced_weather.py.

FEATURES (journey hours, route cells):
- forecast_origin_temperature_c: Frankfurt's cell at the first hour;
- forecast_destination_temperature_c: München's cell at the last hour;
- forecast_route_temp_max_c, forecast_route_temp_min_c;
- forecast_route_precip_max_mm: the highest hourly precipitation;
- forecast_route_precip_total_mm: the journey total at the wettest cell;
- forecast_route_wind_max_ms, forecast_route_gust_max_ms;
- forecast_route_snow_depth_cm: the deepest snow (B14: no precipitation type).
Aggregates use the cells and hours that have a value, as for the DWD weather.
`forecast_hours_missing` counts the journey hours in which some route cell
lacks a value of any variable.
The *_lag24h columns are the same features with the cutoff and the journey
moved 24 h back in real time, for the 24-hour change representation (1.5).

Run from the project root:
.venv/bin/python frankfurt_munich/advanced_modeling/build_forecast_weather.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  shared modelling code and dataset builders
from advanced_data import ADVANCED
from data import DATA, ROW_KEY, TIME_COLUMN, merge_inputs
from holdout_gate import REGISTRY, committed_registration
from weather_features import DESTINATION, ORIGIN, ROUTES, branch_routes, cutoff_utc, utc

GFS = Path(__file__).resolve().parents[1] / "forecast_weather_GFS"
MAX_LEAD_HOURS = 48
FORECAST = ("forecast_origin_temperature_c", "forecast_destination_temperature_c",
            "forecast_route_temp_max_c", "forecast_route_temp_min_c",
            "forecast_route_precip_max_mm", "forecast_route_precip_total_mm",
            "forecast_route_wind_max_ms", "forecast_route_gust_max_ms",
            "forecast_route_snow_depth_cm")
FORECAST_LAG = tuple(f"{column}_lag24h" for column in FORECAST)
ROUTE_COLUMNS = tuple(c for c in FORECAST if c.startswith("forecast_route_"))
GFS_COLUMNS = ["init_time_utc", "valid_time_utc", "grid_latitude", "grid_longitude",
               "temperature_c", "precipitation_1h_mm", "wind_speed_10m_ms", "wind_gust_ms", "snow_depth_cm"]


def load_gfs(directory: Path = GFS / "data") -> pd.DataFrame:
    """Processed GFS rows, one per run, valid hour and grid cell, with the cell as `cell`."""
    paths = sorted(directory.glob("gfs_weather_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no GFS partitions in {directory}")
    gfs = pd.concat([pd.read_parquet(p, columns=GFS_COLUMNS) for p in paths], ignore_index=True)
    key = ["init_time_utc", "valid_time_utc", "grid_latitude", "grid_longitude"]
    if gfs.duplicated(key).any():
        raise ValueError("duplicate GFS rows")
    gfs = gfs[(gfs.valid_time_utc - gfs.init_time_utc).le(pd.Timedelta(hours=MAX_LEAD_HOURS))].copy()
    for column in ("init_time_utc", "valid_time_utc"):
        gfs[column] = gfs[column].astype("datetime64[ns, UTC]")
    return gfs.assign(cell=cell_name(gfs.grid_latitude, gfs.grid_longitude)).drop(
        columns=["grid_latitude", "grid_longitude"])


def cell_name(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """The join key of a grid cell, "lat,lon" with two decimals; formats each distinct cell once."""
    codes, cells = pd.MultiIndex.from_arrays([np.asarray(lat, dtype=float),
                                              np.asarray(lon, dtype=float)]).factorize()
    names = np.array([f"{a:.2f},{b:.2f}" for a, b in cells], dtype=object)
    return pd.Series(names[codes], index=lat.index)


def station_cells(path: Path = GFS / "gfs_stations.csv") -> dict[str, str]:
    """EVA number (as in weather_features.ROUTES) -> GFS cell."""
    stations = pd.read_csv(path, dtype={"railway_eva_id": str})
    return dict(zip(stations.railway_eva_id, cell_name(stations.grid_latitude, stations.grid_longitude)))


def latest_run(at: pd.Series, runs: pd.DatetimeIndex) -> pd.Series:
    """The latest run that started at or before each moment (B13); NaT if none."""
    runs = runs.sort_values()
    position = np.searchsorted(runs.asi8, at.dt.tz_convert("UTC").astype(runs.dtype).astype("int64"),
                               side="right") - 1
    chosen = pd.Series(runs[np.clip(position, 0, None)], index=at.index)
    return chosen.where(position >= 0)


def forecast_features(rows: pd.DataFrame, gfs: pd.DataFrame, cells: dict[str, str], *,
                      lag_hours: int = 0) -> pd.DataFrame:
    """Forecast features per row. `rows` needs the row key, the planned departure (local),
    `destination_arrival_planned_local` and `weather_branch`."""
    if lag_hours < 0:
        raise ValueError("forecast lag must be non-negative")
    rows = rows.reset_index(drop=True)
    branch, stations = branch_routes(rows["weather_branch"])
    lag = pd.Timedelta(hours=lag_hours)
    at = cutoff_utc(rows) - lag
    departure = utc(rows.origin_departure_planned_local) - lag
    arrival = utc(rows.destination_arrival_planned_local) - lag
    if not departure.lt(arrival).all():
        raise ValueError("planned arrival not after planned departure")
    run = latest_run(at, pd.DatetimeIndex(gfs.init_time_utc.unique()))
    # B13: a run is used only if it started by the cutoff, and so by the departure.
    known = run.notna()
    if not (run[known].le(at[known]).all() and run[known].le(departure[known]).all()):
        raise ValueError("a forecast run starts after the cutoff or the departure (B13)")
    first, last = departure.dt.floor("h"), arrival.dt.ceil("h")

    route_stations = {eva for evas in (*ROUTES.values(), [ORIGIN, DESTINATION]) for eva in evas}
    if missing_stations := route_stations - cells.keys():
        raise ValueError(f"route stations without a GFS cell: {sorted(missing_stations)}")
    # The join key is formatted from floats on both sides; a mismatch would leave every value missing.
    if missing_cells := {cells[eva] for eva in route_stations} - set(gfs.cell.unique()):
        raise ValueError(f"station cells not in the GFS data: {sorted(missing_cells)}")
    route = (pd.DataFrame({"row": rows.index, "eva": stations}).explode("eva")
             .assign(cell=lambda f: f.eva.map(cells)).drop_duplicates(["row", "cell"]))
    hours = ((last - first) // pd.Timedelta(hours=1)).astype(int)
    window = pd.DataFrame({"row": rows.index.repeat(hours + 1)})
    window["step"] = window.groupby("row").cumcount()
    window["valid_time_utc"] = (first.reindex(window.row).reset_index(drop=True)
                                + pd.to_timedelta(window.step, unit="h")).astype("datetime64[ns, UTC]")
    window["init_time_utc"] = run.reindex(window.row).reset_index(drop=True).astype("datetime64[ns, UTC]")
    pairs = window.merge(route[["row", "cell"]], on="row").merge(
        gfs, on=["init_time_utc", "valid_time_utc", "cell"], how="left")
    pairs.loc[pairs.step.eq(0), "precipitation_1h_mm"] = np.nan  # fell before the departure hour

    route_rows = pairs.groupby("row")
    totals = pairs.groupby(["row", "cell"]).precipitation_1h_mm.sum(min_count=1).groupby("row").max()
    out = pd.DataFrame(index=rows.index)
    origin, destination = cells[ORIGIN], cells[DESTINATION]
    out["forecast_origin_temperature_c"] = (
        pairs[pairs.cell.eq(origin) & pairs.step.eq(0)].set_index("row").temperature_c)
    is_last = pairs.step.eq(hours.reindex(pairs.row).to_numpy())
    out["forecast_destination_temperature_c"] = (
        pairs[pairs.cell.eq(destination) & is_last].set_index("row").temperature_c)
    out["forecast_route_temp_max_c"] = route_rows.temperature_c.max()
    out["forecast_route_temp_min_c"] = route_rows.temperature_c.min()
    out["forecast_route_precip_max_mm"] = route_rows.precipitation_1h_mm.max()
    out["forecast_route_precip_total_mm"] = totals
    out["forecast_route_wind_max_ms"] = route_rows.wind_speed_10m_ms.max()
    out["forecast_route_gust_max_ms"] = route_rows.wind_gust_ms.max()
    out["forecast_route_snow_depth_cm"] = route_rows.snow_depth_cm.max()
    out.loc[branch.eq("unknown"), list(ROUTE_COLUMNS)] = np.nan
    out["forecast_run_utc"] = run
    # Journey hours in which some route cell lacks a value (precipitation only after the first hour).
    instant = ["temperature_c", "wind_speed_10m_ms", "wind_gust_ms", "snow_depth_cm"]
    gap = pairs[instant].isna().any(axis=1) | (pairs.step.gt(0) & pairs.precipitation_1h_mm.isna())
    out["forecast_hours_missing"] = gap.groupby([pairs.row, pairs.step]).any().groupby(level=0).sum().astype(int)
    return pd.concat([rows[ROW_KEY], out[["forecast_run_utc", "forecast_hours_missing", *FORECAST]]], axis=1)


def build(rows: pd.DataFrame, gfs: pd.DataFrame, cells: dict[str, str]) -> pd.DataFrame:
    current = forecast_features(rows, gfs, cells)
    earlier = forecast_features(rows, gfs, cells, lag_hours=24)
    if not earlier[ROW_KEY].equals(current[ROW_KEY]):
        raise ValueError("the lagged forecast rows are not in the order of the current ones")
    current[list(FORECAST_LAG)] = earlier[list(FORECAST)].to_numpy()
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs-dir", type=Path, default=ADVANCED)
    parser.add_argument("--split", choices=("cv", "holdout"), default="cv")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    if args.split == "holdout":
        committed_registration(args.registry)
    rows = pd.read_parquet(DATA / f"prediction_rows_{args.split}.parquet",
                           columns=[*ROW_KEY, TIME_COLUMN])
    # The planned arrival is part of the prediction cohort (one row per planned departure and arrival).
    arrival = pd.read_parquet(DATA / f"journeys_{args.split}.parquet",
                              columns=["journey_id", "destination_arrival_planned_local"])
    rows = rows.merge(arrival, on="journey_id", how="left", validate="many_to_one")
    timetable = pd.read_parquet(args.inputs_dir / f"timetable_{args.split}.parquet",
                                columns=[*ROW_KEY, "weather_branch"])
    rows = merge_inputs(rows, timetable).reset_index(drop=True)
    output = build(rows, load_gfs(), station_cells())
    output.to_parquet(args.inputs_dir / f"forecast_weather_{args.split}.parquet", index=False,
                      compression="zstd")
    by_time = output.groupby("prediction_time", observed=True)
    audit = {
        "rows": len(output),
        "known": {str(k): {c: int(v) for c, v in g[list(FORECAST)].count().items()} for k, g in by_time},
        "lag24h_known": {str(k): {c: int(v) for c, v in g[list(FORECAST_LAG)].count().items()} for k, g in by_time},
        "rows_with_missing_hours": int(output.forecast_hours_missing.gt(0).sum()),
        "run_age_at_departure_h": {
            str(k): {q: float(v) for q, v in ((utc(rows.loc[g.index, "origin_departure_planned_local"])
                                               - g.forecast_run_utc) / pd.Timedelta(hours=1))
                     .describe()[["min", "50%", "max"]].items()} for k, g in by_time},
    }
    (args.inputs_dir / f"forecast_weather_{args.split}_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
