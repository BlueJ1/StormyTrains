"""Export historical NOAA GFS values and hourly weather by valid month."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DATASET = "noaa-gfs-forecast-virtual"
TIME_ZONE = "Europe/Berlin"
PRECIP = "total_precipitation_surface"
VARIABLES = [
    "temperature_2m", PRECIP, "wind_u_10m", "wind_v_10m",
    "wind_gust_surface", "snow_thickness_surface",
]
KEY = ["init_time_utc", "lead_hour", "grid_latitude", "grid_longitude"]
SCRIPT_DIR = Path(__file__).resolve().parent


def local_date(value):
    date = pd.Timestamp(value)
    if date.tzinfo is not None or date != date.normalize():
        raise argparse.ArgumentTypeError("Use a local date in YYYY-MM-DD format.")
    return date


def utc_boundary(date):
    return date.tz_localize(TIME_ZONE).tz_convert("UTC").tz_localize(None)


def extract(dataset, grids, start, end, max_lead):
    first_run = (start - pd.Timedelta(hours=max_lead)).ceil("6h")
    last_run = (end - pd.Timedelta(hours=1)).floor("6h")
    runs = pd.date_range(first_run, last_run, freq="6h")
    expected = pd.MultiIndex.from_product([runs, range(max_lead + 1)],
                                          names=["init_time", "lead_hour"]).to_frame(index=False)
    expected["lead_time"] = pd.to_timedelta(expected.lead_hour, unit="h")
    expected["valid_time"] = expected.init_time + expected.lead_time
    # Read one extra valid hour to difference precipitation at the boundary.
    expected = expected.loc[expected.valid_time.ge(start - pd.Timedelta(hours=1))
                            & expected.valid_time.lt(end)]
    available = expected.loc[expected.init_time.isin(dataset.init_time.values)]
    if available.empty:
        raise ValueError(f"No archived runs in {first_run}–{last_run} UTC")
    forecast_coords = {"forecast": available.index.to_numpy()}
    selection = dataset[VARIABLES].sel(
        init_time=xr.DataArray(available.init_time.to_numpy(), dims="forecast", coords=forecast_coords),
        lead_time=xr.DataArray(available.lead_time.to_numpy(), dims="forecast", coords=forecast_coords),
        latitude=xr.DataArray(grids.grid_latitude.to_numpy(), dims="grid"),
        longitude=xr.DataArray(grids.grid_longitude.to_numpy(), dims="grid"),
    )
    # Each variable has independent remote chunks. Read a few in parallel.
    with ThreadPoolExecutor(max_workers=4) as workers:
        columns = list(workers.map(lambda name: selection[name].load(), VARIABLES))
    subset = xr.merge(columns)[VARIABLES].reindex(forecast=expected.index)
    frame = subset.to_dataframe().reset_index()
    frame["init_time_utc"] = pd.to_datetime(frame.forecast.map(expected.init_time), utc=True)
    frame["valid_time_utc"] = pd.to_datetime(frame.forecast.map(expected.valid_time), utc=True)
    frame["lead_hour"] = frame.forecast.map(expected.lead_hour).astype("int16")
    frame = frame.rename(columns={"latitude": "grid_latitude", "longitude": "grid_longitude"})
    frame = frame.sort_values(KEY, ignore_index=True)
    hourly = frame.groupby(["init_time_utc", "grid_latitude", "grid_longitude"], sort=False)[PRECIP].diff()
    # GFS accumulates from the last six-hour step, resetting at 1, 7, 13, … h.
    hourly = hourly.where(frame.lead_hour % 6 != 1, frame[PRECIP]).where(frame.lead_hour > 0)
    frame["precipitation_1h_mm"] = hourly.where(hourly >= 0)
    negative_precip = int(hourly.lt(0).sum())
    frame = frame.loc[frame.valid_time_utc.ge(start.tz_localize("UTC"))].reset_index(drop=True)
    if frame.empty or frame.duplicated(KEY).any() or not frame.temperature_2m.notna().any():
        raise ValueError("The export is empty, has duplicate keys, or contains no temperature values.")
    frame["temperature_c"] = frame.temperature_2m
    frame["wind_speed_10m_ms"] = np.hypot(frame.wind_u_10m, frame.wind_v_10m)
    frame["wind_gust_ms"] = frame.wind_gust_surface
    frame["snow_depth_cm"] = frame.snow_thickness_surface * 100
    columns = KEY + ["valid_time_utc"]
    prepared = ["temperature_c", "precipitation_1h_mm", "wind_speed_10m_ms",
                "wind_gust_ms", "snow_depth_cm"]
    return frame[columns + VARIABLES], frame[columns + prepared], negative_precip


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def export_month(dataset, grids, start, end, max_lead, destination):
    label = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
    files = {name: destination / f"gfs_{name}_{label}.parquet" for name in ("raw", "weather")}
    report_path = destination / f"gfs_manifest_{label}.json"
    scope = {"dataset": DATASET, "time_zone": TIME_ZONE, "start_local": str(start.date()),
             "end_local_exclusive": str(end.date()), "max_lead_hours": max_lead,
             "grid_points": grids.to_dict(orient="records")}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["scope"] == scope and all(
            path.is_file() and sha256(path) == report["files"][name]["sha256"]
            for name, path in files.items()
        ):
            print(f"{label}: verified existing files", flush=True)
            return
    print(f"{label}: reading GFS for {len(grids)} grid points", flush=True)
    raw, weather, negative = extract(dataset, grids, utc_boundary(start), utc_boundary(end), max_lead)
    report = {"scope": scope, "rows": len(raw), "files": {},
              "negative_hourly_precip_values_in_read_window": negative}
    for name, frame in (("raw", raw), ("weather", weather)):
        path = files[name]
        temporary = path.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
        report["files"][name] = {"file": path.name, "bytes": path.stat().st_size,
                                 "sha256": sha256(path), "nulls": frame.isna().sum().to_dict()}
    temporary_report = report_path.with_suffix(".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    temporary_report.replace(report_path)
    print(f"{label}: exported {len(raw):,} rows per table", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=local_date, default="2024-07-01")
    parser.add_argument("--end", type=local_date, default="2026-10-01", help="Exclusive local date")
    parser.add_argument("--max-lead", type=int, default=48, help="Maximum GFS forecast lead, 1–120 hours")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "gfs_weather_monthly")
    args = parser.parse_args()
    if args.start >= args.end or not 1 <= args.max_lead <= 120:
        parser.error("Require start < end and a maximum lead between 1 and 120 hours.")

    import dynamical_catalog

    grids = pd.read_csv(SCRIPT_DIR / "gfs_stations.csv", usecols=KEY[2:]).drop_duplicates()
    args.output.mkdir(parents=True, exist_ok=True)
    with dynamical_catalog.open(DATASET, chunks=None) as dataset:
        start = args.start
        while start < args.end:
            end = min(start + pd.offsets.MonthBegin(1), args.end)
            export_month(dataset, grids, start, end, args.max_lead, args.output)
            start = end


if __name__ == "__main__":
    main()
