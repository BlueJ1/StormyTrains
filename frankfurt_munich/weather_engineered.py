"""Engineered weather features at each prediction time along the Frankfurt -> München route.

This script writes data/prediction_rows_weather_fe_<split>.parquet: the row key
of data/prediction_rows_<split>.parquet (`journey_id`, `prediction_time`,
`predicted_at`, same rows and order), the weather hour and the engineered
columns below. It does not change data/prediction_rows_weather_<split>.parquet
(the 14 base weather columns), which it only reads for the weather hour. The
counts quoted in modeling/README.md go to weather_engineered_audit.json.

TIME. The same weather hour as the base features (`weather_hour_utc`, the
latest DWD hour label at or before the prediction time; see
weather_features.py), read from the base weather file, so the two sets follow
the same rule. A window of "24h" is the 24 hour labels H-23 ... H ending with
that hour; likewise 48 h and 72 h. Snow depth uses the latest 06 UTC
measurement at or before the prediction time, as in the base features.

ROUTE. The same fixed station list per branch as the base features (the usual
branch of the train number, see weather_features.ROUTES and route_stations).
Every route feature is therefore a function of (branch, hour).

FEATURES. Thresholds fixed before screening: heat 30 °C, gust 20 m/s, heavy
rain 15 mm/h and a lower level of 5 mm/h, frost 0 °C.
- Heat: heat_degree_hours_24h = sum over route stations and hours of
  max(0, T - 30); heat_hours_24h = hours in which any route station was above
  30 °C; heat_flag_24h = any route station above 30 °C.
- Storm: gust_excess_sq_24h / _48h = sum over stations and hours of
  max(0, gust - 20)^2 (wind load grows with the square of the speed);
  storm_hours_24h / _48h = hours in which any route station's gust was above
  20 m/s; storm_flag_24h.
- Heavy rain: route_precip_48h_mm / _72h_mm = total at the wettest route
  station; for each level L in (15, 5): rain_excess{L}_24h = sum of
  max(0, P - L), rain_hours{L}_24h = hours with any station above L mm,
  rain_flag{L}_24h.
- Frost: frost_degree_hours_24h = sum of max(0, 0 - T); frost_hours_24h =
  hours with any route station below 0 °C; freeze_thaw_24h = the most changes
  between below 0 °C and at or above 0 °C, between consecutive hours, at one
  route station; snow_depth_change_cm = the largest change in snow depth at a
  route station since its measurement of the day before; frost_flag_24h = any
  route station below 0 °C.
- Location: the number of route stations that exceeded each threshold in the
  last 24 h (heat_stations_24h, storm_stations_24h, rain15_stations_24h,
  rain5_stations_24h, frost_stations_24h); and the 24 h maximum gust and the
  24 h precipitation total at Frankfurt Hbf and at München Hbf.
- Interaction (used from 1 h on): current_delay times heat_degree_hours_24h,
  gust_excess_sq_24h, rain_excess5_24h and frost_degree_hours_24h. Missing
  where current_delay is missing. The 5 mm level is used because the 15 mm
  excess is zero in almost every row.

MISSING VALUES. Sums, maxima and counts use whatever stations and hours have a
value (the observation tables have no gaps at present). Nothing is filled in.
The first rows have 48 h and 72 h windows that start before the weather tables
(29 June 2024); they use the hours that exist (counted in the audit).

Run from the project root: .venv/bin/python frankfurt_munich/weather_engineered.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weather_features import DESTINATION, ORIGIN, ROUTES, hourly_table, route_stations

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WEATHER = HERE / "weather"
ROW_KEY = ["journey_id", "prediction_time", "predicted_at"]

HEAT_C, GUST_MS, FROST_C = 30.0, 20.0, 0.0
RAIN_MM = (15, 5)

HEAT = ["heat_degree_hours_24h", "heat_hours_24h", "heat_flag_24h"]
STORM = ["gust_excess_sq_24h", "gust_excess_sq_48h", "storm_hours_24h", "storm_hours_48h",
         "storm_flag_24h"]
RAIN = ["route_precip_48h_mm", "route_precip_72h_mm",
        *(f"rain_{kind}{level}_24h" for level in RAIN_MM for kind in ("excess", "hours", "flag"))]
FROST = ["frost_degree_hours_24h", "frost_hours_24h", "freeze_thaw_24h", "snow_depth_change_cm",
         "frost_flag_24h"]
LOCATION = ["heat_stations_24h", "storm_stations_24h", "rain15_stations_24h", "rain5_stations_24h",
            "frost_stations_24h", "origin_gust_max_24h_ms", "destination_gust_max_24h_ms",
            "origin_precip_24h_mm", "destination_precip_24h_mm"]
INTERACTION_SOURCES = ["heat_degree_hours_24h", "gust_excess_sq_24h", "rain_excess5_24h",
                       "frost_degree_hours_24h"]
INTERACTION = [f"current_delay_x_{column}" for column in INTERACTION_SOURCES]
# Every engineered column, in file order. The interactions are features only from 1 h on.
FEATURE_COLUMNS = [*HEAT, *STORM, *RAIN, *FROST, *LOCATION, *INTERACTION]
GROUPS = {"heat": HEAT, "storm": STORM, "rain": RAIN, "frost": FROST, "location": LOCATION,
          "interaction": INTERACTION}


def rolling(table: pd.DataFrame, hours: int, how: str) -> pd.DataFrame:
    """Per station, over the hour labels H-(hours-1) ... H, skipping missing values."""
    window = table.rolling(f"{hours}h", min_periods=1)
    return getattr(window, how)()


def exceeds(table: pd.DataFrame, threshold: float, *, below: bool = False) -> pd.DataFrame:
    """1 where a value is beyond the threshold, 0 where not, missing where there is no value."""
    hit = table.lt(threshold) if below else table.gt(threshold)
    return hit.astype(float).where(table.notna())


def any_station(table: pd.DataFrame) -> pd.Series:
    """1 if any station is 1 in that hour, 0 if all with a value are 0, missing if none has one."""
    return table.max(axis=1, skipna=True)


def branch_hourly(t: pd.DataFrame, p: pd.DataFrame, g: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    """Route features of one branch by weather hour."""
    t, p, g = t[stations], p[stations], g[stations]
    heat, storm, frost = exceeds(t, HEAT_C), exceeds(g, GUST_MS), exceeds(t, FROST_C, below=True)
    below = t.lt(FROST_C).where(t.notna())
    crossing = below.ne(below.shift()).astype(float).where(below.notna() & below.shift().notna())
    out = {
        "heat_degree_hours_24h": rolling((t - HEAT_C).clip(lower=0), 24, "sum").sum(axis=1, min_count=1),
        "heat_hours_24h": any_station(heat).rolling("24h", min_periods=1).sum(),
        "heat_flag_24h": rolling(heat, 24, "max").max(axis=1),
        "storm_flag_24h": rolling(storm, 24, "max").max(axis=1),
        "route_precip_48h_mm": rolling(p, 48, "sum").max(axis=1),
        "route_precip_72h_mm": rolling(p, 72, "sum").max(axis=1),
        "frost_degree_hours_24h": rolling((FROST_C - t).clip(lower=0), 24, "sum").sum(axis=1, min_count=1),
        "frost_hours_24h": any_station(frost).rolling("24h", min_periods=1).sum(),
        "freeze_thaw_24h": rolling(crossing, 24, "sum").max(axis=1),
        "frost_flag_24h": rolling(frost, 24, "max").max(axis=1),
        "heat_stations_24h": rolling(heat, 24, "max").sum(axis=1, min_count=1),
        "storm_stations_24h": rolling(storm, 24, "max").sum(axis=1, min_count=1),
        "frost_stations_24h": rolling(frost, 24, "max").sum(axis=1, min_count=1),
    }
    for hours in (24, 48):
        out[f"gust_excess_sq_{hours}h"] = rolling((g - GUST_MS).clip(lower=0) ** 2, hours,
                                                  "sum").sum(axis=1, min_count=1)
        out[f"storm_hours_{hours}h"] = any_station(storm).rolling(f"{hours}h", min_periods=1).sum()
    for level in RAIN_MM:
        rain = exceeds(p, level)
        out[f"rain_excess{level}_24h"] = rolling((p - level).clip(lower=0), 24, "sum").sum(axis=1, min_count=1)
        out[f"rain_hours{level}_24h"] = any_station(rain).rolling("24h", min_periods=1).sum()
        out[f"rain_flag{level}_24h"] = rolling(rain, 24, "max").max(axis=1)
        out[f"rain{level}_stations_24h"] = rolling(rain, 24, "max").sum(axis=1, min_count=1)
    return pd.DataFrame(out)


def snow_change(depth: pd.DataFrame, stations: list[str]) -> pd.Series:
    """Largest change at a branch station since its measurement of the previous day, by date."""
    table = depth.pivot(index="snow_date", columns="eva", values="snow_depth_cm")[stations]
    table = table.reindex(pd.date_range(table.index.min(), table.index.max(), freq="D"))
    return table.diff().max(axis=1).rename("snow_depth_change_cm")


def load_weather():
    tables = {name: hourly_table(variable, column) for name, variable, column in
              (("t", "temperature", "temperature_c"), ("p", "precipitation", "precipitation_mm"),
               ("g", "gust", "gust_ms"))}
    # A complete hourly index, so that consecutive rows are consecutive hours.
    hours = pd.date_range(min(t.index.min() for t in tables.values()),
                          max(t.index.max() for t in tables.values()), freq="h")
    tables = {name: table.reindex(hours) for name, table in tables.items()}
    depth = pd.read_parquet(WEATHER / "observations" / "railway_snow_depth_final.parquet",
                            columns=["observation_date", "railway_eva_id", "snow_depth_cm"])
    depth = depth.assign(snow_date=pd.to_datetime(depth.observation_date), eva=depth.railway_eva_id,
                         snow_depth_cm=depth.snow_depth_cm.astype(float))
    return tables, depth


def journey_branches() -> pd.Series:
    """The branch name of each journey's fixed route (see weather_features.route_stations)."""
    stops = pd.read_parquet(DATA / "stops.parquet", columns=["journey_id", "train_number", "eva"])
    stops["eva"] = stops.eva.astype(int).astype(str)
    route, _ = route_stations(stops)
    by_set = {frozenset(stations): branch for branch, stations in ROUTES.items()}
    return route.groupby("journey_id").eva.agg(frozenset).map(by_set)


def engineered(rows: pd.DataFrame, branches: pd.Series, tables: dict, depth: pd.DataFrame) -> pd.DataFrame:
    """The engineered columns for each row of the base weather file, in its order."""
    t, p, g = tables["t"], tables["p"], tables["g"]
    rows = rows.reset_index(drop=True)
    branch = rows.journey_id.map(branches)
    assert branch.notna().all(), "journey without a branch"
    hour = rows.weather_hour_utc
    snow_date = (hour - pd.Timedelta(hours=6)).dt.floor("D").dt.tz_localize(None)
    out = pd.DataFrame(index=rows.index)
    for name, stations in ROUTES.items():
        mask = branch.eq(name)
        hourly = branch_hourly(t, p, g, stations).reindex(hour[mask])
        out.loc[mask, hourly.columns] = hourly.to_numpy()
        out.loc[mask, "snow_depth_change_cm"] = snow_change(depth, stations).reindex(snow_date[mask]).to_numpy()
    nodes = pd.DataFrame({
        "origin_gust_max_24h_ms": rolling(g[[ORIGIN]], 24, "max")[ORIGIN],
        "destination_gust_max_24h_ms": rolling(g[[DESTINATION]], 24, "max")[DESTINATION],
        "origin_precip_24h_mm": rolling(p[[ORIGIN]], 24, "sum")[ORIGIN],
        "destination_precip_24h_mm": rolling(p[[DESTINATION]], 24, "sum")[DESTINATION],
    }).reindex(hour)
    out[nodes.columns] = nodes.to_numpy()
    for source, column in zip(INTERACTION_SOURCES, INTERACTION):
        out[column] = rows.current_delay.to_numpy() * out[source]
    out = out.astype(float)
    return rows[[*ROW_KEY, "weather_hour_utc"]].join(out[FEATURE_COLUMNS])


def main() -> None:
    tables, depth = load_weather()
    branches = journey_branches()
    first_hour = tables["t"].index.min()
    audit = {"thresholds": {"heat_c": HEAT_C, "gust_ms": GUST_MS, "rain_mm_per_h": list(RAIN_MM),
                            "frost_c": FROST_C}}
    for split in ("cv", "holdout"):
        prediction = pd.read_parquet(DATA / f"prediction_rows_{split}.parquet", columns=[*ROW_KEY, "current_delay"])
        base = pd.read_parquet(DATA / f"prediction_rows_weather_{split}.parquet", columns=[*ROW_KEY, "weather_hour_utc"])
        if not base[ROW_KEY].equals(prediction[ROW_KEY]):
            raise ValueError(f"prediction_rows_weather_{split}.parquet is out of date; "
                             "re-run frankfurt_munich/weather_features.py")
        rows = prediction.join(base[["weather_hour_utc"]])
        out = engineered(rows, branches, tables, depth)
        assert out[ROW_KEY].equals(prediction[ROW_KEY])
        out.to_parquet(DATA / f"prediction_rows_weather_fe_{split}.parquet", index=False, compression="zstd")
        weather = out[[c for c in FEATURE_COLUMNS if c not in INTERACTION]]
        audit[split] = {
            "rows": len(out),
            "missing": {c: int(weather[c].isna().sum()) for c in weather if weather[c].isna().any()},
            "rows_with_72h_window_before_first_hour":
                int(out.weather_hour_utc.lt(first_hour + pd.Timedelta(hours=71)).sum()),
            "share_of_rows_nonzero": {c: round(float(weather[c].fillna(0).ne(0).mean()), 4) for c in weather},
        }
    (HERE / "weather_engineered_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["cv"], indent=2))


if __name__ == "__main__":
    main()
