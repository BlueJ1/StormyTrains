"""Weather known at each prediction time along the Frankfurt -> München route.

This script adds the weather features below to every row of
data/prediction_rows_<split>.parquet and writes the combined railway-and-weather
rows, in the same order, to data/prediction_rows_weather_<split>.parquet. The
counts quoted in README.md go to weather_features_audit.json.

PREDICTION TIME. `predicted_at` is German clock time (planned departure minus
24 h, 6 h, ... on the clock, as for current_delay), so around a clock change
it can be 23 or 25 hours of real time before the departure. The weather uses
the earlier of `predicted_at` and the planned departure minus the real lead,
both in UTC. Local times in the repeated hour when the clocks go back take the
earlier moment; a time in the skipped spring hour moves back to just before
the change. So the weather is never later than either definition.

WEATHER HOUR. The latest DWD hour label at or before the prediction time.
DWD's hourly rain, mean wind and gust at label H are built from the 10-minute
values stamped H-60 to H-10 min (checked against the 10-minute data of
Frankfurt airport, 01420, on 17 October 2026), so they contain nothing after
H. Temperature is the value at the hour. The time DWD needs to publish an
observation is not modelled.

ROUTE. A fixed list of stations per branch (ROUTES), not the journey's
recorded stops, which can depend on the outcome (a diverted train records
other stations). A journey takes the usual branch of its train number: the
branch most of that train number's journeys in data/stops.parquet recorded
(via Würzburg or Nürnberg, or via Mannheim, Stuttgart or Ulm; all journeys,
hold-out included, as this is a timetable fact). Each station's weather
comes from weather/observations/railway_<variable>_final.parquet, where gaps
are already filled from a backup weather station.

FEATURES (the latest hour unless "24h": the 24 hours ending with it):
- origin_temperature_c, destination_temperature_c: Frankfurt and München;
- route_temp_max_c, route_temp_min_c, route_temp_max_24h_c, route_temp_min_24h_c;
- route_precip_max_mm: the highest hourly precipitation at a route station;
- route_precip_24h_mm: the 24-hour total at the wettest route station;
- route_wind_max_ms, route_wind_max_24h_ms: the highest hourly mean wind;
- route_gust_max_ms, route_gust_max_24h_ms;
- snow_on_route_24h: a route station reported solid or mixed precipitation
  (WRTR 7 or 8);
- route_snow_depth_cm: the deepest snow at a route station, from the latest
  daily measurement (06 UTC) at or before the prediction time.

MISSING VALUES. Aggregates use whatever stations and hours have a value; a
feature is missing only if none has. Nothing is filled in.

PRECIPITATION TYPE (WRTR) is not in the observation tables, so it is read here
from DWD's hourly precipitation archives, downloaded into weather/raw/wrtr/ on
the first run (delete that folder to download newer versions). For each
railway station: the primary DWD station in weather/station_mapping.json
first, its backup at the same hour when the primary has no usable code
(0, 6, 7 or 8), historical archives before recent ones.

Run from the project root: .venv/bin/python frankfurt_munich/weather_features.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from functools import cache
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "berlin_munich_weather"))
from fetch_weather import CTX  # noqa: E402  SSL context with the system certificates

DATA = HERE / "data"
WEATHER = HERE / "weather"
RAW = WEATHER / "raw" / "wrtr"
DWD = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/"
HOURLY = {"temperature": "temperature_c", "precipitation": "precipitation_mm",
          "mean_wind": "mean_wind_ms", "gust": "gust_ms"}
SNOW_CODES = [7, 8]  # solid; liquid and solid
ORIGIN, DESTINATION = "8000105", "8000261"  # EVA numbers of Frankfurt (Main) Hbf and München Hbf
ROUTES = {
    # Frankfurt, Hanau, Aschaffenburg, Würzburg, Nürnberg, Ingolstadt, München
    "nuernberg": [ORIGIN, "8000150", "8000010", "8000260", "8000284", "8000183", DESTINATION],
    # Frankfurt, Darmstadt, Mainz, Mannheim, Heidelberg, Stuttgart, Ulm, Augsburg, München-Pasing, München
    "stuttgart": [ORIGIN, "8000068", "8000240", "8000244", "8000156", "8000096", "8000170",
                  "8000013", "8004158", DESTINATION],
}
# A recorded stop here puts the journey on that branch: Würzburg, Nürnberg; Mannheim, Stuttgart, Ulm.
BRANCH_STATIONS = {"nuernberg": {"8000260", "8000284"}, "stuttgart": {"8000244", "8000096", "8000170"}}
# The columns added to every prediction row, besides weather_hour_utc (see FEATURES).
FEATURE_COLUMNS = ["origin_temperature_c", "destination_temperature_c",
                   "route_temp_max_c", "route_temp_min_c",
                   "route_temp_max_24h_c", "route_temp_min_24h_c",
                   "route_precip_max_mm", "route_precip_24h_mm",
                   "route_wind_max_ms", "route_wind_max_24h_ms",
                   "route_gust_max_ms", "route_gust_max_24h_ms",
                   "route_snow_depth_cm", "snow_on_route_24h"]


def hourly_table(variable: str, column: str) -> pd.DataFrame:
    """One column per railway station (EVA number), one row per UTC hour."""
    obs = pd.read_parquet(WEATHER / "observations" / f"railway_{variable}_final.parquet",
                          columns=["timestamp_utc", "railway_eva_id", column])
    return obs.pivot(index="timestamp_utc", columns="railway_eva_id", values=column).astype(float)


def fetch(name: str) -> bytes:
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        url = DWD + ("historical/" if name.endswith("_hist.zip") else "recent/") + name
        with urllib.request.urlopen(url, timeout=120, context=CTX) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


@cache
def wrtr_codes(archives: tuple[str, ...]) -> pd.Series:
    """WRTR codes of one DWD station by UTC hour; a historical row wins over a recent one."""
    parts = []
    for name in sorted(archives, key=lambda n: not n.endswith("_hist.zip")):
        with zipfile.ZipFile(io.BytesIO(fetch(name))) as archive:
            members = archive.namelist()
            meta = next(n for n in members if n.startswith("Metadaten_Parameter") and n.endswith(".txt"))
            params = pd.read_csv(archive.open(meta), sep=";", dtype=str, encoding="latin1")
            params = params[params.Parameter.str.strip().eq("WRTR") & params.Bis_Datum.str.strip().ge("20240629")]
            assert len(params) and params["Zusatz-Info"].str.contains("UTC").all(), f"WRTR not in UTC: {name}"
            product = next(n for n in members if n.startswith("produkt_"))
            table = pd.read_csv(archive.open(product), sep=";", dtype=str, skipinitialspace=True)
        table.columns = table.columns.str.strip()
        hours = pd.to_datetime(table.MESS_DATUM.str.strip(), format="%Y%m%d%H", utc=True)
        parts.append(pd.Series(pd.to_numeric(table.WRTR, errors="coerce").to_numpy(), index=hours))
    codes = pd.concat(parts)
    return codes[~codes.index.duplicated(keep="first")]


def precipitation_type(hours: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Usable WRTR code per railway station and hour (see PRECIPITATION TYPE)."""
    mapping = json.loads((WEATHER / "station_mapping.json").read_text())
    usable = mapping["metadata"]["wrtr_usable_codes"]
    study = hours >= pd.Timestamp(mapping["metadata"]["period_start"], tz="UTC")
    codes, audit = {}, {}
    for m in mapping["mappings"]:
        if m["variable"] != "WRTR":
            continue
        primary = wrtr_codes(tuple(m["primary_source_archives"])).reindex(hours)
        backup = wrtr_codes(tuple(m["backup_source_archives"])).reindex(hours)
        use_backup = ~primary.isin(usable) & backup.isin(usable)
        code = primary.where(~use_backup, backup)
        code = code.where(code.isin(usable))
        codes[m["railway_eva_id"]] = code
        audit[m["railway_station"]] = {"backup_hours": int(use_backup[study].sum()),
                                       "missing_hours": int(code[study].isna().sum())}
    return pd.DataFrame(codes), audit


def station_values() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per station: hourly values by (weather_hour_utc, eva), and snow depth by (snow_date, eva)."""
    t, p, w, g = (hourly_table(v, c) for v, c in HOURLY.items())
    wrtr, wrtr_audit = precipitation_type(t.index)
    snow = wrtr.isin(SNOW_CODES).astype(float).where(wrtr.notna())
    # Time-based windows: the 24 hour labels H-23 ... H, using the hours that have a value.
    hourly = {"temp": t, "temp_max_24h": t.rolling("24h").max(), "temp_min_24h": t.rolling("24h").min(),
              "precip": p, "precip_24h": p.rolling("24h").sum(),
              "wind": w, "wind_max_24h": w.rolling("24h").max(),
              "gust": g, "gust_max_24h": g.rolling("24h").max(), "snow_24h": snow.rolling("24h").max()}
    hourly = pd.concat({k: v.rename_axis(index="weather_hour_utc", columns="eva").stack()
                        for k, v in hourly.items()}, axis=1)
    depth = pd.read_parquet(WEATHER / "observations" / "railway_snow_depth_final.parquet",
                            columns=["observation_date", "railway_eva_id", "snow_depth_cm"])
    depth = depth.assign(snow_date=pd.to_datetime(depth.observation_date)).rename(columns={"railway_eva_id": "eva"})
    return hourly, depth.set_index(["snow_date", "eva"])[["snow_depth_cm"]].astype(float), wrtr_audit


def route_stations(stops: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(journey_id, eva) of each journey's fixed route (see ROUTE), and counts for the audit."""
    journeys = stops.groupby("journey_id").agg(train_number=("train_number", "first"), eva=("eva", set))
    recorded = journeys.eva.map(lambda evas: next((b for b, s in BRANCH_STATIONS.items() if evas & s), None))
    usual = recorded.dropna().groupby(journeys.train_number).agg(lambda b: b.mode().iloc[0])
    branch = journeys.train_number.map(usual)
    assert branch.notna().all(), "train number without a recorded branch"
    stations = pd.DataFrame([(b, eva) for b, evas in ROUTES.items() for eva in evas], columns=["branch", "eva"])
    route = branch.rename("branch").rename_axis("journey_id").reset_index().merge(stations, on="branch")
    audit = {"journeys_by_branch": branch.value_counts().to_dict(),
             "journeys_without_recorded_branch": int(recorded.isna().sum()),
             "journeys_recorded_on_the_other_branch": int((recorded.notna() & recorded.ne(branch)).sum())}
    return route[["journey_id", "eva"]], audit


def utc(local: pd.Series) -> pd.Series:
    """German local time in UTC: the earlier moment in a repeated hour, just before a skipped one."""
    return local.dt.tz_localize("Europe/Berlin", ambiguous=True, nonexistent="shift_backward").dt.tz_convert("UTC")


def features(rows: pd.DataFrame, route: pd.DataFrame, hourly: pd.DataFrame,
             depth: pd.DataFrame) -> pd.DataFrame:
    """The weather features of each prediction row (see FEATURES)."""
    rows = rows.reset_index(drop=True)
    departure = rows.origin_departure_planned_local
    real = utc(departure) - (departure - rows.predicted_at)  # departure minus the real lead
    at = utc(rows.predicted_at).where(lambda t: t.le(real), real)  # see PREDICTION TIME
    hour = at.dt.floor("h")
    snow_date = (at - pd.Timedelta(hours=6)).dt.floor("D").dt.tz_localize(None)
    # Nothing measured after the prediction time: the hour label and the 06 UTC snow measurement.
    assert hour.le(at).all() and (snow_date.dt.tz_localize("UTC") + pd.Timedelta(hours=6)).le(at).all()
    pairs = pd.DataFrame({"row": rows.index, "journey_id": rows.journey_id, "weather_hour_utc": hour,
                          "snow_date": snow_date}).merge(route, on="journey_id")
    pairs = pairs.join(hourly, on=["weather_hour_utc", "eva"]).join(depth, on=["snow_date", "eva"])
    aggregates = pairs.groupby("row").agg(
        route_temp_max_c=("temp", "max"), route_temp_min_c=("temp", "min"),
        route_temp_max_24h_c=("temp_max_24h", "max"), route_temp_min_24h_c=("temp_min_24h", "min"),
        route_precip_max_mm=("precip", "max"), route_precip_24h_mm=("precip_24h", "max"),
        route_wind_max_ms=("wind", "max"), route_wind_max_24h_ms=("wind_max_24h", "max"),
        route_gust_max_ms=("gust", "max"), route_gust_max_24h_ms=("gust_max_24h", "max"),
        route_snow_depth_cm=("snow_depth_cm", "max"), snow_on_route_24h=("snow_24h", "max"))
    assert len(aggregates) == len(rows)
    aggregates["snow_on_route_24h"] = aggregates.snow_on_route_24h.astype("boolean")
    out = rows[["journey_id", "prediction_time"]].assign(weather_hour_utc=hour)
    for name, eva in (("origin_temperature_c", ORIGIN), ("destination_temperature_c", DESTINATION)):
        out[name] = pairs[pairs.eva.eq(eva)].set_index("row").temp
    out = out.join(aggregates)
    assert list(out.columns) == ["journey_id", "prediction_time", "weather_hour_utc", *FEATURE_COLUMNS]
    return out


def main() -> None:
    stops = pd.read_parquet(DATA / "stops.parquet", columns=["journey_id", "train_number", "eva"])
    stops["eva"] = stops.eva.astype(int).astype(str)  # "08000105" in the stops, "8000105" in the weather tables
    route, route_audit = route_stations(stops)
    hourly, depth, wrtr_audit = station_values()
    assert set(route.eva) <= set(hourly.index.get_level_values("eva")), "route station without weather"
    first_hour = hourly.index.get_level_values("weather_hour_utc").min()
    audit = {"route": route_audit, "precipitation_type_by_station": wrtr_audit,
             "wrtr_archives_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob("*.zip"))}}
    for split in ("cv", "holdout"):
        rows = pd.read_parquet(DATA / f"prediction_rows_{split}.parquet").reset_index(drop=True)
        out = features(rows, route, hourly, depth)
        assert out.journey_id.equals(rows.journey_id) and out.prediction_time.equals(rows.prediction_time)
        rows.join(out.drop(columns=["journey_id", "prediction_time"])).to_parquet(
            DATA / f"prediction_rows_weather_{split}.parquet", index=False, compression="zstd")
        audit[split] = {"rows": len(out), "missing": {c: int(out[c].isna().sum()) for c in out if out[c].isna().any()},
                        "rows_with_24h_window_before_first_hour":
                            int(out.weather_hour_utc.lt(first_hour + pd.Timedelta(hours=23)).sum()),
                        "snow_on_route_24h_true": int(out.snow_on_route_24h.sum())}
    (HERE / "weather_features_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in audit.items() if k != "wrtr_archives_sha256"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
