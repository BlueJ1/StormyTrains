"""Download DWD CDC station observations and match them to recorded train stops.

Every stop of a journey is matched to the weather at the same hour: the last full
UTC hour at or before the journey's scheduled Berlin Hbf departure. This is the
weather along the route when the train is due to leave, and it does not depend on
any delay.

Run from the project root with .venv/bin/python berlin_munich_weather/fetch_weather.py.
Raw public downloads and a SHA256 manifest are retained to make the join auditable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import ssl
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
DATA = ROOT / "data"
BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"
PRODUCTS = {
    "temperature": ("air_temperature", "TU", "TT_TU", "QN_9", "temperature_c"),
    "precipitation": ("precipitation", "RR", "R1", "QN_8", "precipitation_mm"),
    "wind": ("wind", "FF", "F", "QN_3", "wind_speed_ms"),
}
DB_STATIONS_VERSION = "5.0.2"
COORD_URL = f"https://unpkg.com/db-stations@{DB_STATIONS_VERSION}/data.ndjson"


def ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without CA certificates; fall back to the
    # system bundle there. Elsewhere the platform defaults work.
    if not ssl.get_default_verify_paths().cafile and Path("/etc/ssl/cert.pem").exists():
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ssl.create_default_context()


CTX = ssl_context()


def download(url: str, name: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    meta = path.with_name(path.name + ".source.json")
    if path.exists() and meta.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "University railway-weather research"})
            with urllib.request.urlopen(request, timeout=90, context=CTX) as response:
                content = response.read()
                details = {"url": url, "resolved_url": response.url,
                           "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                           "last_modified": response.headers.get("Last-Modified"),
                           "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            path.write_bytes(content)
            meta.write_text(json.dumps(details, indent=2))
            return content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def get_catalogue(product: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    folder, code, *_ = PRODUCTS[product]
    listings = {}
    for period in ("historical", "recent"):
        text = download(BASE + f"{folder}/{period}/", f"{code}_{period}_index.html").decode()
        listings[period] = re.findall(r'href="([^"]+\.zip)"', text)
    raw = download(BASE + f"{folder}/recent/{code}_Stundenwerte_Beschreibung_Stationen.txt",
                   f"{code}_stations.txt").decode("latin1")
    rows = []
    for line in raw.splitlines():
        match = re.match(r"\s*(\d{5})\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+(.+)", line)
        if match:
            station, start, end, height, lat, lon, rest = match.groups()
            rows.append({"dwd_station_id": station, "start": pd.Timestamp(start), "end": pd.Timestamp(end),
                         "height_m": int(height), "latitude": float(lat), "longitude": float(lon),
                         "dwd_station_name": re.split(r"\s{2,}", rest)[0]})
    if not rows:
        raise ValueError("Cannot parse station catalogue for " + product)
    catalogue = pd.DataFrame(rows)
    exclusion_file = ROOT / "dwd_excluded_stations.json"
    if exclusion_file.exists():
        excluded = json.loads(exclusion_file.read_text()).get(product, {})
        catalogue = catalogue[~catalogue.dwd_station_id.isin(excluded)].copy()
    return catalogue, listings


def railway_coordinates(stops: pd.DataFrame) -> pd.DataFrame:
    records = [json.loads(line) for line in download(COORD_URL, f"db_stations_{DB_STATIONS_VERSION}.ndjson").decode().splitlines()]
    by_id = {str(int(row["id"])): row for row in records}
    by_name = {row["name"]: row for row in records}
    rows = []
    for station_name, eva in stops[["station_name", "eva"]].drop_duplicates().itertuples(index=False, name=None):
        row = by_id.get(str(int(eva)))
        method = "exact_eva"
        if row is None:
            row = by_name.get(station_name)
            method = "exact_station_name_alternate_eva"
        if row is None:
            raise ValueError(f"No station coordinate for {station_name} EVA {eva}")
        rows.append({"station_name": station_name, "eva": eva, "latitude": row["location"]["latitude"],
                     "longitude": row["location"]["longitude"], "coordinate_source_id": row["id"],
                     "coordinate_source_name": row["name"], "coordinate_match": method})
    coords = pd.DataFrame(rows)
    coords.to_csv(DATA / "railway_station_coordinates.csv", index=False)
    return coords


def haversine(lat: float, lon: float, catalog: pd.DataFrame) -> np.ndarray:
    phi, lam = np.radians(lat), np.radians(lon)
    phi2, lam2 = np.radians(catalog.latitude), np.radians(catalog.longitude)
    a = np.sin((phi2-phi)/2)**2 + np.cos(phi)*np.cos(phi2)*np.sin((lam2-lam)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def station_observations(product: str, station_id: str, listings: dict, start: pd.Timestamp,
                         end: pd.Timestamp) -> pd.DataFrame:
    folder, code, variable, quality, output = PRODUCTS[product]
    parts = []
    for period in ("historical", "recent"):
        for filename in listings[period]:
            if not filename.startswith(f"stundenwerte_{code}_{station_id}_"):
                continue
            span = re.search(r"_(\d{8})_(\d{8})_hist", filename)
            if span and (pd.Timestamp(span[2], tz="UTC") + pd.Timedelta(days=1) < start or pd.Timestamp(span[1], tz="UTC") > end):
                continue
            content = download(BASE + f"{folder}/{period}/{filename}", filename)
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                metadata_file = next(n for n in archive.namelist() if "Metadaten_Parameter" in n and n.endswith(".txt"))
                metadata = pd.read_csv(archive.open(metadata_file), sep=";", encoding="latin1", dtype=str)
                relevant = metadata[metadata.Parameter.eq(variable) &
                                    metadata.Bis_Datum.ge(start.strftime("%Y%m%d")) &
                                    metadata.Von_Datum.le(end.strftime("%Y%m%d"))]
                if relevant.empty or not relevant["Zusatz-Info"].fillna("").str.contains("UTC").all():
                    raise ValueError(f"UTC time reference not verified for {filename}; inspect {metadata_file}")
                files = [n for n in archive.namelist() if n.startswith("produkt_") and n.endswith(".txt")]
                if len(files) != 1:
                    raise ValueError(f"Expected one data file in {filename}: {files}")
                frame = pd.read_csv(archive.open(files[0]), sep=";", dtype=str)
            frame.columns = frame.columns.str.strip()
            frame["weather_hour_utc"] = pd.to_datetime(frame.MESS_DATUM.str.strip(), format="%Y%m%d%H", utc=True)
            frame = frame[frame.weather_hour_utc.between(start, end)].copy()
            frame[output] = pd.to_numeric(frame[variable].str.strip(), errors="coerce").replace(-999, np.nan)
            frame["quality"] = pd.to_numeric(frame[quality].str.strip(), errors="coerce").astype("Int64")
            frame["source_file"] = filename
            frame["archive_period"] = period
            frame["dwd_station_id"] = station_id
            frame["product"] = product
            parts.append(frame[["product", "dwd_station_id", "weather_hour_utc", output, "quality", "source_file", "archive_period"]])
    if not parts:
        raise ValueError(f"No observations archive for {product} {station_id}")
    # Historical data have completed quality control and take precedence in overlap.
    result = pd.concat(parts, ignore_index=True).drop_duplicates("weather_hour_utc", keep="first")
    return result.sort_values("weather_hour_utc")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    stops = pd.read_parquet(DATA / "train_stops.parquet")
    stops["event_kind"] = np.where(stops.final_destination, "arrival", "departure")
    stops["selected_event_canceled"] = stops.departure_is_canceled.where(~stops.final_destination, stops.arrival_is_canceled)
    berlin_departure = stops[stops.stop_index.eq(0)].set_index("journey_id").departure_planned_time
    stops["weather_reference_time_local"] = stops.journey_id.map(berlin_departure)
    # An ambiguous autumn or nonexistent spring wall time cannot be resolved from these fields.
    stops["weather_reference_time_utc"] = stops.weather_reference_time_local.dt.tz_localize(
        "Europe/Berlin", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    stops["timezone_unresolved"] = stops.weather_reference_time_local.notna() & stops.weather_reference_time_utc.isna()
    stops["weather_hour_utc"] = stops.weather_reference_time_utc.dt.floor("h")
    stops["weather_timestamp_lag_minutes"] = (stops.weather_reference_time_utc-stops.weather_hour_utc).dt.total_seconds()/60
    coords = railway_coordinates(stops)
    start = stops.weather_hour_utc.min()
    end = stops.weather_hour_utc.max()
    mappings, observations = [], []
    for product in PRODUCTS:
        catalog, listings = get_catalogue(product)
        available_ids = {name.split("_")[2] for names in listings.values() for name in names}
        # A fixed station per railway stop and variable, with metadata coverage spanning
        # that stop's requested dates. Gaps remain missing, never silently interpolated.
        product_map = []
        for row in coords.itertuples(index=False):
            selected = stops[stops.station_name.eq(row.station_name) & stops.eva.eq(row.eva)]
            first, last = selected.weather_hour_utc.min(), selected.weather_hour_utc.max()
            candidates = catalog[catalog.dwd_station_id.isin(available_ids)].copy()
            candidates = candidates[(candidates.start <= first.tz_localize(None).normalize()) &
                                    (candidates.end >= last.tz_localize(None).normalize())].copy()
            if candidates.empty:
                raise ValueError(f"No {product} station covering dates for {row.station_name}")
            candidates["distance_km"] = haversine(row.latitude, row.longitude, candidates)
            chosen = candidates.sort_values(["distance_km", "dwd_station_id"]).iloc[0]
            product_map.append({"product": product, "station_name": row.station_name, "eva": row.eva,
                                **chosen.to_dict()})
        mapping = pd.DataFrame(product_map)
        mappings.append(mapping)
        ids = sorted(mapping.dwd_station_id.unique())
        print(f"{product}: fetching {len(ids)} DWD stations for {len(mapping)} railway station IDs", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            frames = list(pool.map(lambda sid: station_observations(product, sid, listings, start, end), ids))
        obs = pd.concat(frames, ignore_index=True)
        observations.append(obs)
        _, _, _, _, output = PRODUCTS[product]
        prefix = product + "_"
        renamed_mapping = mapping[["station_name", "eva", "dwd_station_id", "dwd_station_name", "distance_km"]].rename(
            columns={c: prefix+c for c in ["dwd_station_id", "dwd_station_name", "distance_km"]})
        stops = stops.merge(renamed_mapping, on=["station_name", "eva"], how="left", validate="many_to_one")
        renamed_obs = obs.drop(columns="product").rename(columns={c: prefix+c for c in ["dwd_station_id", "quality", "source_file", "archive_period"]})
        stops = stops.merge(renamed_obs, on=[prefix+"dwd_station_id", "weather_hour_utc"], how="left", validate="many_to_one")
        stops[prefix+"missing"] = stops[output].isna()
        print(f"{product}: {stops[output].notna().sum():,}/{len(stops):,} stops matched", flush=True)
    pd.concat(mappings, ignore_index=True).to_csv(DATA / "dwd_station_mapping.csv", index=False)
    pd.concat(observations, ignore_index=True).to_parquet(DATA / "dwd_hourly_observations.parquet", index=False, compression="zstd")
    stops.to_parquet(DATA / "journey_stops_weather.parquet", index=False, compression="zstd")
    stops.to_parquet(DATA / "journey_stops_weather_nearest_only.parquet", index=False, compression="zstd")
    stops.to_csv(DATA / "journey_stops_weather.csv.gz", index=False, compression="gzip")
    value_columns = [spec[-1] for spec in PRODUCTS.values()]
    stats = {"time_basis": "scheduled Berlin Hbf departure, same hour for every stop of a journey",
             "journeys": stops.journey_id.nunique(), "stops": len(stops),
             "first_berlin_departure_local": str(stops.weather_reference_time_local.min()),
             "last_berlin_departure_local": str(stops.weather_reference_time_local.max()),
             "timezone_unresolved": int(stops.timezone_unresolved.sum()),
             "selected_event_canceled": int(stops.selected_event_canceled.fillna(False).sum()),
             "complete_weather_stops": int(stops[value_columns].notna().all(axis=1).sum()),
             "missing_by_variable": stops[value_columns].isna().sum().to_dict(),
             "max_station_distance_km": {p: float(stops[p+"_distance_km"].max()) for p in PRODUCTS}}
    (ROOT / "weather_audit.json").write_text(json.dumps(stats, indent=2, default=int))
    manifest = {p.name.removesuffix(".source.json"): json.loads(p.read_text()) for p in sorted(CACHE.glob("*.source.json"))}
    (ROOT / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
