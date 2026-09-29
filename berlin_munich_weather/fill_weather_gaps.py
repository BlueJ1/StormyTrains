"""Use the nearest alternative DWD station with a value at the same labeled hour.

Keep the original nearest-station match and expose every substitution. Do not
interpolate time or select a later hour. Stations further than MAX_DISTANCE_KM
are never used; a value that would need one stays missing and is flagged in
`<variable>_beyond_distance_limit`. Ambiguous DST times remain unresolved,
with both possible weather-hour matches written to a separate candidate file.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
MAX_DISTANCE_KM = 50.0

from fetch_weather import ROOT, DATA, CACHE, PRODUCTS, get_catalogue, haversine, station_observations


def main():
    source = DATA / "journey_stops_weather_nearest_only.parquet"
    if not source.exists():
        source.write_bytes((DATA / "journey_stops_weather.parquet").read_bytes())
    s = pd.read_parquet(source)
    observations = pd.read_parquet(DATA / "dwd_hourly_observations.parquet")
    coordinates = pd.read_csv(DATA / "railway_station_coordinates.csv", dtype={"eva": str})
    additional_observations = []
    start, end = s.weather_hour_utc.min(), s.weather_hour_utc.max()
    for product, (_, _, _, _, variable) in PRODUCTS.items():
        catalog, listings = get_catalogue(product)
        available_ids = {name.split("_")[2] for names in listings.values() for name in names}
        s[product+"_primary_dwd_station_id"] = s[product+"_dwd_station_id"]
        too_far = s[product+"_distance_km"].gt(MAX_DISTANCE_KM)
        s.loc[too_far, variable] = np.nan
        s.loc[too_far, [product+"_"+c for c in ["dwd_station_id", "dwd_station_name", "distance_km",
                                                  "quality", "source_file", "archive_period"]]] = None
        s[product+"_primary_missing"] = s[variable].isna()
        s[product+"_station_rank"] = 1
        s[product+"_used_fallback_station"] = False
        cached = {}
        for sid, frame in observations[observations["product"].eq(product)].groupby("dwd_station_id"):
            cached[sid] = frame
        groups = s[s[variable].isna() & s.weather_hour_utc.notna()].groupby(["station_name", "eva"])
        for (station_name, eva), missing_group in groups:
            coord = coordinates[coordinates.eva.eq(eva) & coordinates.station_name.eq(station_name)].iloc[0]
            all_station = s[s.eva.eq(eva) & s.station_name.eq(station_name)]
            first, last = all_station.weather_hour_utc.min(), all_station.weather_hour_utc.max()
            eligible = catalog[catalog.dwd_station_id.isin(available_ids) &
                               catalog.start.le(first.tz_localize(None).normalize()) &
                               catalog.end.ge(last.tz_localize(None).normalize())].copy()
            eligible["distance_km"] = haversine(coord.latitude, coord.longitude, eligible)
            eligible = eligible[eligible.distance_km.le(MAX_DISTANCE_KM)].sort_values(["distance_km", "dwd_station_id"])
            primary_id = missing_group[product+"_primary_dwd_station_id"].iloc[0]
            unresolved = missing_group.index
            for rank, candidate in enumerate(eligible.itertuples(index=False), start=1):
                if len(unresolved) == 0:
                    break
                if candidate.dwd_station_id == primary_id:
                    continue
                sid = candidate.dwd_station_id
                if sid not in cached:
                    frame = station_observations(product, sid, listings, start, end)
                    cached[sid] = frame
                    additional_observations.append(frame)
                frame = cached[sid].set_index("weather_hour_utc")
                matched = frame.reindex(s.loc[unresolved, "weather_hour_utc"])
                valid = matched[variable].notna().to_numpy()
                idx = unresolved[valid]
                if not len(idx):
                    continue
                values = matched.loc[matched[variable].notna()]
                s.loc[idx, variable] = values[variable].to_numpy()
                s.loc[idx, product+"_dwd_station_id"] = sid
                s.loc[idx, product+"_dwd_station_name"] = candidate.dwd_station_name
                s.loc[idx, product+"_distance_km"] = candidate.distance_km
                for col in ["quality", "source_file", "archive_period"]:
                    s.loc[idx, product+"_"+col] = values[col].to_numpy()
                s.loc[idx, product+"_station_rank"] = rank
                s.loc[idx, product+"_used_fallback_station"] = True
                unresolved = unresolved[~valid]
            print(f"{product}: {station_name} filled {len(missing_group)-len(unresolved)}/{len(missing_group)}", flush=True)
        s[product+"_missing"] = s[variable].isna()
        # Missing although the hour is known: no station within the limit had a value.
        s[product+"_beyond_distance_limit"] = s[product+"_missing"] & s.weather_hour_utc.notna()
    if additional_observations:
        observations = pd.concat([observations, *additional_observations], ignore_index=True).drop_duplicates(
            ["product", "dwd_station_id", "weather_hour_utc"])
        observations.to_parquet(DATA / "dwd_hourly_observations.parquet", index=False, compression="zstd")
    # The source has no UTC offset. Expose both UTC interpretations; do not pick one.
    ambiguous_candidates = []
    for row in s[s.timezone_unresolved].itertuples(index=False):
        for summer_time in (True, False):
            utc = pd.Timestamp(row.weather_reference_time_local).tz_localize("Europe/Berlin", ambiguous=summer_time, nonexistent="NaT").tz_convert("UTC")
            if pd.isna(utc):
                continue
            hour = utc.floor("h")
            item = {"id": row.id, "journey_id": row.journey_id, "station_name": row.station_name,
                    "weather_reference_time_local": row.weather_reference_time_local,
                    "interpretation": "CEST" if summer_time else "CET",
                    "weather_reference_time_utc_candidate": utc, "weather_hour_utc_candidate": hour}
            for product, spec in PRODUCTS.items():
                station_id = getattr(row, product+"_dwd_station_id")
                match = observations[observations["product"].eq(product) & observations.dwd_station_id.eq(station_id) & observations.weather_hour_utc.eq(hour)]
                item[product+"_dwd_station_id"] = station_id
                item[spec[-1]] = match.iloc[0][spec[-1]] if len(match) else np.nan
                item[product+"_source_file"] = match.iloc[0].source_file if len(match) else None
            ambiguous_candidates.append(item)
    pd.DataFrame(ambiguous_candidates).to_csv(DATA / "ambiguous_time_weather_candidates.csv", index=False)
    s.to_parquet(DATA / "journey_stops_weather.parquet", index=False, compression="zstd")
    s.to_csv(DATA / "journey_stops_weather.csv.gz", index=False, compression="gzip")
    used = []
    for product in PRODUCTS:
        columns = ["station_name", "eva", product+"_dwd_station_id", product+"_dwd_station_name", product+"_distance_km", product+"_station_rank"]
        table = s.groupby(columns, dropna=False).size().rename("stop_events").reset_index()
        table.columns = ["station_name", "eva", "dwd_station_id", "dwd_station_name", "distance_km", "station_rank", "stop_events"]
        table.insert(0, "product", product)
        used.append(table)
    pd.concat(used, ignore_index=True).to_csv(DATA / "dwd_station_usage.csv", index=False)
    stats = json.loads((ROOT / "weather_audit.json").read_text())
    stats.update({"gap_policy": "nearest available alternative station at the same UTC hour; original nearest-only data retained",
                  "complete_weather_stops": int(s[[x[-1] for x in PRODUCTS.values()]].notna().all(axis=1).sum()),
                  "missing_by_variable": {spec[-1]: int(s[spec[-1]].isna().sum()) for spec in PRODUCTS.values()},
                  "fallback_stops_by_variable": {p: int(s[p+"_used_fallback_station"].sum()) for p in PRODUCTS},
                  "max_station_distance_km": {p: float(s[p+"_distance_km"].max()) for p in PRODUCTS},
                  "max_station_rank": {p: int(s[p+"_station_rank"].max()) for p in PRODUCTS},
                  "max_distance_limit_km": MAX_DISTANCE_KM,
                  "beyond_distance_limit_by_variable": {p: int(s[p+"_beyond_distance_limit"].sum()) for p in PRODUCTS},
                  "ambiguous_time_candidate_rows": len(ambiguous_candidates)})
    (ROOT / "weather_audit.json").write_text(json.dumps(stats, indent=2))
    manifest = {p.name.removesuffix(".source.json"): json.loads(p.read_text()) for p in sorted(CACHE.glob("*.source.json"))}
    (ROOT / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
