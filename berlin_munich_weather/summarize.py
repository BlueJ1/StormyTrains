"""Verify the saved join and write compact journey, coverage and example tables.

The journey table carries the prediction targets: München arrival delay, the
cancellation flag and the severity class from the project instructions.
"""
from pathlib import Path
import json
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VARIABLES = {"temperature": "temperature_c", "precipitation": "precipitation_mm", "wind": "wind_speed_ms"}
MAX_DISTANCE_KM = 50.0


def severity(delay: pd.Series, canceled: pd.Series) -> pd.Series:
    """Normal/minor below 20 min, moderate 20 to below 60, severe 60 or more."""
    classes = pd.cut(delay, [-np.inf, 20, 60, np.inf], right=False,
                     labels=["normal_minor", "moderate", "severe"]).astype("string")
    return classes.mask(canceled, "canceled")


def main():
    s = pd.read_parquet(DATA / "journey_stops_weather.parquet")
    train = pd.read_parquet(DATA / "train_stops.parquet")
    assert s.id.is_unique and len(s) == len(train) and set(s.id) == set(train.id)
    berlin_departure = s[s.stop_index.eq(0)].set_index("journey_id").departure_planned_time
    assert s.weather_reference_time_local.equals(s.journey_id.map(berlin_departure))
    assert s.groupby("journey_id").weather_hour_utc.nunique().le(1).all()
    assert s.loc[s.final_destination, "event_kind"].eq("arrival").all()
    assert s.loc[~s.final_destination, "event_kind"].eq("departure").all()
    lag = s.weather_timestamp_lag_minutes.dropna()
    assert lag.ge(0).all() and lag.lt(60).all()
    assert s.weather_reference_time_utc.isna().equals(s.weather_hour_utc.isna())
    ordered = s.sort_values(["journey_id", "stop_index", "id"])
    first = ordered.groupby("journey_id", sort=False).head(1).set_index("journey_id")
    last = ordered.groupby("journey_id", sort=False).tail(1).set_index("journey_id")
    assert first.station_name.eq("Berlin Hauptbahnhof").all()
    assert last.station_name.eq("München Hbf").all() and last.final_destination.all()
    assert s.groupby("journey_id").final_destination.sum().eq(1).all()
    assert set(first.index) == set(last.index)
    assert first.event_planned_time.lt(last.event_planned_time).all()
    obs = pd.read_parquet(DATA / "dwd_hourly_observations.parquet")
    assert not obs.duplicated(["product", "dwd_station_id", "weather_hour_utc"]).any()
    raw_checks = []
    time_reference_checks = []
    codes = {"temperature": "TT_TU", "precipitation": "R1", "wind": "F"}
    for product, variable in VARIABLES.items():
        assert not s[variable].eq(-999).any()
        for filename in s[product+"_source_file"].dropna().unique():
            with zipfile.ZipFile(ROOT / "cache" / filename) as z:
                name = next(n for n in z.namelist() if "Metadaten_Parameter" in n and n.endswith(".txt"))
                metadata = pd.read_csv(z.open(name), sep=";", encoding="latin1", dtype=str)
            relevant = metadata[metadata.Parameter.eq(codes[product]) &
                                metadata.Bis_Datum.ge(s.weather_hour_utc.min().strftime("%Y%m%d")) &
                                metadata.Von_Datum.le(s.weather_hour_utc.max().strftime("%Y%m%d"))]
            assert len(relevant) and relevant["Zusatz-Info"].fillna("").str.contains("UTC").all()
            time_reference_checks.append({"source_file": filename, "product": product,
                                          "time_reference": relevant["Zusatz-Info"].unique().tolist()})
        lookup = obs[obs["product"].eq(product)].set_index(["dwd_station_id", "weather_hour_utc"])[variable]
        keys = pd.MultiIndex.from_arrays([s[product+"_dwd_station_id"], s.weather_hour_utc])
        assert np.isclose(s[variable].to_numpy(), lookup.reindex(keys).to_numpy(), equal_nan=True).all()
        assert s.loc[s[variable].notna(), product+"_distance_km"].le(MAX_DISTANCE_KM).all()
        if product+"_used_fallback_station" in s:
            fallback = s[product+"_used_fallback_station"]
            assert s.loc[fallback, product+"_primary_missing"].all()
            assert s.loc[fallback, product+"_station_rank"].gt(1).all()
            assert s.loc[fallback, product+"_dwd_station_id"].ne(s.loc[fallback, product+"_primary_dwd_station_id"]).all()
        for period in ("historical", "recent"):
            subset = s[s[product+"_archive_period"].eq(period) & s[variable].notna()]
            row = subset.iloc[len(subset)//2]
            filename = row[product+"_source_file"]
            with zipfile.ZipFile(ROOT / "cache" / filename) as z:
                name = next(n for n in z.namelist() if n.startswith("produkt_") and n.endswith(".txt"))
                raw = pd.read_csv(z.open(name), sep=";", dtype=str)
            raw.columns = raw.columns.str.strip()
            match = raw[raw.MESS_DATUM.str.strip().eq(row.weather_hour_utc.strftime("%Y%m%d%H"))]
            assert len(match) == 1
            assert np.isclose(float(match.iloc[0][codes[product]]), row[variable])
            raw_checks.append({"product": product, "period": period, "stop_id": row.id, "source_file": filename})
    complete = s[list(VARIABLES.values())].notna().all(axis=1)
    s["weather_complete"] = complete
    summary = pd.DataFrame({
        "train_type": first.train_type, "train_number": first.train_number,
        "berlin_departure_planned_local": first.departure_planned_time,
        "berlin_departure_reported_local": first.departure_change_time,
        "munich_arrival_planned_local": last.arrival_planned_time,
        "munich_arrival_reported_local": last.arrival_change_time,
        "origin_departure_canceled": first.departure_is_canceled,
        "destination_arrival_canceled": last.arrival_is_canceled,
    })
    groups = s.groupby("journey_id")
    summary["observed_stops"] = groups.size()
    summary["stops_with_complete_weather"] = groups.weather_complete.sum()
    summary["any_selected_stop_canceled"] = groups.selected_event_canceled.any()
    summary["recorded_route"] = ordered.groupby("journey_id").station_name.agg(" | ".join)
    gaps = pd.read_csv(DATA / "train_missing_sequence_coverage.csv").set_index("journey_id")
    summary["unobserved_sequence_positions"] = gaps.missing_sequence_count.reindex(summary.index).fillna(0).astype(int)
    # Targets. The source's delay_in_min is the departure delay when the train
    # continues beyond München, so arrival delay is computed from arrival times.
    summary["journey_canceled"] = (summary.origin_departure_canceled.fillna(False).astype(bool)
                                   | summary.destination_arrival_canceled.fillna(False).astype(bool))
    arrival_delay = (last.arrival_change_time - last.arrival_planned_time).dt.total_seconds() / 60
    summary["munich_arrival_delay_min"] = arrival_delay.mask(summary.journey_canceled)
    summary["severity_class"] = severity(summary.munich_arrival_delay_min, summary.journey_canceled)
    summary["severe_or_canceled"] = summary.severity_class.isin(["severe", "canceled"])
    assert summary.severity_class.notna().all()
    # München Hbf weather at the hour of the scheduled Berlin departure.
    summary["weather_hour_utc"] = last.weather_hour_utc
    for variable in VARIABLES.values():
        summary["munich_"+variable] = last[variable]
    summary = summary.sort_values("berlin_departure_planned_local").reset_index()
    summary.to_parquet(DATA / "journeys.parquet", index=False, compression="zstd")
    summary.to_csv(DATA / "journeys.csv", index=False)
    s["departure_month"] = s.journey_id.map(summary.set_index("journey_id").berlin_departure_planned_local.dt.strftime("%Y-%m"))
    for label, keys in [("month", ["departure_month"]), ("station", ["station_name", "eva"])]:
        coverage = s.groupby(keys).agg(stops=("id", "size"), journeys=("journey_id", "nunique"),
                                      complete_weather_stops=("weather_complete", "sum"))
        for variable in VARIABLES.values():
            coverage[variable+"_missing"] = s.groupby(keys)[variable].agg(lambda x: x.isna().sum())
        coverage.to_csv(DATA / f"weather_coverage_by_{label}.csv")
    s.loc[~complete].to_csv(DATA / "stops_with_missing_weather.csv", index=False)
    candidates = summary[summary.unobserved_sequence_positions.eq(0) & summary.stops_with_complete_weather.eq(summary.observed_stops) & ~summary.any_selected_stop_canceled]
    modal_route = candidates.recorded_route.value_counts().index[0]
    example = candidates[candidates.recorded_route.eq(modal_route)].iloc[len(candidates[candidates.recorded_route.eq(modal_route)])//2]
    columns = ["journey_id", "train_type", "train_number", "stop_index", "station_name", "event_kind", "event_planned_time", "event_reported_time",
               "weather_reference_time_local", "weather_hour_utc", *VARIABLES.values(),
               *[p+"_dwd_station_name" for p in VARIABLES], *[p+"_distance_km" for p in VARIABLES]]
    example_rows = ordered[ordered.journey_id.eq(example.journey_id)][columns]
    example_rows.to_csv(DATA / "example_journey.csv", index=False)
    result = {"passed": True, "journeys": len(summary), "stops": len(s),
              "complete_weather_stops": int(complete.sum()),
              "complete_weather_percent": round(100*complete.mean(), 4),
              "journeys_with_weather_for_every_observed_stop": int(summary.stops_with_complete_weather.eq(summary.observed_stops).sum()),
              "raw_archive_spot_checks": raw_checks, "archives_with_verified_UTC_metadata": len(time_reference_checks),
              "severity_class_counts": summary.severity_class.value_counts().to_dict(),
              "journeys_canceled": int(summary.journey_canceled.sum()),
              "example_journey_id": example.journey_id}
    (ROOT / "dwd_time_reference_checks.json").write_text(json.dumps(time_reference_checks, indent=2))
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(example_rows.to_string(index=False))


if __name__ == "__main__":
    main()
