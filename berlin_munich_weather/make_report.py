"""Write README.md from the verified output counts.

README.md is generated: edit the template below, not README.md itself.
"""
from pathlib import Path
import json
import pandas as pd

from fetch_weather import DB_STATIONS_VERSION

ROOT = Path(__file__).resolve().parent
COVERAGE_CHANGE = "2025-11-03"


def main():
    a = json.loads((ROOT / "extraction_audit.json").read_text())
    w = json.loads((ROOT / "weather_audit.json").read_text())
    v = json.loads((ROOT / "verification.json").read_text())
    excluded_wind = json.loads((ROOT / "dwd_excluded_stations.json").read_text()).get("wind", {})
    journeys = pd.read_parquet(ROOT / "data/journeys.parquet")
    ambiguous_file = ROOT / "data/ambiguous_time_weather_candidates.csv"  # empty when there are none
    try:
        ambiguous = pd.read_csv(ambiguous_file)
    except pd.errors.EmptyDataError:
        ambiguous = pd.DataFrame()
    example = pd.read_csv(ROOT / "data/example_journey.csv")
    example_journey = journeys.set_index("journey_id").loc[example.iloc[0].journey_id]

    station_names = len({row["station_name"] for row in a["stations"]})
    first_day = pd.Timestamp(w["first_berlin_departure_local"]).strftime("%-d %B %Y")
    last_day = pd.Timestamp(w["last_berlin_departure_local"]).strftime("%-d %B %Y")
    fallback = w.get("fallback_stops_by_variable", {})
    distance = w["max_station_distance_km"]
    limit = w["max_distance_limit_km"]
    beyond = sum(w["beyond_distance_limit_by_variable"].values())
    missing_weather = w["stops"] - w["complete_weather_stops"]
    excluded_wind_names = ", ".join(reason.split(":")[0] for reason in excluded_wind.values())
    ic_note = ("no IC trains matched" if a["IC_journeys"] == 0
               else f"plus {a['IC_journeys']:,} IC runs")
    after_change = journeys.berlin_departure_planned_local.ge(COVERAGE_CHANGE)
    stops_before = journeys.loc[~after_change, "observed_stops"].mean()
    stops_after = journeys.loc[after_change, "observed_stops"].mean()
    classes = journeys.severity_class.value_counts()
    class_share = journeys.severity_class.value_counts(normalize=True) * 100

    if missing_weather:
        coverage_text = f"{w['complete_weather_stops']:,} stops have all three variables; {missing_weather:,} do not"
    else:
        coverage_text = "Every stop has all three variables"
    if beyond:
        limit_text = (f"**{beyond:,} values are missing because no station within {limit:.0f} km had one.** "
                      "They are flagged in the `_beyond_distance_limit` columns.")
    else:
        limit_text = f"No value needed a station further than {limit:.0f} km, so none are flagged."
    if len(ambiguous):
        ambiguous_text = (f"{ambiguous.id.nunique()} stops belong to journeys that were scheduled to leave Berlin in the "
                          "repeated hour when clocks went back. Their UTC time cannot be determined, so their weather "
                          "is left empty. Both possible matches are in `ambiguous_time_weather_candidates.csv`.")
    else:
        ambiguous_text = ("No journey was scheduled to leave Berlin in the repeated hour when clocks go back, "
                          "so every departure has a single UTC time. The script would record any such case in "
                          "`ambiguous_time_weather_candidates.csv`, which is currently empty.")

    lines = ["| Stop | Event | Scheduled | Reported | Temperature °C | Precipitation mm | Wind m/s |",
             "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in example.itertuples(index=False):
        lines.append(f"| {row.station_name} | {row.event_kind} | {row.event_planned_time[11:16]} | {row.event_reported_time[11:16]} "
                     f"| {row.temperature_c:.1f} | {row.precipitation_mm:.1f} | {row.wind_speed_ms:.1f} |")
    sample_table = "\n".join(lines)
    reference_local = pd.Timestamp(example.iloc[0].weather_reference_time_local)
    reference_hour = pd.Timestamp(example.iloc[0].weather_hour_utc)
    reference_berlin = reference_local.tz_localize("Europe/Berlin")
    season = "summer" if reference_berlin.dst() else "winter"

    text = f"""# Berlin → München ICE journeys with weather

This folder holds every ICE journey from **Berlin Hauptbahnhof to München Hbf** in our railway data, with scheduled Berlin departures from {first_day} to {last_day}. For each journey it gives:

- the **prediction targets**: München arrival delay, whether the journey was cancelled, and the disruption severity class from the project instructions;
- the **weather along the route when the train is due to leave Berlin**: temperature, precipitation and wind observed by the German Weather Service (DWD) at each stop, including München Hbf, all for the same hour as the scheduled Berlin departure.

It is the starting dataset for the project's core task: predicting delay and disruption for one selected origin–destination pair, with and without weather.

## At a glance

| | |
| --- | --- |
| Period | Scheduled Berlin departures from {first_day} to {last_day} ({len(a['source_files'])} monthly railway files) |
| Journeys | **{a['journeys']:,}** dated ICE runs ({ic_note}) |
| Stops | **{a['observed_stops']:,}** rows, one per recorded stop from Berlin to München, at {station_names} stable-panel stations |
| Weather coverage | {coverage_text} |
| Normal / minor | {classes.get('normal_minor', 0):,} journeys ({class_share.get('normal_minor', 0):.1f}%) |
| Moderate | {classes.get('moderate', 0):,} journeys ({class_share.get('moderate', 0):.1f}%) |
| Severe | {classes.get('severe', 0):,} journeys ({class_share.get('severe', 0):.1f}%) |
| Canceled | {classes.get('canceled', 0):,} journeys ({class_share.get('canceled', 0):.1f}%) |

## Quick start

Run from the project directory:

```python
import pandas as pd

journeys = pd.read_parquet("berlin_munich_weather/data/journeys.parquet")
stops = pd.read_parquet("berlin_munich_weather/data/journey_stops_weather.parquet")

# Targets and München weather at the scheduled Berlin departure, one row per journey
journeys[["journey_id", "munich_arrival_delay_min", "severity_class", "severe_or_canceled",
          "munich_temperature_c", "munich_precipitation_mm", "munich_wind_speed_ms"]]

# All stops of one journey, in order
journey_id = journeys.iloc[0]["journey_id"]
journey = stops.loc[stops["journey_id"].eq(journey_id)].sort_values("stop_index")
```

## Key terms

- **Journey**: one train on one date, from Berlin Hbf to München Hbf. ICE 1103 on 11 June and ICE 1103 on 12 June are two journeys. Identified by `journey_id`.
- **Stop**: one row in the stop-level dataset, for one station where the journey stopped between Berlin and München, including both ends. `stop_index` is 0 at Berlin and counts up.
- **Weather hour**: the one DWD hour used for the whole journey. It is the last full UTC hour at or before the **scheduled** Berlin departure (see [How weather is matched](#how-weather-is-matched)).
- **Stable panel**: the stations in `data/stations_present_in_all_months.csv`, which appear in every monthly railway file. Only stops at these stations are kept.

## Targets

These columns are in `journeys.parquet` and follow the working definition in the project instructions.

| Column | Meaning |
| --- | --- |
| `munich_arrival_delay_min` | Reported minus scheduled arrival at München Hbf, in minutes. Negative means early. Empty for canceled journeys. |
| `journey_canceled` | `True` if the Berlin departure **or** the München arrival is canceled. Skipped intermediate stops do not count. |
| `severity_class` | `normal_minor` (delay below 20 min), `moderate` (20 to below 60), `severe` (60 or more), or `canceled`. |
| `severe_or_canceled` | The initial binary target: `True` for `severe` or `canceled`. |

The arrival delay is computed from `arrival_change_time − arrival_planned_time`. The source's `delay_in_min` column cannot be used for this: when the train continues beyond München, it holds the *departure* delay there.

## Files

Most work needs only the first two files.

| File | What it is |
| --- | --- |
| [data/journeys.parquet](data/journeys.parquet) | **One row per journey**: scheduled and reported Berlin departure and München arrival, the targets, München weather at the scheduled Berlin departure, the list of stops and coverage counts. Also saved as `journeys.csv`. |
| [data/journey_stops_weather.parquet](data/journey_stops_weather.parquet) | **One row per stop**, with all original railway columns plus the weather at that stop. Also saved as `journey_stops_weather.csv.gz`. |
| [data/example_journey.csv](data/example_journey.csv) | The example journey shown [below](#example-journey). |

<details>
<summary>Supporting and audit files</summary>

| File | What it is |
| --- | --- |
| `data/train_stops.parquet`, `data/train_journeys.csv` | Railway-only extract from step 1, before weather is added. |
| `data/train_missing_sequence_coverage.csv` | Journeys whose stop numbers have gaps, meaning the source did not record some stops. |
| `data/train_excluded_candidates.csv` | {a['excluded_reverse_endpoint_candidates']} candidate journeys dropped because München comes before Berlin in time. |
| `data/journey_stops_weather_nearest_only.parquet` | Stop-level data *before* gaps were filled from backup weather stations. |
| `data/dwd_hourly_observations.parquet` | All DWD hourly observations downloaded, by variable, station and UTC hour. |
| `data/dwd_station_mapping.csv` | The nearest (primary) weather station for each railway station and variable. |
| `data/dwd_station_usage.csv` | Every weather station actually used, including backups, with distance and number of stops. |
| `data/railway_station_coordinates.csv` | Coordinates used for each railway station. |
| `data/weather_coverage_by_month.csv`, `data/weather_coverage_by_station.csv` | Missing weather counts by month and by station. |
| `data/stops_with_missing_weather.csv` | Stops without complete weather ({missing_weather:,} at present). |
| `data/ambiguous_time_weather_candidates.csv` | Both possible weather matches for departures in the repeated clock-change hour ({len(ambiguous)} rows at present). |
| `dwd_excluded_stations.json` | Weather stations deliberately not used, with reasons. |
| `extraction_audit.json`, `weather_audit.json` | Detailed counts from each step. |
| `verification.json`, `dwd_time_reference_checks.json` | Results of the automated checks in step 4. |
| `download_manifest.json`, `train_source_manifest.json` | Where each input came from, when and its checksum or size. |
| `cache/` | Raw downloads (DWD ZIP archives, station lists, product PDFs), reused on reruns. |

</details>

## Main columns

### `journeys.parquet`

Besides the [targets](#targets):

| Column | Meaning |
| --- | --- |
| `berlin_departure_planned_local`, `berlin_departure_reported_local` | Scheduled and reported Berlin Hbf departure, German local time. |
| `munich_arrival_planned_local`, `munich_arrival_reported_local` | Scheduled and reported München Hbf arrival, German local time. |
| `origin_departure_canceled`, `destination_arrival_canceled` | The two cancellation flags behind `journey_canceled`. |
| `weather_hour_utc` | The DWD hour used for this journey's weather. |
| `munich_temperature_c`, `munich_precipitation_mm`, `munich_wind_speed_ms` | München Hbf weather in that hour. |
| `observed_stops`, `recorded_route` | Number and names of the recorded stable-panel stops. |
| `unobserved_sequence_positions` | Stops the source did not record, counted from gaps in the stop numbers. |
| `any_selected_stop_canceled` | `True` if any stop, including intermediate ones, is canceled. |

### `journey_stops_weather.parquet`

The stop-level data keeps every column from the railway source and adds the columns below. Railway times are **German local time** (CET/CEST), as in the source.

**Journey and stop**

| Column | Meaning |
| --- | --- |
| `journey_id` | Dated train run; the key linking stops to `journeys.parquet`. |
| `id` | Source row id: `journey_id` plus the stop number. |
| `train_number`, `train_type` | For example `1103`, `ICE`. |
| `station_name`, `eva` | Station name and DB station number. |
| `stop_index` | 0 at Berlin, counting up. Gaps mean a stop was not recorded or is not in the stable panel. |
| `final_destination` | `True` on the München row. |

**Railway times and status (from the source)**

| Column | Meaning |
| --- | --- |
| `arrival_planned_time`, `departure_planned_time` | Timetable times. |
| `arrival_change_time`, `departure_change_time` | Reported times. If there was no update, the source copies the planned time here. |
| `event_kind`, `event_planned_time`, `event_reported_time` | The departure at Berlin and intermediate stops, or the arrival at München, with its scheduled and reported time. |
| `delay_in_min` | Source delay at this stop, departure delay where there is a departure. |
| `arrival_is_canceled`, `departure_is_canceled`, `selected_event_canceled` | Cancellation flags; the last one is for `event_kind`. |

**Weather time**

| Column | Meaning |
| --- | --- |
| `weather_reference_time_local` | The journey's scheduled Berlin departure. Same for every stop of a journey. |
| `weather_reference_time_utc` | The same time in UTC. |
| `weather_hour_utc` | The DWD hour matched to it. |
| `weather_timestamp_lag_minutes` | Minutes between that hour and the scheduled departure (0–59). |
| `timezone_unresolved` | `True` if the departure falls in the repeated clock-change hour. |

**Weather**

| Column | Meaning | Units | DWD field |
| --- | --- | --- | --- |
| `temperature_c` | Air temperature | °C | `TT_TU` |
| `precipitation_mm` | Precipitation in the hour | mm | `R1` |
| `wind_speed_ms` | Mean wind speed | m/s | `F` |

**Where each weather value came from**: for each variable, prefixed `temperature_`, `precipitation_` or `wind_`:

| Column suffix | Meaning |
| --- | --- |
| `dwd_station_id`, `dwd_station_name`, `distance_km` | The weather station that supplied the value, and its straight-line distance. |
| `quality` | DWD quality level, kept as-is and not filtered. |
| `used_fallback_station`, `station_rank` | `True` / rank > 1 if the nearest station had no value and a backup was used. |
| `primary_dwd_station_id`, `primary_missing` | The nearest station, and whether it had no value. |
| `source_file`, `archive_period` | The DWD ZIP file, and whether it was the `historical` (quality-checked) or `recent` archive. |
| `missing` | `True` if no value was found. |
| `beyond_distance_limit` | `True` if no station within {limit:.0f} km had a value for the hour. |

## How the dataset was built

Five scripts run in order. Each reads the previous step's output.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `extract_journeys.py` | Finds the journeys in the local monthly railway files and keeps their stops at stable-panel stations. |
| 2 | `fetch_weather.py` | Finds each journey's weather hour, picks the nearest weather station for each stop, downloads DWD data and joins it. |
| 3 | `fill_weather_gaps.py` | Fills missing hours from the next-nearest station within {limit:.0f} km. |
| 4 | `summarize.py` | Checks the result and writes the journey table with the targets, the coverage tables and the example. |
| 5 | `make_report.py` | Writes this README from the counts produced above. |

### How journeys are selected

- A journey is kept if the same dated train run stops at Berlin Hauptbahnhof and later at München Hbf. The dated run is the source `id` without its final stop number. The column `train_line_ride_id` is not enough on its own, because it repeats on different dates.
- Only stops at stable-panel stations are kept. Before {pd.Timestamp(COVERAGE_CHANGE):%-d %B %Y} the source recorded only about the 100 largest stations, and afterwards all of them. The panel keeps the set of stations the same over the whole period. It removed {a['stops_removed_outside_panel']:,} of {a['stops_before_panel_filter']:,} stops at {len(a['stations_removed_outside_panel'])} stations (listed in `extraction_audit.json`). Berlin Hbf and München Hbf are both in the panel, so no journey is lost.
- Routes still differ between journeys, and there is no filter on duration or route.
- {a['journeys_with_multiple_endpoint_pairs']} runs pass through Berlin or München more than once; the first Berlin→München pair is used.
- {a['excluded_reverse_endpoint_candidates']} malformed candidates, where the München time is before the Berlin time, are dropped and saved in `train_excluded_candidates.csv`.

### How weather is matched

1. **Which time.** The journey's **scheduled** departure from Berlin Hbf. Every stop of the journey, including München Hbf, uses this same time. The weather therefore shows conditions along the route when the train is due to leave. It does not depend on any delay.
2. **Which hour.** The scheduled departure is converted to UTC and rounded *down* to the full hour. For example, ICE {example_journey.train_number} is scheduled to leave Berlin at {reference_local:%H:%M} German {season} time, which is {reference_berlin.tz_convert('UTC'):%H:%M} UTC. It uses the DWD values labelled {reference_hour:%H:%M} UTC. The weather value can be up to 59 minutes older than the departure.
3. **Which station.** For each railway station and each variable separately, the nearest DWD station that was operating for the whole period is used. Temperature, precipitation and wind can therefore come from different stations. Distance is a straight line between coordinates.
4. **Missing hours.** If that station has no value for the hour, the next-nearest station *with a value for the same hour* is used, up to {limit:.0f} km away. Values are never interpolated or taken from another hour. Backup stations supplied {fallback.get('temperature', 0):,} temperature, {fallback.get('precipitation', 0):,} precipitation and {fallback.get('wind', 0):,} wind values. The furthest station used is {distance['temperature']:.1f} km away for temperature, {distance['precipitation']:.1f} km for precipitation and {distance['wind']:.1f} km for wind. {limit_text}
5. **Time-zone check.** Every DWD archive's metadata is checked to confirm that its timestamps are UTC. {len(excluded_wind)} wind stations ({excluded_wind_names}) report in MEZ (German winter time) instead and are excluded; see `dwd_excluded_stations.json`.

DWD's missing-value code `-999` becomes empty. Where DWD's quality-checked `historical` and preliminary `recent` archives overlap, `historical` is used. `recent` values can still be revised by DWD.

## Limitations

1. **Observed weather, not forecast.** The weather is the observation for the hour of the scheduled departure. It suits a prediction made at departure time. A prediction made earlier, for example the day before, would need a weather forecast instead. The dataset also doesn't account for how long DWD takes to publish an observation.
2. **One hour for the whole route.** Weather at later stops is what it was when the train was due to leave Berlin, not when the train reached that stop, up to several hours later.
3. **Stops per journey still grow slowly.** With the stable panel there is no jump at {pd.Timestamp(COVERAGE_CHANGE):%-d %B %Y}. Journeys still average {stops_before:.2f} recorded stops before that date and {stops_after:.2f} after it, because the route mix changes gradually, for example more runs via Frankfurt, Stuttgart and Augsburg. {a['journeys_missing_sequence']:,} journeys have gaps in their stop numbers ({a['missing_sequence_rows']:,} missing positions in total), mostly before {pd.Timestamp(COVERAGE_CHANGE):%-d %B %Y}.
4. **Clock change.** {ambiguous_text}
5. **Approximate weather.** Each value is an hourly station observation up to {max(distance.values()):.0f} km away, not a measurement at the station itself. DWD's metadata does not give the exact within-hour measurement window for every variable.

## Example journey

ICE {example_journey.train_number} on {pd.Timestamp(example_journey.berlin_departure_planned_local):%-d %B %Y}, journey `{example_journey.name}`. It arrived in München {example_journey.munich_arrival_delay_min:.0f} minutes late, so its severity class is `{example_journey.severity_class}`. All stops use the weather for {reference_hour:%H:%M} UTC. Weather station names and distances are in [data/example_journey.csv](data/example_journey.csv).

{sample_table}

## Reproduce

From the project directory:

```bash
.venv/bin/python berlin_munich_weather/extract_journeys.py
.venv/bin/python berlin_munich_weather/fetch_weather.py
.venv/bin/python berlin_munich_weather/fill_weather_gaps.py
.venv/bin/python berlin_munich_weather/summarize.py
.venv/bin/python berlin_munich_weather/make_report.py
```

- Step 1 reads `data/monthly_processed_data/` and `data/stations_present_in_all_months.csv`. These files and `eda.ipynb` are not modified.
- The first run of step 2 needs internet access. Downloads are cached in `cache/` and reused. The station-coordinate package is pinned to db-stations {DB_STATIONS_VERSION}.
- Step 4 stops with an error if any check fails. It checks that no rows are lost, each journey starts in Berlin and ends in München, every stop uses its journey's scheduled Berlin departure hour, every weather value matches its DWD observation and comes from within {limit:.0f} km, and spot checks against the raw DWD files agree. The results are in `verification.json`.
- This README is generated by `make_report.py`. Edit that script, not this file.

## Sources

- **Railway data:** [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data), based on the Deutsche Bahn Timetables API (CC BY 4.0). Its README and processing code are cached in `cache/`.
- **Station coordinates:** the [db-stations](https://github.com/derhuerst/db-stations) package, a redistribution of DB station data. When a station number is not listed, the station is matched by exact name and marked in `railway_station_coordinates.csv`.
- **Weather:** DWD Climate Data Center hourly station observations for [temperature](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/), [precipitation](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/) and [wind](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/wind/). The raw ZIP files and product descriptions are in `cache/`.
"""
    (ROOT / "README.md").write_text(text)
    print(ROOT / "README.md")


if __name__ == "__main__":
    main()
