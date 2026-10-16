# Frankfurt–München weather data

Updated: 16 October 2026.

[Short quality report](QUALITY_REPORT.md) · [Railway-to-DWD station mapping and original sources](railway_to_dwd_station_mapping.md)

## Current data

Five variables are available for 19 railway stations: temperature, precipitation, mean wind, gust and daily snow depth. Precipitation type (WRTR) is not included at this stage. The shared mapping still contains its entries for reference.

Use **`_final.parquet`** for integration. **`_initial.parquet`** preserves the first primary/backup selection and original screening state. There are ten railway-observation files in [observations/](observations/).

| Variable | DWD field | Final dataset | Notebook | Detailed report |
| --- | --- | --- | --- | --- |
| Temperature | `TT_TU` | [temperature](observations/railway_temperature_final.parquet) | [Run](01_temperature_read_raw.ipynb) | [Report](quality/temperature/temperature_report.md) |
| Precipitation | `R1` | [precipitation](observations/railway_precipitation_final.parquet) | [Run](02_precipitation.ipynb) | [Report](quality/precipitation/precipitation_report.md) |
| Mean wind | `F` | [mean_wind](observations/railway_mean_wind_final.parquet) | [Run](04_mean_wind.ipynb) | [Report](quality/mean_wind/mean_wind_report.md) |
| Gust | `FX_911` | [gust](observations/railway_gust_final.parquet) | [Run](05_gust.ipynb) | [Report](quality/gust/gust_report.md) |
| Snow depth | `SHK_TAG` | [snow_depth](observations/railway_snow_depth_final.parquet) | [Run](06_snow_depth.ipynb) | [Report](quality/snow_depth/snow_depth_report.md) |

## Match to journeys

- Hourly tables: key = `railway_eva_id` + `timestamp_utc`. Convert railway times to UTC before matching.
- Snow depth: key = `railway_eva_id` + `observation_date`; `measurement_time_utc` records its 06 UTC reference. Measurement time is not publication time.
- IDs are strings. Preserve leading zeros in DWD IDs. These are station observations, so each journey still needs its applicable route and agreed prediction time to produce journey-level features. Keep retrospective quality/review fields out of prediction features.

Study period: 2024-07-01 to 2026-10-01 exclusive; exports include a preceding 48-hour buffer (two dates for snow depth). Each hourly final file has 375,744 rows including the buffer; snow depth has 15,656 rows. The final study tables have no remaining missing values or pending review under the current rules. See the short report for review outcomes and remaining scope.

## Run and reproduce

Keep the notebooks beside `station_mapping.json`, `source_manifest.json`, `weather_processing.py` and the three variable-specific review modules. Requirements are in [requirements.txt](requirements.txt). The temperature notebook contains its own processing code. All notebooks end with a consolidated report.

Keep the decision registries under `quality/`: temperature uses `temperature/railway_initial/manual_review_decisions.json`; precipitation, gust and snow depth use `<variable>/review_decisions.json`. These reproduce the recorded review decisions.

Original DWD ZIPs are excluded from GitHub. Supply the exact recorded archive versions and configure the archive directory in each notebook: temperature defaults to `raw/air_temperature/`; the other four default to `raw/dwd/`. URLs and hashes are recorded in `source_manifest.json`, the temperature workflow manifest, and each other variable's `archive_inventory.csv`. Notebooks may generate DWD-station intermediate tables locally; only the ten railway initial/final observation files are published here.

`quality/` retains monthly coverage, gaps, duplicate/conflict checks, quality codes, screening evidence, decisions and processing manifests. The [short report](QUALITY_REPORT.md) summarizes these results. The shared `coverage_snapshot.json` and `verify_mapping.py` check the earlier mapping-coverage snapshot; they do not replace the observation-quality workflow.

## Remaining integration work

Align observations with prediction times, define the daily snow-depth lag, construct route/rolling features and check publication-time availability. Detailed instrument/site interpretation remains separate. Screening and manual review do not establish independent measurement ground truth.
