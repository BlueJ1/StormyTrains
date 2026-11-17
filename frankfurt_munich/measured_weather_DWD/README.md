# Frankfurt–München weather data

Updated: 16 October 2026.

[Short quality report](QUALITY_REPORT.md) · [Station mapping and original sources](railway_to_dwd_station_mapping.md) · [Saved review decisions](quality/README.md)

## Data for integration

Five variables are available for 19 railway stations. Precipitation type (WRTR) is not included at this stage. Shared mapping entries and generic code for WRTR remain for reference.

Use **`_final.parquet`** for integration; **`_initial.parquet`** preserves the first primary/backup selection and screening state. There are ten railway-observation files in [observations/](observations/).

| Variable | Final dataset | Notebook |
| --- | --- | --- |
| Temperature | [temperature](observations/railway_temperature_final.parquet) | [Notebook](01_temperature_read_raw.ipynb) |
| Precipitation | [precipitation](observations/railway_precipitation_final.parquet) | [Notebook](02_precipitation.ipynb) |
| Mean wind | [mean wind](observations/railway_mean_wind_final.parquet) | [Notebook](04_mean_wind.ipynb) |
| Gust | [gust](observations/railway_gust_final.parquet) | [Notebook](05_gust.ipynb) |
| Snow depth | [snow depth](observations/railway_snow_depth_final.parquet) | [Notebook](06_snow_depth.ipynb) |

Hourly keys are `railway_eva_id` + `timestamp_utc`. Snow depth uses `railway_eva_id` + `observation_date`, with a 06 UTC measurement reference. Convert train times to UTC before matching and preserve station IDs as strings. Measurement time is not publication time.

The study period is 2024-07-01 to 2026-10-01 exclusive. Files include a preceding 48-hour buffer (two dates for snow depth): 375,744 rows per hourly variable and 15,656 snow-depth rows. Final study tables have zero remaining missing values and zero pending review under the recorded rules. These are station observations. [`../weather_features.py`](../weather_features.py) matches them to the prediction rows at each prediction time and builds the route features, including a snow flag from WRTR (see the [dataset README](../README.md#weather)). Retrospective quality flags are not model features.

## Quality and reproduction

The [short report](QUALITY_REPORT.md) describes checks, outcomes and limitations. The compact [quality folder](quality/README.md) contains four saved decision files. Detailed audit tables, charts and execution manifests are kept locally; the notebooks regenerate them when run with the original inputs. DWD-station intermediate tables are generated locally and are not among the ten delivered datasets.

Keep the notebooks beside the existing `station_mapping.json`, `source_manifest.json`, `weather_processing.py`, `precipitation_review.py`, `gust_review.py` and `snow_depth_review.py`. Requirements are listed in [requirements.txt](requirements.txt). Temperature has its own processing code in the notebook. Preserve the four decision files at their exact paths.

Original DWD ZIPs are not uploaded. Configure each notebook's archive directory: temperature defaults to `raw/air_temperature/`, the other four to `raw/dwd/`. Exact source URLs and hashes are recorded in `source_manifest.json` and checked before reading. Full local execution packages retain the archives and detailed audit results.

The shared `coverage_snapshot.json` and `verify_mapping.py` check saved mapping coverage, not the full observation-quality workflow. Station mappings are unchanged. DWD publication-time availability and detailed instrument/site interpretation have not been verified; see the short report for the limits of this delivery.
