# Frankfurt–München weather data

Updated: 2026-10-16.

[Station mapping and original sources](railway_to_dwd_station_mapping.md) · [Temperature notebook](01_temperature_read_raw.ipynb) · [Temperature report](quality/temperature/temperature_report.md)

## Scope and delivery status

Railway scope: 17,507 CV journeys and 4,324 holdout journeys; 21,831 distinct journeys and 19 recorded railway stations. The existing split is preserved. Shared station mappings, DWD source versions and mapping-support files remain in this folder.

Temperature was delivered previously. This update supplies **precipitation, mean wind, gust and daily snow depth**. Precipitation type (WRTR) is not part of this delivery; its notebook, observation tables and quality outputs are intentionally excluded. The shared mapping and generic processing code still contain its definition, which does not mean that its observations have been delivered.

## Four-variable quality and review results

| Variable | Frequency | Backup positions | Remaining missing | Initially flagged | Pending review |
| --- | --- | --- | --- | --- | --- |
| Precipitation | hourly | 1837 | 0 | 52 | 0 |
| Mean wind | hourly | 1872 | 0 | 0 | 0 |
| Gust | hourly | 1630 | 0 | 3 | 0 |
| Snow depth | daily | 9 | 0 | 1 | 0 |

Hourly counts are railway station-hours; daily counts are railway station-dates. Counts cover 2024-07-01 to 2026-10-01 exclusive. Exports additionally contain a 48-hour buffer, or two preceding dates for daily snow depth. Shared DWD observations at different railway stations are repeated mappings, not independent measurements.

- Precipitation: all 52 initially flagged records retain the primary. Sudden heavy rainfall, rapid changes, repeated small amounts and spatial differences alone do not trigger substitution.
- Mean wind: no railway study records trigger the configured screening rules; the validated initial table is the current integration input.
- Gust: all three flagged railway records retain their respective primary observations; values and sources are unchanged.
- Snow depth: the one flagged Plochingen record retains 25 cm from primary DWD 02485. The project reviewer accepted the current mapping after inspecting station distances and nearby observations. No nearest-available remapping has been applied. Daily dates and the 06 UTC measurement reference are preserved.

For reviewed variables, `review_required` is false and applied decisions remain traceable. Diagnostic flags remain visible; a completed source-choice review does not independently prove every measurement accurate. These diagnostics are retrospective audit fields, not model features.

## Notebooks, reports and integration inputs

- **Precipitation**: [notebook](02_precipitation.ipynb) · [report](quality/precipitation/precipitation_report.md) · [current integration table](observations/railway_precipitation_reviewed.parquet).
- **Mean wind**: [notebook](04_mean_wind.ipynb) · [report](quality/mean_wind/mean_wind_report.md) · [current integration table](observations/railway_mean_wind_initial.parquet).
- **Gust**: [notebook](05_gust.ipynb) · [report](quality/gust/gust_report.md) · [current integration table](observations/railway_gust_reviewed.parquet).
- **Snow depth**: [notebook](06_snow_depth.ipynb) · [report](quality/snow_depth/snow_depth_report.md) · [current integration table](observations/railway_snow_depth_reviewed.parquet).

The notebooks use six sections, ending with one consolidated report. Source observations are saved as `observations/<variable>.parquet`; railway initial tables preserve the original screening state. Reviewed railway tables include the recorded decisions. Keys are railway EVA ID plus UTC hour, or observation date for daily snow depth. These are station-observation tables, not journey-feature tables.

The current temperature integration input remains [railway_temperature_reviewed.parquet](observations/railway_temperature_reviewed.parquet), delivered previously. This update does not replace temperature files.

## Processing and reproducibility

Primary observations are selected first. If a primary is unavailable, use the mapped backup at the same UTC hour, or the same compatible daily reference date for snow depth. `-999`, blanks and non-numeric/non-finite tokens become unavailable while raw tokens remain traceable. Historical rows take priority over recent overlaps, including missing historical rows; same-priority conflicting values remain unavailable pending review. No interpolation or zero filling is used.

Each variable's `quality/` folder contains monthly station/variable coverage, continuous gaps, duplicate/conflict diagnostics, quality-code distributions, screening records, context, validation checks and input/output hashes. Precipitation, gust and snow depth also include saved decision registries and applied decision logs. Screening thresholds are project review rules, not DWD error/deletion standards.

Requirements: Python 3, pandas, numpy, matplotlib and pyarrow; notebook dependencies are listed in `requirements.txt`. Keep the notebooks beside `weather_processing.py`, `precipitation_review.py`, `gust_review.py`, `snow_depth_review.py`, the existing `station_mapping.json` and `source_manifest.json`. Preserve each reviewed variable's `quality/<variable>/review_decisions.json`.

Place the exact original DWD ZIP versions in `raw/dwd/`, or set `ARCHIVE_DIR` in the notebook setup cell. Required URLs, original download times and SHA-256 hashes are listed in each variable's `archive_inventory.csv`. Raw ZIPs are excluded from this GitHub upload. The previously supplied local execution package contains the original archives; use this update's notebooks/modules and review registries with those archives to rerun the latest decisions.

## Remaining project work

Precipitation-type observations are not delivered in this update. Detailed instrument/site interpretation, observation publication-time availability and prediction-time journey integration remain separate tasks. Measurement reference times do not establish when observations became available for prediction.

The existing mapping files describe 114 station-variable pairs. Keep those shared files unchanged; the present snow-depth decision retains their mapping. `python3 verify_mapping.py` checks the saved mapping-coverage snapshot and does not rerun this observation-processing workflow.
