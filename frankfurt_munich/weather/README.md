# Frankfurt–München weather data

Updated: 2026-10-16

[Station mapping and original sources](railway_to_dwd_station_mapping.md) · [Temperature notebook](01_temperature_read_raw.ipynb) · [Temperature report](quality/temperature/temperature_report.md)

## Scope and status

- Railway scope: 17,507 CV journeys and 4,324 holdout journeys; 21,831 distinct journeys and 19 recorded railway stations. The original CV/holdout split is preserved.
- Mapping: 114 railway-station/variable pairs, each with a primary and backup DWD station within 50 km.
- Temperature: 40 ZIP archives for 20 DWD stations processed. All 28 recorded review decisions are applied: 22 railway station-hours retain the primary and 6 use the same-hour backup.
- Precipitation, precipitation type, mean wind, gusts and daily snow depth remain pending. Comprehensive instrument/site review, observation publication-time availability and prediction-time journey matching remain separate tasks.

## Temperature workflow

The notebook runs as one sequence and ends with one consolidated report:

1. Setup and rules.
2. Read and standardize DWD observations.
3. Match primary/backup stations and screen.
4. Apply recorded review decisions.
5. Validate and save final outputs.
6. Report.

Historical records take priority over recent records within each DWD station. Missing markers and unavailable values remain traceable. At each railway station-hour, the primary is selected first; missing primary observations use the mapped backup at the same UTC hour. Numeric review candidates are retained until a recorded source decision is applied. No interpolation or zero filling is used.

The project reviewer selected backup observations for G10, G15, G17, G20 and G21 at their flagged hours; the other 16 groups retain the primary. G10 covers two railway stations, giving six substitutions. These choices do not independently confirm that primary observations are measurement errors.

## Current integration input

[observations/railway_temperature_reviewed.parquet](observations/railway_temperature_reviewed.parquet) is the current hourly railway-station temperature table. Its key is `railway_eva_id` plus `timestamp_utc`; it is not yet a journey-feature table.

The study interval is 2024-07-01 00:00 UTC to 2026-10-01 00:00 UTC (exclusive). Of 374,832 railway station-hours, 372,122 use the primary and 2,710 use the backup, including the six manual substitutions. Remaining missing values: zero for these input versions. The file additionally contains 912 preceding buffer rows, giving 375,744 rows in total.

Twenty-two retained records still carry diagnostic flags and an applied review status. No new or unreviewed flags remain after the recorded decisions. `diagnostic_flagged` retains screening evidence; `review_required` identifies outstanding decisions. These are retrospective audit fields, not model features. Source IDs, raw values, quality codes, archive hashes, `initial_temperature_c` and manual-review fields support traceability.

## Report and audit files

The [temperature report](quality/temperature/temperature_report.md) summarizes source checks, initial screening, applied decisions, final checks and remaining scope. Detailed files support that report:

- [Applied review decisions](quality/temperature/railway_reviewed/review_decisions.csv), [retained diagnostics](quality/temperature/railway_reviewed/retained_diagnostics.csv), and [outstanding candidates](quality/temperature/railway_reviewed/new_review_candidates.csv).
- [Final station summary](quality/temperature/railway_reviewed/station_summary.csv) and [final monthly summary](quality/temperature/railway_reviewed/monthly_summary.csv).
- [Original review locations and context charts](quality/temperature/railway_initial/review_candidates.md) and [recorded decision registry](quality/temperature/railway_initial/manual_review_decisions.json).
- [Workflow versions, rules and hashes](quality/temperature/temperature_workflow_manifest.json).

The DWD source table, [temperature.parquet](observations/temperature.parquet), stores the 20 weather stations separately. The [initial railway table](observations/railway_temperature_initial.parquet) preserves the pre-review selection. Their counts differ from the reviewed railway table because source observations may serve multiple railway stations and the railway tables apply primary/backup selection. Station IDs are strings; CSV readers must preserve leading zeros.

## Run the notebook

Requirements: Python 3, pandas, numpy, matplotlib and pyarrow.

1. Keep the notebook beside the matching `station_mapping.json` and `source_manifest.json`.
2. Place the 40 original temperature ZIPs in `raw/air_temperature/`, or configure `ARCHIVE_DIR` in the setup cell.
3. Keep `quality/temperature/railway_initial/manual_review_decisions.json` with the project.
4. Run all six code cells in order. Outputs are written under `observations/` and `quality/temperature/`.

The complete local execution package supplies these inputs together. DWD recent archives can change after download; reproducing the recorded decisions requires the versions identified by the saved hashes. The 48-hour buffer supports local departure-time alignment and lookbacks up to 24 hours; longer lookbacks require a larger extraction window.

## Reproduce mapping coverage

The six mapping-support files remain together in this directory. From this directory, run:

```bash
python3 verify_mapping.py
```

This Python-standard-library script checks saved mapping-coverage flags, identifiers and distances. It does not repeat raw-observation processing or candidate-station selection. Original sources and the mapping method are documented in the linked station-mapping file.
