# Temperature data report

Status: **temperature observations processed and all 28 recorded review decisions applied**.

Inputs: 40 hash-verified DWD temperature ZIPs from 20 weather stations, mapped to 19 railway stations. Temperature units and UTC metadata were checked.

Before railway-level backup selection, the 20 DWD stations provide 391,898 of 394,560 source station-hours; 2,662 are missing. These source-level counts differ from the railway grid below.

Study interval: **2024-07-01 00:00 UTC to 2026-10-01 00:00 UTC (exclusive)**. The final table contains 375,744 rows, including 912 preceding buffer rows. Counts below use the study interval only.

## Results

| Check | Initial primary/backup selection | After recorded review |
|---|---:|---:|
| Railway station-hours | 374,832 | 374,832 |
| Primary selected | 372,128 | 372,122 |
| Backup selected | 2,704 | 2,710 |
| Remaining missing | 0 | 0 |
| Hourly-change flags (>10 °C) | 17 | 11 |
| Primary/backup-gap flags (>10 °C) | 12 | 12 |
| Distinct records with diagnostic flags | 28 | 22 |
| Outstanding review decisions | 28 | 0 |

Final retained temperature range: **-15.3 to 40.3 °C**. Duplicate final station-hour keys: 0. Values outside the screening range: 0; long constant-run flags: 0. The longest continuous backup interval in the initial selection was 849 hours; this describes primary-source availability, not a remaining temperature gap.

## Recorded review decisions

Primary observations were retained for 16 groups (22 railway station-hours). The following five groups use the same-hour backup (six railway station-hours). Overrides apply only to the listed hour; neighbouring values are unchanged.

| Group | Railway station | German local time | Primary → backup °C | DWD primary → backup |
|---|---|---|---:|---|
| G10 | München Hbf | 2026-05-31 15:00 CEST | 18.8 → 26.5 | 03379 → 07431 |
| G10 | München-Pasing | 2026-05-31 15:00 CEST | 18.8 → 26.5 | 03379 → 07431 |
| G15 | Stuttgart Hbf | 2026-07-16 21:00 CEST | 17.6 → 25.9 | 04928 → 06275 |
| G17 | Augsburg Hbf | 2026-07-31 18:00 CEST | 21.6 → 29.2 | 00232 → 02905 |
| G20 | Darmstadt Hbf | 2026-08-15 19:00 CEST | 24.9 → 31.9 | 00917 → 01424 |
| G21 | Ingolstadt Hbf | 2026-08-20 17:00 CEST | 15.2 → 20.0 | 02410 → 02700 |

The retained primary observations still carry their diagnostic flags and an applied review status. Project source choices do not independently confirm that an observation is physically correct or erroneous.

## Checks and remaining scope

Original ZIP hashes, required columns, station IDs, UTC metadata and temperature units were checked. Source-level monthly coverage, gaps, duplicate/conflicting observations and quality-code distributions were recorded. The final table passed unique-key, selected-source provenance and Parquet round-trip checks. Diagnostics were recomputed after review, including neighbouring hours; no new or unreviewed flags remain.

The review flags are retrospective diagnostics, not prediction-time model features. Instrument/site metadata has not been comprehensively reviewed, and observation publication latency has not been established. Journey matching must use the agreed prediction-time information cutoff. Other weather variables remain outside this temperature workflow.

## Files

- **Current integration input:** [railway_temperature_reviewed.parquet](../../observations/railway_temperature_reviewed.parquet).
- Source and initial tables: [DWD observations](../../observations/temperature.parquet) and [initial railway selection](../../observations/railway_temperature_initial.parquet).
- [All 28 applied decisions](railway_reviewed/review_decisions.csv), [retained diagnostics](railway_reviewed/retained_diagnostics.csv), [outstanding candidates](railway_reviewed/new_review_candidates.csv), [final station summary](railway_reviewed/station_summary.csv) and [final monthly summary](railway_reviewed/monthly_summary.csv).
- Source checks: [station summary](station_quality.csv), [monthly summary](monthly_quality.csv), [gaps](missing_gaps.csv), [quality codes](quality_codes_by_month.csv), [duplicates](duplicate_summary.csv) and [value conflicts](duplicate_value_conflicts.csv).
- [Original review locations and context charts](railway_initial/review_candidates.md), [decision registry](railway_initial/manual_review_decisions.json) and [processing manifest](temperature_workflow_manifest.json).

Station identifiers are strings; preserve leading zeros when reading CSV files. Raw inputs and the initial version are retained for traceability.
