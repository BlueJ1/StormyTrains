# Weather data quality report

Review date: 16 October 2026.

## Scope and checks

Five variables were prepared for 19 railway stations, covering 1 July 2024 to 30 September 2026. Precipitation type (WRTR) is not included.

Checks covered archive hashes, station IDs, units, time references, monthly coverage, consecutive gaps, duplicate/conflicting values, quality codes, extreme values, sudden changes, repeated values and primary/backup differences. Exported files passed unique-key, source-provenance and Parquet read-back checks.

## Cleaning

Missing markers (`-999`), blanks and non-numeric/non-finite values become unavailable, with raw tokens retained. Historical records take priority over recent overlaps. Use the mapped primary first, then the same-time backup when the primary is unavailable. Snow-depth backups use a compatible observation date/reference. All mapped stations are within 50 km. No interpolation or zero filling is used.

## Results and review

| Variable | Final backup positions | Remaining missing | Review outcome |
| --- | ---: | ---: | --- |
| Temperature | 2,710 hours | 0 | Of 28 flagged records, 22 retain primary and 6 use approved backup. |
| Precipitation | 1,837 hours | 0 | All 52 flagged records retain primary. |
| Mean wind | 1,872 hours | 0 | No railway study records flagged under the configured rules. |
| Gust | 1,630 hours | 0 | All 3 flagged records retain primary. |
| Snow depth | 9 days | 0 | The flagged 25 cm record and current mapping are retained. |

Counts are railway station-hours/days, excluding the buffer. Each hourly variable covers 374,832 study positions; snow depth covers 15,618. All five final tables have zero pending review. Diagnostic flags and decision logs remain available. Flags are screening evidence, not confirmed errors; shared DWD observations across railway stations are not independent measurements.

## Use and remaining scope

Use the five `_final.parquet` files in [observations/](observations/); `_initial` files preserve the first selection. Zero remaining missingness applies to the selected railway tables for these input versions, not every original DWD series.

Weather stations approximate railway conditions. Snow-depth mapping partly prioritizes coverage over proximity. Prediction-time matching, daily snow-depth lag and route features remain integration tasks; DWD publication-time availability and detailed instrument/site interpretation have not been verified. Manual review records project source choices, not independent measurement validation.

See the [README](README.md) for datasets, notebooks, sources and detailed reports.
