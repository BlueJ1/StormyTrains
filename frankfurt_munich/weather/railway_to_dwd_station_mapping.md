# Railway-to-DWD station mapping

Updated: 2026-10-16

Status: **mapping and observation-coverage review**. 

**EVA ID identifies a railway station/stop place, not a journey. DWD station ID identifies a weather station.** Each weather cell is **primary / backup**, both five-digit DWD IDs. One railway station can use several weather stations.

The 19 railway stations are recorded across 21,831 journeys: 17,507 CV journeys and 4,324 holdout journeys. Primary coverage was reviewed on 2026-10-13; Bamberg was added on 2026-10-16. Backup coverage reviewed on 2026-10-16: **114/114 station-variable pairs have a backup passing the criteria below**. Joint primary-and-backup coverage is complete for **101 pairs**. **13 pairs still contain unusable observations**; variables and counts appear below. Counts are reported in the expandable coverage table below.

The current files use disjoint calendar weeks: 95 CV weeks (folds 1–5) and 23 holdout weeks, with no overlapping journey IDs, dates or weeks. Both splits span parts of the review period. CV is the development set for training and validation; holdout is reserved for final evaluation. The union defines weather-data scope and is not a merged model-training set.

Weather columns: temperature `TT_TU`, precipitation amount `R1`, precipitation type `WRTR`, mean wind `F`, gust `FX_911`, and daily snow depth `SHK_TAG`. Precipitation type supports identifying solid/mixed precipitation; snow depth is not snowfall amount.

## Origin and destination

| Railway station | EVA ID | Temp. | Precip. | Precip. type | Mean wind | Gust | Snow depth |
|---|---|---|---|---|---|---|---|
| Frankfurt (Main) Hbf **(origin)** | `8000105` | `01424` / `02480` | `01424` / `07341` | `07341` / `02601` | `07341` / `01420` | `07341` / `01420` | `01420` / `07341` |
| München Hbf **(destination)** | `8000261` | `03379` / `07431` | `03379` / `03722` | `03379` / `01262` | `03379` / `01262` | `03379` / `01262` | `03379` / `05404` |

## Intermediate stations towards Nürnberg and Ingolstadt

| Railway station | EVA ID | Temp. | Precip. | Precip. type | Mean wind | Gust | Snow depth |
|---|---|---|---|---|---|---|---|
| Frankfurt (Main) Süd | `8002041` | `01424` / `02480` | `01424` / `07341` | `07341` / `02601` | `07341` / `01420` | `07341` / `01420` | `01420` / `07341` |
| Hanau Hbf | `8000150` | `02480` / `01424` | `03136` / `07341` | `07341` / `02601` | `07341` / `01420` | `07341` / `01420` | `01420` / `07341` |
| Aschaffenburg Hbf | `8000010` | `02480` / `00917` | `01851` / `07412` | `07412` / `07341` | `07412` / `07341` | `07412` / `07341` | `03287` / `07412` |
| Würzburg Hbf | `8000260` | `05705` / `00191` | `05705` / `00130` | `05705` / `02597` | `05705` / `07412` | `05705` / `07412` | `05705` / `07412` |
| Fürth (Bay) Hbf | `8000114` | `03668` / `03667` | `03668` / `02860` | `03668` / `00282` | `03668` / `04280` | `03668` / `04280` | `03668` / `04280` |
| Nürnberg Hbf | `8000284` | `03668` / `03667` | `03668` / `04599` | `03668` / `00282` | `03668` / `04280` | `03668` / `04280` | `03668` / `04280` |
| Ingolstadt Hbf | `8000183` | `02410` / `02700` | `02409` / `02700` | `01587` / `05440` | `01587` / `05404` | `01587` / `05404` | `05404` / `01587` |

## Alternative branch via Bamberg

| Railway station | EVA ID | Temp. | Precip. | Precip. type | Mean wind | Gust | Snow depth |
|---|---|---|---|---|---|---|---|
| Bamberg | `8000025` | `00282` / `03668` | `00282` / `01279` | `00282` / `03668` | `00282` / `03668` | `00282` / `03668` | `00282` / `03668` |

## Intermediate stations towards Stuttgart and Augsburg

| Railway station | EVA ID | Temp. | Precip. | Precip. type | Mean wind | Gust | Snow depth |
|---|---|---|---|---|---|---|---|
| Mainz Hbf | `8000240` | `03137` / `01424` | `14007` / `01420` | `01420` / `02601` | `15976` / `01420` | `14007` / `01420` | `01420` / `07341` |
| Darmstadt Hbf | `8000068` | `00917` / `01424` | `00917` / `07341` | `01420` / `02601` | `01420` / `07341` | `01420` / `07341` | `01420` / `07341` |
| Mannheim Hbf | `8000244` | `05906` / `05275` | `05906` / `06317` | `05906` / `05426` | `05906` / `13674` | `05906` / `14009` | `05906` / `05426` |
| Heidelberg Hbf | `8000156` | `05906` / `05275` | `15514` / `03448` | `05906` / `05426` | `05906` / `13674` | `05906` / `13674` | `05906` / `05426` |
| Stuttgart Hbf | `8000096` | `04928` / `06275` | `04928` / `04931` | `04928` / `02485` | `04928` / `10510` | `04928` / `10510` | `04928` / `04931` |
| Plochingen | `8000302` | `06275` / `04928` | `00279` / `06275` | `04931` / `15444` | `10510` / `04928` | `10510` / `04928` | `02485` / `04931` |
| Ulm Hbf | `8000170` | `15444` / `06275` | `15444` / `05444` | `15444` / `04887` | `15444` / `01886` | `15444` / `01886` | `15444` / `04887` |
| Augsburg Hbf | `8000013` | `00232` / `02905` | `00232` / `02792` | `00232` / `02023` | `00232` / `01886` | `00232` / `01886` | `00232` / `02023` |
| München-Pasing | `8004158` | `03379` / `07431` | `03379` / `03722` | `03379` / `00232` | `03379` / `01262` | `03379` / `01262` | `03379` / `05404` |

Stations are arranged roughly from Frankfurt towards München within each group. Journeys follow different branches and do not stop at every station listed. Bamberg belongs to an alternative approach to Nürnberg. Mainz–Mannheim and Darmstadt–Heidelberg are different approaches to the Stuttgart corridor. The groupings are for reading this table; use each journey's applicable route when matching weather.

<details>
<summary>Backup checks and remaining gaps</summary>

- Observation review period: 2024-07-01 to 2026-09-30. Hourly variables have 19,728 expected hours; snow depth has 822 dates. Station scope covers both project CV and holdout files. Only identifiers, routes and scheduled dates are inspected for scope and split checks; delay labels and model scores are not used for station selection. CV and holdout weeks are interleaved, not consecutive time blocks.
- Backups must be different stations within 50 km, cover the review period in the station catalogue, provide at least 95% usable observations in matching units, have verified time metadata, and have no unresolved duplicate conflict or hard-rule violation. The 95% minimum is a provisional screening cutoff for this mapping review, not a DWD quality standard. Hard-rule checks flag negative precipitation/wind/snow depth and invalid precipitation-type codes; negative temperatures are allowed.
- For WRTR, only codes 0/6/7/8 count as usable here. Unknown precipitation type remains unknown. For daily snow depth, primary and backup observation-window metadata must agree.
- Previously reviewed stations are reused before further candidates are checked. For precipitation type and snow depth, multi-variable stations are prioritised. Among the tested candidates that pass the criteria, the backup covers the most primary gaps; ties are broken by distance. The search is not an exhaustive ranking of all possible stations.
- Recovery is checked at the same UTC hour, or the same date with compatible daily observation windows. Historical observations take priority over recent overlaps. No interpolation or substitution from another hour is used.
- For 27 mappings the primary is already complete: the backup has been checked as a standby, but there are no actual primary gaps on which to measure recovery.
- Status: station mapping and observation-coverage review completed. Detailed extreme-value checks, instrument/relocation review, quality-flag interpretation and publication-time availability checks are in progress. Original observations remain unchanged; recoverable coverage is calculated from availability flags.

Counts below are hours for hourly variables and days for SHK_TAG. Primary gaps include unusable/unknown WRTR codes. Remaining gaps stay missing; they are not zero.

| Railway station | Variable | Primary | Backup | Backup distance (km) | Backup usable | Primary gaps | Recoverable | Remaining | Joint coverage |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Frankfurt (Main) Hbf | TT_TU | `01424` | `02480` | 23.97 | 99.81% | 0 | 0 | 0 | 100.00% |
| Frankfurt (Main) Hbf | R1 | `01424` | `07341` | 8.94 | 99.28% | 0 | 0 | 0 | 100.00% |
| Frankfurt (Main) Hbf | WRTR | `07341` | `02601` | 20.03 | 98.85% | 106 | 106 | 0 | 100.00% |
| Frankfurt (Main) Hbf | F | `07341` | `01420` | 13.60 | 99.36% | 66 | 66 | 0 | 100.00% |
| Frankfurt (Main) Hbf | FX_911 | `07341` | `01420` | 13.60 | 99.35% | 67 | 67 | 0 | 100.00% |
| Frankfurt (Main) Hbf | SHK_TAG | `01420` | `07341` | 8.94 | 99.64% | 0 | 0 | 0 | 100.00% |
| München Hbf | TT_TU | `03379` | `07431` | 14.15 | 99.87% | 76 | 76 | 0 | 100.00% |
| München Hbf | R1 | `03379` | `03722` | 11.58 | 100.00% | 68 | 68 | 0 | 100.00% |
| München Hbf | WRTR | `03379` | `01262` | 29.82 | 99.01% | 143 | 126 | 17 | 99.91% |
| München Hbf | F | `03379` | `01262` | 29.82 | 98.99% | 76 | 76 | 0 | 100.00% |
| München Hbf | FX_911 | `03379` | `01262` | 29.82 | 98.98% | 77 | 77 | 0 | 100.00% |
| München Hbf | SHK_TAG | `03379` | `05404` | 30.85 | 100.00% | 0 | 0 | 0 | 100.00% |
| Frankfurt (Main) Süd | TT_TU | `01424` | `02480` | 22.22 | 99.81% | 0 | 0 | 0 | 100.00% |
| Frankfurt (Main) Süd | R1 | `01424` | `07341` | 7.19 | 99.28% | 0 | 0 | 0 | 100.00% |
| Frankfurt (Main) Süd | WRTR | `07341` | `02601` | 21.83 | 98.85% | 106 | 106 | 0 | 100.00% |
| Frankfurt (Main) Süd | F | `07341` | `01420` | 14.34 | 99.36% | 66 | 66 | 0 | 100.00% |
| Frankfurt (Main) Süd | FX_911 | `07341` | `01420` | 14.34 | 99.35% | 67 | 67 | 0 | 100.00% |
| Frankfurt (Main) Süd | SHK_TAG | `01420` | `07341` | 7.19 | 99.64% | 0 | 0 | 0 | 100.00% |
| Hanau Hbf | TT_TU | `02480` | `01424` | 18.52 | 100.00% | 38 | 38 | 0 | 100.00% |
| Hanau Hbf | R1 | `03136` | `07341` | 10.75 | 99.28% | 32 | 32 | 0 | 100.00% |
| Hanau Hbf | WRTR | `07341` | `02601` | 36.12 | 98.85% | 106 | 106 | 0 | 100.00% |
| Hanau Hbf | F | `07341` | `01420` | 30.96 | 99.36% | 66 | 66 | 0 | 100.00% |
| Hanau Hbf | FX_911 | `07341` | `01420` | 30.96 | 99.35% | 67 | 67 | 0 | 100.00% |
| Hanau Hbf | SHK_TAG | `01420` | `07341` | 10.75 | 99.64% | 0 | 0 | 0 | 100.00% |
| Aschaffenburg Hbf | TT_TU | `02480` | `00917` | 35.13 | 95.23% | 38 | 38 | 0 | 100.00% |
| Aschaffenburg Hbf | R1 | `01851` | `07412` | 20.26 | 99.25% | 7 | 7 | 0 | 100.00% |
| Aschaffenburg Hbf | WRTR | `07412` | `07341` | 28.29 | 99.46% | 389 | 389 | 0 | 100.00% |
| Aschaffenburg Hbf | F | `07412` | `07341` | 28.29 | 99.67% | 83 | 83 | 0 | 100.00% |
| Aschaffenburg Hbf | FX_911 | `07412` | `07341` | 28.29 | 99.66% | 88 | 88 | 0 | 100.00% |
| Aschaffenburg Hbf | SHK_TAG | `03287` | `07412` | 20.26 | 99.76% | 0 | 0 | 0 | 100.00% |
| Würzburg Hbf | TT_TU | `05705` | `00191` | 18.72 | 99.85% | 82 | 82 | 0 | 100.00% |
| Würzburg Hbf | R1 | `05705` | `00130` | 14.48 | 99.70% | 0 | 0 | 0 | 100.00% |
| Würzburg Hbf | WRTR | `05705` | `02597` | 48.05 | 98.58% | 84 | 68 | 16 | 99.92% |
| Würzburg Hbf | F | `05705` | `07412` | 43.26 | 99.58% | 117 | 117 | 0 | 100.00% |
| Würzburg Hbf | FX_911 | `05705` | `07412` | 43.26 | 99.55% | 119 | 119 | 0 | 100.00% |
| Würzburg Hbf | SHK_TAG | `05705` | `07412` | 43.26 | 99.76% | 1 | 1 | 0 | 100.00% |
| Fürth (Bay) Hbf | TT_TU | `03668` | `03667` | 19.69 | 99.81% | 0 | 0 | 0 | 100.00% |
| Fürth (Bay) Hbf | R1 | `03668` | `02860` | 15.15 | 98.90% | 9 | 9 | 0 | 100.00% |
| Fürth (Bay) Hbf | WRTR | `03668` | `00282` | 45.26 | 98.73% | 79 | 71 | 8 | 99.96% |
| Fürth (Bay) Hbf | F | `03668` | `04280` | 29.36 | 99.47% | 8 | 8 | 0 | 100.00% |
| Fürth (Bay) Hbf | FX_911 | `03668` | `04280` | 29.36 | 99.47% | 9 | 9 | 0 | 100.00% |
| Fürth (Bay) Hbf | SHK_TAG | `03668` | `04280` | 29.36 | 99.76% | 0 | 0 | 0 | 100.00% |
| Nürnberg Hbf | TT_TU | `03668` | `03667` | 12.55 | 99.81% | 0 | 0 | 0 | 100.00% |
| Nürnberg Hbf | R1 | `03668` | `04599` | 11.60 | 99.92% | 9 | 9 | 0 | 100.00% |
| Nürnberg Hbf | WRTR | `03668` | `00282` | 49.08 | 98.73% | 79 | 71 | 8 | 99.96% |
| Nürnberg Hbf | F | `03668` | `04280` | 25.55 | 99.47% | 8 | 8 | 0 | 100.00% |
| Nürnberg Hbf | FX_911 | `03668` | `04280` | 25.55 | 99.47% | 9 | 9 | 0 | 100.00% |
| Nürnberg Hbf | SHK_TAG | `03668` | `04280` | 25.55 | 99.76% | 0 | 0 | 0 | 100.00% |
| Ingolstadt Hbf | TT_TU | `02410` | `02700` | 10.20 | 100.00% | 782 | 782 | 0 | 100.00% |
| Ingolstadt Hbf | R1 | `02409` | `02700` | 10.20 | 99.99% | 0 | 0 | 0 | 100.00% |
| Ingolstadt Hbf | WRTR | `01587` | `05440` | 47.47 | 99.41% | 188 | 180 | 8 | 99.96% |
| Ingolstadt Hbf | F | `01587` | `05404` | 42.49 | 99.52% | 86 | 86 | 0 | 100.00% |
| Ingolstadt Hbf | FX_911 | `01587` | `05404` | 42.49 | 99.50% | 87 | 87 | 0 | 100.00% |
| Ingolstadt Hbf | SHK_TAG | `05404` | `01587` | 22.64 | 99.76% | 0 | 0 | 0 | 100.00% |
| Bamberg | TT_TU | `00282` | `03668` | 45.62 | 100.00% | 293 | 293 | 0 | 100.00% |
| Bamberg | R1 | `00282` | `01279` | 28.97 | 99.92% | 153 | 153 | 0 | 100.00% |
| Bamberg | WRTR | `00282` | `03668` | 45.62 | 99.60% | 251 | 243 | 8 | 99.96% |
| Bamberg | F | `00282` | `03668` | 45.62 | 99.96% | 330 | 330 | 0 | 100.00% |
| Bamberg | FX_911 | `00282` | `03668` | 45.62 | 99.95% | 336 | 336 | 0 | 100.00% |
| Bamberg | SHK_TAG | `00282` | `03668` | 45.62 | 100.00% | 5 | 5 | 0 | 100.00% |
| Mainz Hbf | TT_TU | `03137` | `01424` | 32.48 | 100.00% | 0 | 0 | 0 | 100.00% |
| Mainz Hbf | R1 | `14007` | `01420` | 18.96 | 99.73% | 5 | 5 | 0 | 100.00% |
| Mainz Hbf | WRTR | `01420` | `02601` | 27.97 | 98.85% | 220 | 220 | 0 | 100.00% |
| Mainz Hbf | F | `15976` | `01420` | 18.96 | 99.36% | 336 | 336 | 0 | 100.00% |
| Mainz Hbf | FX_911 | `14007` | `01420` | 18.96 | 99.35% | 61 | 61 | 0 | 100.00% |
| Mainz Hbf | SHK_TAG | `01420` | `07341` | 38.94 | 99.64% | 0 | 0 | 0 | 100.00% |
| Darmstadt Hbf | TT_TU | `00917` | `01424` | 28.43 | 100.00% | 941 | 941 | 0 | 100.00% |
| Darmstadt Hbf | R1 | `00917` | `07341` | 26.65 | 99.28% | 1061 | 1061 | 0 | 100.00% |
| Darmstadt Hbf | WRTR | `01420` | `02601` | 40.97 | 98.85% | 220 | 220 | 0 | 100.00% |
| Darmstadt Hbf | F | `01420` | `07341` | 26.65 | 99.67% | 127 | 127 | 0 | 100.00% |
| Darmstadt Hbf | FX_911 | `01420` | `07341` | 26.65 | 99.66% | 129 | 129 | 0 | 100.00% |
| Darmstadt Hbf | SHK_TAG | `01420` | `07341` | 26.65 | 99.64% | 0 | 0 | 0 | 100.00% |
| Mannheim Hbf | TT_TU | `05906` | `05275` | 26.58 | 99.86% | 127 | 127 | 0 | 100.00% |
| Mannheim Hbf | R1 | `05906` | `06317` | 12.19 | 99.96% | 161 | 161 | 0 | 100.00% |
| Mannheim Hbf | WRTR | `05906` | `05426` | 27.66 | 99.33% | 118 | 111 | 7 | 99.96% |
| Mannheim Hbf | F | `05906` | `13674` | 37.70 | 96.26% | 138 | 138 | 0 | 100.00% |
| Mannheim Hbf | FX_911 | `05906` | `14009` | 12.35 | 99.64% | 141 | 141 | 0 | 100.00% |
| Mannheim Hbf | SHK_TAG | `05906` | `05426` | 27.66 | 100.00% | 0 | 0 | 0 | 100.00% |
| Heidelberg Hbf | TT_TU | `05906` | `05275` | 20.32 | 99.86% | 127 | 127 | 0 | 100.00% |
| Heidelberg Hbf | R1 | `15514` | `03448` | 9.39 | 100.00% | 105 | 105 | 0 | 100.00% |
| Heidelberg Hbf | WRTR | `05906` | `05426` | 40.23 | 99.33% | 118 | 111 | 7 | 99.96% |
| Heidelberg Hbf | F | `05906` | `13674` | 20.61 | 96.26% | 138 | 138 | 0 | 100.00% |
| Heidelberg Hbf | FX_911 | `05906` | `13674` | 20.61 | 96.23% | 141 | 141 | 0 | 100.00% |
| Heidelberg Hbf | SHK_TAG | `05906` | `05426` | 40.23 | 100.00% | 0 | 0 | 0 | 100.00% |
| Stuttgart Hbf | TT_TU | `04928` | `06275` | 24.18 | 99.99% | 41 | 41 | 0 | 100.00% |
| Stuttgart Hbf | R1 | `04928` | `04931` | 11.08 | 99.97% | 13 | 13 | 0 | 100.00% |
| Stuttgart Hbf | WRTR | `04928` | `02485` | 39.83 | 98.90% | 87 | 83 | 4 | 99.98% |
| Stuttgart Hbf | F | `04928` | `10510` | 23.78 | 100.00% | 51 | 51 | 0 | 100.00% |
| Stuttgart Hbf | FX_911 | `04928` | `10510` | 23.78 | 100.00% | 53 | 53 | 0 | 100.00% |
| Stuttgart Hbf | SHK_TAG | `04928` | `04931` | 11.08 | 99.76% | 2 | 2 | 0 | 100.00% |
| Plochingen | TT_TU | `06275` | `04928` | 20.06 | 99.79% | 1 | 1 | 0 | 100.00% |
| Plochingen | R1 | `00279` | `06275` | 6.08 | 99.99% | 9 | 9 | 0 | 100.00% |
| Plochingen | WRTR | `04931` | `15444` | 48.19 | 98.93% | 87 | 82 | 5 | 99.97% |
| Plochingen | F | `10510` | `04928` | 20.06 | 99.74% | 0 | 0 | 0 | 100.00% |
| Plochingen | FX_911 | `10510` | `04928` | 20.06 | 99.73% | 0 | 0 | 0 | 100.00% |
| Plochingen | SHK_TAG | `02485` | `04931` | 14.02 | 99.76% | 0 | 0 | 0 | 100.00% |
| Ulm Hbf | TT_TU | `15444` | `06275` | 48.70 | 99.99% | 40 | 40 | 0 | 100.00% |
| Ulm Hbf | R1 | `15444` | `05444` | 19.28 | 99.81% | 137 | 137 | 0 | 100.00% |
| Ulm Hbf | WRTR | `15444` | `04887` | 30.83 | 98.55% | 211 | 189 | 22 | 99.89% |
| Ulm Hbf | F | `15444` | `01886` | 22.78 | 97.49% | 57 | 57 | 0 | 100.00% |
| Ulm Hbf | FX_911 | `15444` | `01886` | 22.78 | 97.49% | 58 | 58 | 0 | 100.00% |
| Ulm Hbf | SHK_TAG | `15444` | `04887` | 30.83 | 99.64% | 1 | 1 | 0 | 100.00% |
| Augsburg Hbf | TT_TU | `00232` | `02905` | 20.24 | 99.60% | 42 | 42 | 0 | 100.00% |
| Augsburg Hbf | R1 | `00232` | `02792` | 9.28 | 99.85% | 0 | 0 | 0 | 100.00% |
| Augsburg Hbf | WRTR | `00232` | `02023` | 49.21 | 98.95% | 127 | 96 | 31 | 99.84% |
| Augsburg Hbf | F | `00232` | `01886` | 48.07 | 97.49% | 43 | 43 | 0 | 100.00% |
| Augsburg Hbf | FX_911 | `00232` | `01886` | 48.07 | 97.49% | 44 | 44 | 0 | 100.00% |
| Augsburg Hbf | SHK_TAG | `00232` | `02023` | 49.21 | 99.76% | 0 | 0 | 0 | 100.00% |
| München-Pasing | TT_TU | `03379` | `07431` | 16.64 | 99.87% | 76 | 76 | 0 | 100.00% |
| München-Pasing | R1 | `03379` | `03722` | 12.48 | 100.00% | 68 | 68 | 0 | 100.00% |
| München-Pasing | WRTR | `03379` | `00232` | 49.19 | 99.36% | 143 | 129 | 14 | 99.93% |
| München-Pasing | F | `03379` | `01262` | 34.08 | 98.99% | 76 | 76 | 0 | 100.00% |
| München-Pasing | FX_911 | `03379` | `01262` | 34.08 | 98.98% | 77 | 77 | 0 | 100.00% |
| München-Pasing | SHK_TAG | `03379` | `05404` | 32.94 | 100.00% | 0 | 0 | 0 | 100.00% |

The accompanying mapping data and availability snapshot support these counts. Original DWD sources are listed below and in the source manifest.

</details>

<details>
<summary>DWD station names</summary>

| DWD station ID | Weather station |
|---|---|
| `00130` | Altertheim-Oberaltertheim |
| `00191` | Arnstein-Müdesheim |
| `00232` | Augsburg |
| `00279` | Baltmannsweiler-Hohengehren |
| `00282` | Bamberg |
| `00917` | Darmstadt |
| `01262` | München-Flughafen |
| `01279` | Möhrendorf-Kleinseebach |
| `01420` | Frankfurt/Main |
| `01424` | Frankfurt/Main-Westend |
| `01587` | Gelbelsee |
| `01851` | Großostheim |
| `01886` | Günzburg |
| `02023` | Harburg |
| `02409` | Ingolstadt (Flussmeisterstelle) |
| `02410` | Ingolstadt-Manching |
| `02480` | Kahl/Main |
| `02485` | Kaisersbach-Cronhütte |
| `02597` | Kissingen, Bad |
| `02601` | Kleiner Feldberg/Taunus |
| `02700` | Kösching |
| `02792` | Diedorf/Schwaben-Biburg |
| `02860` | Langenzenn |
| `02905` | Lechfeld |
| `03136` | Mühlheim/Main |
| `03137` | Mainz-Lerchenberg (ZDF) |
| `03287` | Michelstadt-Vielbrunn |
| `03379` | München-Stadt |
| `03448` | Neckargemünd-Kleingemünd |
| `03667` | Nürnberg-Netzstall |
| `03668` | Nürnberg |
| `03722` | Oberschleißheim |
| `04280` | Roth |
| `04599` | Wendelstein-Kleinschwarzenlohe |
| `04887` | Stötten |
| `04928` | Stuttgart (Schnarrenberg) |
| `04931` | Stuttgart-Echterdingen |
| `05275` | Waghäusel-Kirrlach |
| `05404` | Weihenstephan-Dürnast |
| `05426` | Weinbiet |
| `05440` | Weißenburg-Emetzheim |
| `05444` | Weißenhorn-Oberreichenbach |
| `05705` | Würzburg |
| `05906` | Mannheim |
| `06275` | Notzingen |
| `06317` | Dannstadt-Schauernheim |
| `07341` | Offenbach-Wetterpark |
| `07412` | Neuhütten/Spessart |
| `07431` | Oberhaching-Laufzorn |
| `10510` | Winterbach/Remstal |
| `13674` | Waibstadt |
| `14007` | Mainz-Marienborn |
| `14009` | Schifferstadt |
| `15444` | Ulm-Mähringen |
| `15514` | Heidelberg-Kurpfalzhof |
| `15976` | Ober-Olm/Bellem |

</details>

<details>
<summary>Railway station coordinates and sources</summary>

| EVA ID | Railway station | Latitude (°N) | Longitude (°E) |
|---|---|---:|---:|
| `8000105` | Frankfurt (Main) Hbf | 50.107145 | 8.663789 |
| `8000261` | München Hbf | 48.140232 | 11.558335 |
| `8002041` | Frankfurt (Main) Süd | 50.099365 | 8.686457 |
| `8000150` | Hanau Hbf | 50.120953 | 8.929 |
| `8000010` | Aschaffenburg Hbf | 49.980557 | 9.143697 |
| `8000260` | Würzburg Hbf | 49.801796 | 9.93578 |
| `8000114` | Fürth (Bay) Hbf | 49.469707 | 10.989988 |
| `8000284` | Nürnberg Hbf | 49.445616 | 11.082989 |
| `8000183` | Ingolstadt Hbf | 48.744541 | 11.437337 |
| `8000025` | Bamberg | 49.900759 | 10.899489 |
| `8000240` | Mainz Hbf | 50.001117 | 8.25872 |
| `8000068` | Darmstadt Hbf | 49.872503 | 8.629636 |
| `8000244` | Mannheim Hbf | 49.479354 | 8.468921 |
| `8000156` | Heidelberg Hbf | 49.403567 | 8.675442 |
| `8000096` | Stuttgart Hbf | 48.784084 | 9.181635 |
| `8000302` | Plochingen | 48.71308 | 9.41081 |
| `8000170` | Ulm Hbf | 48.399437 | 9.982227 |
| `8000013` | Augsburg Hbf | 48.365441 | 10.88557 |
| `8004158` | München-Pasing | 48.149852 | 11.461872 |

- Railway names: `recorded_route` in the project's `journeys_cv.parquet` and `journeys_holdout.parquet`. Only recorded stations are included.
- EVA IDs and coordinates: exact station-name matches in `railway_station_coordinates.csv`. All 19 selected names have one coordinate record, unique EVA IDs and non-missing coordinates within Germany's broad geographic bounds. Coordinates have not been independently checked against an external source.
- EVA IDs are seven-digit text identifiers; the padding zero is removed (`08000105` becomes `8000105`). DWD IDs retain all five digits (`01424`).
- [DB station documentation](https://developers.deutschebahn.com/db-api-marketplace/apis/product/ris-stations) uses EVA numbers to identify stop places. In this project, `journey_id` identifies an individual journey.
- Original weather sources: [DWD Climate Data Center (CDC) Open Data](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/). Station IDs, coordinates and operating periods come from the official station-description files for [air temperature](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/recent/TU_Stundenwerte_Beschreibung_Stationen.txt), [precipitation](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/recent/RR_Stundenwerte_Beschreibung_Stationen.txt), [mean wind](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/wind/recent/FF_Stundenwerte_Beschreibung_Stationen.txt), [gust](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/extreme_wind/recent/FX_Stundenwerte_Beschreibung_Stationen.txt), and [daily climate](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/recent/KL_Tageswerte_Beschreibung_Stationen.txt). Observations come from the `produkt_*.txt` files in the corresponding historical and recent ZIP archives. Weather stations are matched to the railway list and coordinates in the project's `journeys_cv.parquet` and `railway_station_coordinates.csv`.

</details>

## Supporting files and reproducibility

Keep these six files together in `frankfurt_munich/weather/`:

1. [README.md](README.md): current scope, deliverable status and the entry point to the weather work.
2. `railway_to_dwd_station_mapping.md`: station mapping, method, sources and remaining gaps.
3. [station_mapping.json](station_mapping.json): the 114 mappings, EVA/DWD IDs, station coordinates, distances and coverage results for use in code. Read station IDs as strings to preserve leading zeros.
4. [source_manifest.json](source_manifest.json): original DWD URLs, archive download times, file hashes and identification of the project railway inputs.
5. [coverage_snapshot.json](coverage_snapshot.json): a fixed snapshot of usable/missing observation positions. It contains availability flags, not weather measurement values.
6. [verify_mapping.py](verify_mapping.py): checks mapping consistency, distances and coverage counts using the saved snapshot. It uses the Python 3 standard library.

Run from this folder:

```bash
python3 verify_mapping.py
```

This reproduces the reported coverage counts without downloading the full observation archives. It does not rerun the raw-observation quality checks or candidate-station search. Original downloaded ZIPs can optionally be checked against their recorded hashes:

```bash
python3 verify_mapping.py --archives /path/to/original_dwd_zips
```

DWD recent archives are revised over time. The recorded download times and hashes identify the versions used in this review. Cleaned weather measurements and the full quality-check workflow will be documented as the next project deliverable.

The fixed hourly review starts at 2024-07-01 00:00 UTC. Early departures expressed in German local time and rolling lookback features need observations before this boundary. Extraction for cleaned observations must include that buffer; the coverage percentages above refer only to the stated review window.
