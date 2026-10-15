# Frankfurt–München weather data

Updated: 2026-10-16

[Station mapping, coverage results and original sources](railway_to_dwd_station_mapping.md)

## Current scope

- Railway inputs: 17,507 CV journeys and 4,324 holdout journeys; 21,831 distinct journeys and 19 recorded railway stations in total.
- CV and holdout use separate, interleaved calendar weeks. The union defines the required weather scope; the original split remains unchanged.
- Each of the 114 station-variable pairs has a primary and a backup DWD station within 50 km. Station IDs are text identifiers.
- The mapping review includes Bamberg, which was absent from the earlier 18-station version.

## This week's deliverables

| Deliverable | Status |
|---|---|
| Station mapping and coverage review | Updated for the current CV and holdout inputs; remaining coverage gaps are documented. |
| Cleaned weather observations | In progress. The availability snapshot contains no measurement values. |
| Short weather-quality report | In progress. Coverage results are available; full observation-quality checks remain. |

## Reproduce coverage counts

Keep the six files in this directory together. From this directory, run:

```bash
python3 verify_mapping.py
```

The script uses the Python 3 standard library and checks saved coverage flags, mapping consistency and station distances. It does not rerun candidate selection or the full raw-observation audit. Data sources, version hashes and the remaining checks are described in the linked mapping document.

The fixed observation review covers UTC hours from 2024-07-01 through 2026-09-30. Weather extraction for early local-time departures and rolling lookback features will also require observations before that UTC boundary; the cleaned-observation stage must include this buffer.
