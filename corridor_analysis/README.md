# Corridor analysis: Berlin, Hamburg and Frankfurt to München, and Hamburg to Frankfurt

This is a railway-only analysis, with no weather, of ICE and IC journeys between July 2024 and August 2026 on four corridors:

| Corridor | Origin | Destination | Journeys |
|---|---|---|---|
| berlin | Berlin Hbf | München Hbf | 27,412 |
| hamburg | Hamburg Hbf | München Hbf | 26,526 |
| frankfurt | Frankfurt (Main) Hbf | München Hbf | 22,544 |
| hamburg_frankfurt | Hamburg Hbf | Frankfurt (Main) Hbf | 21,557 |

The journey definition, the panel filter and the targets are the same as in `berlin_munich_weather/`:

- **Journey:** a dated run that stops at the origin and later at the destination, with the same train at both ends.
- **Panel filter:** only stations in `data/stations_present_in_all_months.csv`.
- **Targets:**
  - `arrival_delay_min`
  - `journey_canceled`
  - `severity_class` (<20, 20–59, 60+ min, canceled)
  - `severe_or_canceled`

The destination columns have neutral names (`destination_arrival_*`, `arrival_delay_min`). With those names mapped back to its `munich_*` columns, the Berlin rows reproduce `berlin_munich_weather/data/journeys.parquet` exactly.

## Rebuild

```bash
.venv/bin/python corridor_analysis/extract.py      # about 30 s; writes data/journeys.parquet, data/stops.parquet, extraction_audit.json
.venv/bin/python corridor_analysis/analyse.py      # writes stats/<corridor>.json and stats/comparison.json
.venv/bin/python corridor_analysis/make_pages.py   # writes pages/hamburg.html, frankfurt.html, hamburg_frankfurt.html, comparison.html
```

`analyse.py` downloads the station coordinates (db-stations 5.0.2) into `cache/` if they are missing.

## Files

- `corridors.py`: the origins and destinations, the route-family rules and the en-route stations used for each corridor. The route families are our own grouping by the stations a train passes through; they are not an official DB classification.
- `extract.py`: a single pass over the monthly files that produces journeys and stops for all corridors, each carrying a `corridor` column. A run can belong to more than one corridor.
- The comparison page covers only the three München corridors (`COMPARED` in `corridors.py`).
- `analyse.py`: the figures behind the pages.
- `make_pages.py` and `templates/`: the HTML pages. Their text is filled with numbers from `stats/`.

The Berlin page from the earlier analysis is not rebuilt by this folder.
