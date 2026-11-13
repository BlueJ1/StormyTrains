# StormyTrains GFS weather data

The files in `data/` cover weather valid from 2025-09-30 through 2026-09-30, inclusive, in `Europe/Berlin`. There are 13 date partitions and 1,147,776 rows in each of the raw and processed datasets. The weather tables are provided for joining to railway schedules. It keeps every six-hour NOAA GFS cycle and hourly lead from 0 to 48 hours that falls in this window, including cycles initialized before the first day. The 48-hour limit is a project choice for the planned 24-hour prediction horizon and a buffer for the weather forecast's age and the train journey.

`gfs_stations.csv` maps the project's 19 Frankfurt–München railway stations to 16 nearest 0.25° GFS cells. Coordinates come from the existing Frankfurt–München weather station mapping. The Parquet tables contain one row per grid point, initialization time and lead hour. Both time columns are UTC. A later stage can use the station map and select a forecast that was available by each train prediction time. The GFS initialization time records when the run starts. The archive extract has no historical publication timestamp. Apply and document an availability rule when matching a run to a train prediction cutoff.

| File | Contents |
|---|---|
| `gfs_raw_*.parquet` | Archive values for 2 m temperature (°C), precipitation accumulation (kg/m²), 10 m wind components and surface gust (m/s), and snow depth (m). |
| `gfs_weather_*.parquet` | Temperature (°C), precipitation in the preceding hour (mm), 10 m wind speed and surface gust (m/s), and snow depth (cm). |
| `gfs_manifest_*.json` | Date bounds, grid points, row counts, missing-value counts, file sizes and SHA-256 hashes. |

GFS precipitation accumulates over the preceding 1–6 hours. The export differences consecutive leads within each six-hour accumulation period and uses the source amount directly at the first hour of each period. Precipitation at lead zero and negative hourly differences are missing. The raw table preserves the original archive values.

## Run

Use Python 3.12 or newer. Keep the script and `gfs_stations.csv` together.

```sh
python -m venv .venv-gfs
source .venv-gfs/bin/activate
python -m pip install 'dynamical-catalog==1.1.1' pyarrow
ZARR_ASYNC__CONCURRENCY=16 python download_gfs_weather.py --start 2025-09-30 --end 2025-10-01 --output gfs_weather_sample
```

The one-day command writes its output to `gfs_weather_sample/`. To regenerate this dataset, run `python download_gfs_weather.py --start 2025-09-30 --end 2026-10-01 --output data`. For other ranges, `--end` is exclusive. Rerunning an interval verifies and skips completed parts. The remote archive reads full spatial chunks, so the data transferred can be much larger than the station Parquet files.

NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives. [Dataset and CC BY 4.0 attribution](https://dynamical.org/catalog/noaa-gfs-forecast-virtual/).
