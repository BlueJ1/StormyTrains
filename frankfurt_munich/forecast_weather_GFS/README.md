# StormyTrains GFS weather data

We chose NOAA's Global Forecast System (GFS) because the [dynamical.org archive](https://dynamical.org/catalog/noaa-gfs-forecast-virtual/) covers our study period and preserves each forecast run. GFS starts a new run every six hours. Our extract provides hourly temperature, precipitation, wind speed, gusts and snow depth along the Frankfurt–München route, up to 48 hours after each run starts. The run and valid times help us match the weather forecast to each train prediction cutoff and planned journey.

Historical coverage drove the choice of forecast source. The DWD regional forecast archives we found did not provide continuous coverage of individual runs across the full study period. For example, dynamical.org's DWD ICON-EU archive starts on 10 February 2026. [DWD ICON-EU archive](https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/)

The files in `data/` cover weather valid from 2025-09-30 through 2026-09-30, inclusive, in `Europe/Berlin`. There are 13 date partitions and 1,147,776 rows in each of the raw and processed datasets. The weather tables are provided for joining to railway schedules. It keeps every six-hour NOAA GFS cycle and hourly lead from 0 to 48 hours that falls in this window, including cycles initialized before the first day. The 48-hour limit is a project choice for the planned 24-hour prediction horizon and a buffer for the weather forecast's age and the train journey.

`gfs_stations.csv` maps the project's 19 Frankfurt–München railway stations to 16 nearest 0.25° GFS cells. Coordinates come from the existing Frankfurt–München weather station mapping. The Parquet tables contain one row per grid point, initialization time and lead hour. Both time columns are UTC. A later stage can use the station map and select a forecast that was available by each train prediction time. The GFS initialization time records when the run starts. The archive extract has no historical publication timestamp. Apply and document an availability rule when matching a run to a train prediction cutoff.

| File | Contents |
|---|---|
| `gfs_raw_*.parquet` | Archive values for 2 m temperature (°C), precipitation accumulation (kg/m²), 10 m wind components and surface gust (m/s), and snow depth (m). |
| `gfs_weather_*.parquet` | Temperature (°C), precipitation in the preceding hour (mm), 10 m wind speed and surface gust (m/s), snow depth (cm), and precipitation type (`wrtr`). |
| `gfs_manifest_*.json` | Date bounds, grid points, row counts, missing-value counts, file sizes and SHA-256 hashes. |

GFS precipitation accumulates over the preceding 1–6 hours. The export differences consecutive leads within each six-hour accumulation period and uses the source amount directly at the first hour of each period. Precipitation at lead zero and negative hourly differences are missing. The raw table preserves the original archive values.

## Precipitation type

`add_gfs_wrtr.py` adds a nullable integer `wrtr` column to the processed Parquet. It follows the DWD code meanings used in this project. `0` means no precipitation, `6` means liquid (rain or freezing rain), `7` means solid (snow or ice pellets), and `8` means both liquid and solid. [DWD WRTR codes](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf)

The script reads the four NOAA GFS precipitation-type flags for the same run, lead and grid point. For a wet hour, it uses the interval flags if that hour accounts for all precipitation in the 1–6 hour accumulation window. Otherwise, it uses the instantaneous flags at the end of the hour. Missing hourly amounts and wet hours without a usable type stay null. The manifest records the number of wet rows with an unknown type. [NOAA GFS product inventory](https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml)

From 2025-09-30 through 2026-09-30, GFS forecast a positive hourly precipitation amount in 215,506 rows. In 64,807 of those rows (30.1%), the available flags did not identify the precipitation type, so `wrtr` is null.

Run this on completed exports and write the updated files to a separate directory.

```sh
python -m pip install eccodes requests
python add_gfs_wrtr.py gfs_weather_base --output data_with_wrtr
```

## Run

Use Python 3.12 or newer. Keep the script and `gfs_stations.csv` together.

```sh
python -m venv .venv-gfs
source .venv-gfs/bin/activate
python -m pip install 'dynamical-catalog==1.1.1' pyarrow
ZARR_ASYNC__CONCURRENCY=16 python download_gfs_weather.py --start 2025-09-30 --end 2025-10-01 --output gfs_weather_sample
```

The one-day command writes its output to `gfs_weather_sample/`. For the full published date range, run `python download_gfs_weather.py --start 2025-09-30 --end 2026-10-01 --output gfs_weather_base`, then run the precipitation-type step above. For other ranges, `--end` is exclusive. Rerunning an interval verifies and skips completed parts. The remote archive reads full spatial chunks, so the data transferred can be much larger than the station Parquet files.

NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives. [Dataset and CC BY 4.0 attribution](https://dynamical.org/catalog/noaa-gfs-forecast-virtual/).
