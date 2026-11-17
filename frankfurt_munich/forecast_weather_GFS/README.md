# StormyTrains GFS weather data

We chose NOAA's Global Forecast System (GFS) because the [dynamical.org archive](https://dynamical.org/catalog/noaa-gfs-forecast-virtual/) covers our study period and preserves each forecast run. GFS starts a new run every six hours. Our extract provides hourly temperature, precipitation amount and type (`wrtr`), wind speed, gusts, and snow depth along the Frankfurt–München route, up to 48 hours after each run starts. The run and valid times help us match the weather forecast to each train prediction cutoff and planned journey.

Historical coverage drove the choice of forecast source. The DWD regional forecast archives we found did not provide continuous coverage of individual runs across the full study period. For example, dynamical.org's DWD ICON-EU archive starts on 10 February 2026. [DWD ICON-EU archive](https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/)

The files in `data/` cover weather valid from 2024-07-01 through 2026-09-30, inclusive, in `Europe/Berlin`. Each of the raw and processed datasets has 2,577,792 rows across 28 date partitions. September 2025 is split into two partitions at the download boundary on 30 September. The tables retain every six-hour NOAA GFS cycle and hourly lead from 0 to 48 hours with a valid time in this window, including cycles initialized before the first day. The 48-hour limit is a project choice for the planned 24-hour prediction horizon and a buffer for the weather forecast's age and the train journey.

`gfs_stations.csv` maps the project's 19 Frankfurt–München railway stations to 16 nearest 0.25° GFS cells. Coordinates come from the existing Frankfurt–München weather station mapping. The Parquet tables contain one row per grid point, initialization time and lead hour. Both time columns are UTC. A later stage can use the station map and select a forecast that was available by each train prediction time. The GFS initialization time records when the run starts. The archive extract has no historical publication timestamp. Apply and document an availability rule when matching a run to a train prediction cutoff.

| File | Contents |
|---|---|
| `gfs_raw_*.parquet` | Archive values for 2 m temperature (°C), precipitation accumulation (kg/m²), 10 m wind components and surface gust (m/s), and snow depth (m). |
| `gfs_weather_*.parquet` | Temperature (°C), precipitation in the preceding hour (mm), 10 m wind speed and surface gust (m/s), snow depth (cm), and precipitation type (`wrtr`). |
| `gfs_manifest_*.json` | Date bounds, grid points, row counts, missing-value counts, file sizes and SHA-256 hashes. |

GFS precipitation accumulates over the preceding 1–6 hours. The export differences consecutive leads within each six-hour accumulation period and uses the source amount directly at the first hour of each period. Precipitation at lead zero and negative hourly differences are missing. The raw table preserves the original archive values.

## Load the processed forecasts

Run this from the repository root with pandas and PyArrow installed. The `gfs_weather_*.parquet` pattern selects all 28 processed partitions and loads them into one DataFrame.

```python
from pathlib import Path

import pandas as pd

data_dir = Path("frankfurt_munich/forecast_weather_GFS/data")
weather = pd.concat(
    [pd.read_parquet(path) for path in sorted(data_dir.glob("gfs_weather_*.parquet"))],
    ignore_index=True,
)
```

## Precipitation type

`add_gfs_wrtr.py` adds a nullable integer `wrtr` column to the processed Parquet. It follows the DWD code meanings used in this project. `0` means no precipitation, `6` means liquid (rain or freezing rain), `7` means solid (snow or ice pellets), and `8` means both liquid and solid. [DWD WRTR codes](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf)

The script reads the four NOAA GFS precipitation-type flags for the same run, lead and grid point. For a wet hour, it uses the interval flags if that hour accounts for all precipitation in the 1–6 hour accumulation window. Otherwise, it uses the instantaneous flags at the end of the hour. Missing hourly amounts and wet hours without a usable type stay null. The manifest records the number of wet rows with an unknown type. [NOAA GFS product inventory](https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml)

Across the full dataset, GFS forecast a positive hourly precipitation amount in 510,859 rows. In 159,538 of those rows (31.2%), the available flags did not identify the precipitation type, so `wrtr` is null.

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

The one-day command writes its output to `gfs_weather_sample/`. To reproduce the published partitions, run both date ranges, then run the precipitation-type step above.

```sh
python download_gfs_weather.py --start 2024-07-01 --end 2025-09-30 --output gfs_weather_base
python download_gfs_weather.py --start 2025-09-30 --end 2026-10-01 --output gfs_weather_base
```

The `--end` date is exclusive. Rerunning an interval verifies and skips completed parts. The remote archive reads full spatial chunks, so the data transferred can be much larger than the station Parquet files.

## References

- **dynamical.org.** *NOAA GFS forecast, virtual.* Archive used for the weather extract. NOAA NWS NCEP GFS data were processed by dynamical.org from NOAA Open Data Dissemination archives. [Dataset and CC BY 4.0 attribution](https://dynamical.org/catalog/noaa-gfs-forecast-virtual/)
- **NOAA National Centers for Environmental Prediction.** *Global Forecast System (GFS).* The original GRIB2 archive used for the precipitation-type flags and the product inventory that lists those fields. [Open data archive](https://registry.opendata.aws/noaa-gfs-bdp-pds/), [GFS 0.25° product inventory](https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml)
- **dynamical.org.** *DWD ICON-EU forecast, 5 day.* Archive consulted when comparing historical forecast coverage. [Dataset catalog](https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/)
- **Deutscher Wetterdienst (2024).** *Hourly station observations of precipitation for Germany* (v24.03). Dataset documentation for the DWD `WRTR` codes. [Dataset description, p. 2](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf#page=2)
