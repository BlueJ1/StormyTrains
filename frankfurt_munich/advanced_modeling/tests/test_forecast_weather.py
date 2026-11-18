import numpy as np
import pandas as pd
import pytest

from build_forecast_weather import (FORECAST, FORECAST_LAG, GFS_COLUMNS, build, forecast_features, latest_run,
                                    load_gfs, station_cells)
from weather_features import ROUTES

CELLS = {eva: f"cell-{eva}" for evas in ROUTES.values() for eva in evas}
UTC = "UTC"


def gfs_frame(runs, *, max_lead=48, value=lambda run, lead, cell: lead):
    """Synthetic GFS rows for every route cell: each variable equals `value` (default: the lead)."""
    rows = []
    for run in pd.to_datetime(runs, utc=True):
        for lead in range(max_lead + 1):
            for cell in sorted(set(CELLS.values())):
                v = float(value(run, lead, cell))
                rows.append({"init_time_utc": run, "valid_time_utc": run + pd.Timedelta(hours=lead),
                             "cell": cell, "temperature_c": v,
                             "precipitation_1h_mm": np.nan if lead == 0 else v,
                             "wind_speed_10m_ms": v, "wind_gust_ms": v, "snow_depth_cm": v})
    frame = pd.DataFrame(rows)
    for column in ("init_time_utc", "valid_time_utc"):
        frame[column] = frame[column].astype("datetime64[ns, UTC]")
    return frame


def row(predicted_at, departure, arrival, branch="nuernberg", prediction_time="1h"):
    return pd.DataFrame({
        "journey_id": ["j"], "prediction_time": [prediction_time],
        "predicted_at": [pd.Timestamp(predicted_at)],
        "origin_departure_planned_local": [pd.Timestamp(departure)],
        "destination_arrival_planned_local": [pd.Timestamp(arrival)],
        "weather_branch": pd.array([branch], dtype="string"),
    })


def test_latest_run_never_after_the_cutoff():
    runs = pd.DatetimeIndex(pd.to_datetime(["2025-01-15 00:00", "2025-01-15 06:00"], utc=True))
    at = pd.Series(pd.to_datetime(["2025-01-14 23:59", "2025-01-15 05:59", "2025-01-15 06:00"], utc=True))
    chosen = latest_run(at, runs)
    assert chosen.isna().iloc[0]
    assert chosen.iloc[1] == runs[0]
    assert chosen.iloc[2] == runs[1]  # available from its start time (B13)


def test_journey_hours_and_precipitation_after_the_first_hour():
    # Winter: local = UTC + 1. Cutoff 07:10 UTC -> run 06 UTC. Journey 08:10-12:20 local = 07:10-11:20 UTC,
    # so hours 07..12 UTC, leads 1..6; precipitation hours 08..12 UTC, leads 2..6.
    gfs = gfs_frame(["2025-01-15 00:00", "2025-01-15 06:00"])
    out = forecast_features(row("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20"), gfs, CELLS)
    assert out.forecast_run_utc.iloc[0] == pd.Timestamp("2025-01-15 06:00", tz=UTC)
    assert out.forecast_origin_temperature_c.iloc[0] == 1
    assert out.forecast_destination_temperature_c.iloc[0] == 6
    assert out.forecast_route_temp_min_c.iloc[0] == 1
    assert out.forecast_route_temp_max_c.iloc[0] == 6
    assert out.forecast_route_precip_max_mm.iloc[0] == 6
    assert out.forecast_route_precip_total_mm.iloc[0] == 2 + 3 + 4 + 5 + 6
    assert out.forecast_hours_missing.iloc[0] == 0


def test_departure_on_the_run_hour_needs_no_lead_zero_precipitation():
    gfs = gfs_frame(["2025-01-15 06:00"])
    out = forecast_features(row("2025-01-15 07:00", "2025-01-15 07:00", "2025-01-15 09:00",
                                prediction_time="0min"), gfs, CELLS)
    assert out.forecast_origin_temperature_c.iloc[0] == 0  # lead 0 temperature is used
    assert out.forecast_route_precip_total_mm.iloc[0] == 1 + 2


def test_leads_beyond_the_extract_are_missing():
    gfs = gfs_frame(["2025-01-15 06:00"], max_lead=3)
    out = forecast_features(row("2025-01-15 07:10", "2025-01-15 08:10", "2025-01-15 12:20"), gfs, CELLS)
    assert out.forecast_route_temp_max_c.iloc[0] == 3
    assert np.isnan(out.forecast_destination_temperature_c.iloc[0])
    assert out.forecast_hours_missing.iloc[0] > 0


def test_unknown_branch_keeps_only_endpoint_temperatures():
    gfs = gfs_frame(["2025-01-15 06:00"])
    out = forecast_features(row("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20", branch=None),
                            gfs, CELLS)
    assert out.forecast_origin_temperature_c.notna().all()
    assert out.forecast_destination_temperature_c.notna().all()
    assert out[[c for c in FORECAST if c.startswith("forecast_route_")]].isna().all(axis=None)


def test_route_cells_follow_the_branch():
    hot = CELLS["8000096"]  # Stuttgart, only on the Stuttgart branch
    gfs = gfs_frame(["2025-01-15 06:00"], value=lambda run, lead, cell: 30 if cell == hot else 0)
    args = ("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20")
    assert forecast_features(row(*args, branch="stuttgart"), gfs, CELLS).forecast_route_temp_max_c.iloc[0] == 30
    assert forecast_features(row(*args, branch="nuernberg"), gfs, CELLS).forecast_route_temp_max_c.iloc[0] == 0


def test_lag_moves_cutoff_and_journey_back_24_hours():
    runs = ["2025-01-14 06:00", "2025-01-15 06:00"]
    gfs = gfs_frame(runs, value=lambda run, lead, cell: run.day * 100 + lead)
    out = build(row("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20"), gfs, CELLS)
    assert out.forecast_origin_temperature_c.iloc[0] == 1501
    assert out.forecast_origin_temperature_c_lag24h.iloc[0] == 1401
    assert set(FORECAST_LAG) <= set(out.columns)


def test_unrecognized_branch_is_rejected():
    gfs = gfs_frame(["2025-01-15 06:00"], max_lead=1)
    with pytest.raises(ValueError, match="unrecognized"):
        forecast_features(row("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20", branch="elsewhere"),
                          gfs, CELLS)



def test_journeys_across_the_clock_changes():
    # Autumn 2025: departure 01:30 CEST (23:30 UTC), arrival 02:30 in the repeated hour, taken as
    # the earlier moment (00:30 UTC). Cutoff 22:30 UTC -> run 18 UTC; hours 23..01 UTC, leads 5..7.
    gfs = gfs_frame(["2025-10-25 18:00", "2025-10-26 00:00"])
    out = forecast_features(row("2025-10-26 00:30", "2025-10-26 01:30", "2025-10-26 02:30"), gfs, CELLS)
    assert out.forecast_run_utc.iloc[0] == pd.Timestamp("2025-10-25 18:00", tz=UTC)
    assert out.forecast_origin_temperature_c.iloc[0] == 5
    assert out.forecast_destination_temperature_c.iloc[0] == 7
    assert out.forecast_route_precip_total_mm.iloc[0] == 6 + 7
    # Spring 2026: departure 01:30 CET (00:30 UTC), arrival 04:30 CEST (02:30 UTC), two real hours.
    # Cutoff 23:30 UTC -> run 18 UTC; hours 00..03 UTC, leads 6..9.
    gfs = gfs_frame(["2026-03-28 18:00", "2026-03-29 00:00"])
    out = forecast_features(row("2026-03-29 00:30", "2026-03-29 01:30", "2026-03-29 04:30"), gfs, CELLS)
    assert out.forecast_run_utc.iloc[0] == pd.Timestamp("2026-03-28 18:00", tz=UTC)
    assert out.forecast_origin_temperature_c.iloc[0] == 6
    assert out.forecast_destination_temperature_c.iloc[0] == 9
    assert out.forecast_route_precip_total_mm.iloc[0] == 7 + 8 + 9


def write_partition(directory, stations, *, longitude_offset=0.0, max_lead=50):
    """A GFS partition in the export's layout for the cells of the real station map."""
    grid = stations[["grid_latitude", "grid_longitude"]].drop_duplicates()
    run = pd.Timestamp("2025-01-15 06:00", tz=UTC)
    frame = pd.DataFrame([
        {"init_time_utc": run, "valid_time_utc": run + pd.Timedelta(hours=lead), "lead_hour": lead,
         "grid_latitude": lat, "grid_longitude": lon + longitude_offset, "temperature_c": float(lead),
         "precipitation_1h_mm": np.nan if lead == 0 else float(lead), "wind_speed_10m_ms": 1.0,
         "wind_gust_ms": 2.0, "snow_depth_cm": 0.0}
        for lead in range(max_lead + 1) for lat, lon in grid.itertuples(index=False)])
    frame.to_parquet(directory / "gfs_weather_2025-01-01_2025-02-01.parquet", index=False)
    assert set(GFS_COLUMNS) <= set(frame.columns)


def test_load_gfs_and_station_cells_round_trip(tmp_path):
    from build_forecast_weather import GFS
    stations = pd.read_csv(GFS / "gfs_stations.csv")
    write_partition(tmp_path, stations)
    gfs = load_gfs(tmp_path)
    assert (gfs.valid_time_utc - gfs.init_time_utc).max() == pd.Timedelta(hours=48)  # lead filter
    cells = station_cells()
    assert set(cells.values()) <= set(gfs.cell)  # the float-formatted keys agree
    out = forecast_features(row("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20"), gfs, cells)
    assert out.forecast_route_temp_max_c.iloc[0] == 6
    assert out.forecast_route_wind_max_ms.iloc[0] == 1.0


def test_cells_missing_from_the_gfs_data_are_rejected(tmp_path):
    from build_forecast_weather import GFS
    stations = pd.read_csv(GFS / "gfs_stations.csv")
    write_partition(tmp_path, stations, longitude_offset=360.0)  # 0-360 longitudes
    with pytest.raises(ValueError, match="station cells not in the GFS data"):
        forecast_features(row("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20"),
                          load_gfs(tmp_path), station_cells())


def test_missing_hours_count_hours_not_cells():
    gfs = gfs_frame(["2025-01-15 06:00"])
    gap = gfs.valid_time_utc.eq(pd.Timestamp("2025-01-15 09:00", tz=UTC)) & gfs.cell.eq(CELLS["8000284"])
    gfs.loc[gap, "wind_gust_ms"] = np.nan  # one variable, one cell, one hour
    out = forecast_features(row("2025-01-15 08:10", "2025-01-15 08:10", "2025-01-15 12:20"), gfs, CELLS)
    assert out.forecast_hours_missing.iloc[0] == 1
