import numpy as np
import pandas as pd
import pytest

from build_forecast_weather import FORECAST, FORECAST_LAG, build, forecast_features, latest_run
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

