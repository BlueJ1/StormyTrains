"""The engineered weather arms (frankfurt_munich/weather_engineered.py)."""
from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

import data
from data import DATA, PREDICTION_TIMES, load_rows
from features import (CURRENT_DELAY_TIMES, ENDPOINT_TEMPERATURES, WEATHER, WEATHER_FE,
                      WEATHER_FE_INTERACTION, feature_columns, weather_columns)
from weather_engineered import GUST_MS, HEAT_C, ROUTES, journey_branches
from weather_features import hourly_table


@cache
def rows_for(prediction_time, weather=False):
    return load_rows(prediction_time, weather=weather)


@pytest.mark.parametrize("arm", ["fe_replace", "fe_add"])
@pytest.mark.parametrize("prediction_time", PREDICTION_TIMES)
def test_engineered_arms_only_add_columns(prediction_time, arm):
    """Same rows, order and values as the base-weather rows, plus the engineered columns."""
    base, engineered = rows_for(prediction_time, True), rows_for(prediction_time, arm)
    pd.testing.assert_frame_equal(engineered[base.columns], base)
    assert set(engineered.columns) - set(base.columns) == set(WEATHER_FE)


def test_out_of_date_engineered_file_is_refused(tmp_path, monkeypatch):
    for name in ("prediction_rows_cv.parquet", "prediction_rows_weather_cv.parquet"):
        pd.read_parquet(DATA / name).to_parquet(tmp_path / name)
    engineered = pd.read_parquet(DATA / "prediction_rows_weather_fe_cv.parquet")
    engineered.iloc[1:].to_parquet(tmp_path / "prediction_rows_weather_fe_cv.parquet")
    monkeypatch.setattr(data, "DATA", tmp_path)
    with pytest.raises(ValueError, match="weather_engineered.py"):
        load_rows("1h", weather="fe_add")


def test_arm_columns():
    for prediction_time in PREDICTION_TIMES:
        railway, categorical = feature_columns(prediction_time)
        engineered = [c for c in WEATHER_FE if c not in WEATHER_FE_INTERACTION
                      or prediction_time in CURRENT_DELAY_TIMES]
        assert feature_columns(prediction_time, "fe_replace") == (
            railway + [*ENDPOINT_TEMPERATURES, *engineered], categorical)
        assert feature_columns(prediction_time, "fe_add") == (railway + WEATHER + engineered, categorical)
        assert weather_columns(prediction_time, True) == weather_columns(prediction_time, "base") == WEATHER
        assert not set(WEATHER_FE) & set(railway) and not set(WEATHER_FE) & set(WEATHER)
    with pytest.raises(ValueError, match="unknown weather arm"):
        weather_columns("1h", "fe")


@pytest.mark.parametrize("prediction_time", PREDICTION_TIMES)
def test_engineered_features_are_complete(prediction_time):
    """Only the first days lack the previous day's snow depth; interactions follow current_delay."""
    rows = rows_for(prediction_time, "fe_add")
    weather = [c for c in WEATHER_FE if c not in WEATHER_FE_INTERACTION]
    missing = rows[weather].isna().sum()
    assert missing.drop("snow_depth_change_cm").eq(0).all()
    assert rows.loc[rows["snow_depth_change_cm"].isna(), "weather_hour_utc"].lt(
        pd.Timestamp("2024-07-02", tz="UTC")).all()
    for column in WEATHER_FE_INTERACTION:
        assert rows[column].isna().equals(rows["current_delay"].isna())


def test_flags_agree_with_the_base_columns():
    rows = rows_for("1h", "fe_add")
    assert rows["heat_flag_24h"].eq(rows["route_temp_max_24h_c"].gt(HEAT_C).astype(float)).all()
    assert rows["storm_flag_24h"].eq(rows["route_gust_max_24h_ms"].gt(GUST_MS).astype(float)).all()
    assert rows["frost_flag_24h"].eq(rows["route_temp_min_24h_c"].lt(0).astype(float)).all()


def test_features_match_a_hand_computation():
    """Recompute some features from the station tables for a sample of rows."""
    rows = rows_for("6h", "fe_add")
    branches = journey_branches()
    t = hourly_table("temperature", "temperature_c")
    g = hourly_table("gust", "gust_ms")
    p = hourly_table("precipitation", "precipitation_mm")
    sample = pd.concat([rows[rows["heat_flag_24h"].eq(1)].sample(20, random_state=0),
                        rows[rows["storm_flag_24h"].eq(1)].sample(20, random_state=0),
                        rows[rows["frost_flag_24h"].eq(1)].sample(20, random_state=0)])
    for _, row in sample.iterrows():
        stations = ROUTES[branches[row.journey_id]]
        hour = row.weather_hour_utc
        day = lambda table, hours=24: table.loc[hour - pd.Timedelta(hours=hours - 1):hour, stations]
        temp, gust = day(t), day(g)
        assert np.isclose(row.heat_degree_hours_24h, (temp - HEAT_C).clip(lower=0).to_numpy().sum())
        assert row.heat_hours_24h == temp.gt(HEAT_C).any(axis=1).sum()
        assert row.heat_stations_24h == temp.gt(HEAT_C).any(axis=0).sum()
        assert np.isclose(row.gust_excess_sq_48h, ((day(g, 48) - GUST_MS).clip(lower=0) ** 2).to_numpy().sum())
        assert row.storm_hours_24h == gust.gt(GUST_MS).any(axis=1).sum()
        assert np.isclose(row.frost_degree_hours_24h, (-temp).clip(lower=0).to_numpy().sum())
        below = t.loc[hour - pd.Timedelta(hours=24):hour, stations].lt(0)
        assert row.freeze_thaw_24h == below.ne(below.shift()).iloc[1:].sum().max()
        assert np.isclose(row.route_precip_72h_mm, day(p, 72).sum().max())
        assert np.isclose(row.rain_excess5_24h, (day(p) - 5).clip(lower=0).to_numpy().sum())
