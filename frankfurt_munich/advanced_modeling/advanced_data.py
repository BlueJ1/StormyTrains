"""Inputs for advanced experiments. No fallback to the legacy live features."""
from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

import _paths  # noqa: F401  shared modelling code and dataset builders
from data import DATA, ROW_KEY, load_rows, merge_inputs
from features import CURRENT_DELAY_TIMES, FEATURE_COLUMNS_BY_GROUP, HOLIDAYS, PLANNED_TIMETABLE, WEATHER
from holiday_features import add_holiday_features
from representations import WEATHER_LAG
from sequence_inputs import SEQUENCE_COLUMN

TIMETABLE_INPUTS = {
    "scheduled_train_type": "train_type",
    "scheduled_starts_at_origin": "starts_at_origin",
}
ROUTE_INPUTS = (*PLANNED_TIMETABLE[0], *PLANNED_TIMETABLE[1])
ADVANCED = DATA / "advanced"
ADVANCED_RESULTS = Path(__file__).resolve().parent / "results"


@cache
def read_inputs(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing cutoff-specific inputs: {path}")
    return pd.read_parquet(path, columns=[*ROW_KEY, *columns])


def load_advanced_rows(prediction_time: str, *, weather: bool = False,
                       inputs_dir: Path = ADVANCED, extra_inputs: tuple[str, ...] = (),
                       split: str = "cv", allow_holdout: bool = False) -> pd.DataFrame:
    """Load paired rows with only reconstructed timetable, live and weather inputs."""
    rows = load_rows(prediction_time, split=split, allow_holdout=allow_holdout)
    timetable_columns = (*TIMETABLE_INPUTS, *(c for c in ROUTE_INPUTS if c in extra_inputs))
    timetable = read_inputs(inputs_dir / f"timetable_{split}.parquet", timetable_columns)
    rows = merge_inputs(rows, timetable.rename(columns=TIMETABLE_INPUTS))
    for column in [*TIMETABLE_INPUTS.values(), "route_id"]:
        if column in rows:
            rows[column] = rows[column].astype("string").fillna("unknown")
    if "planned_stops" in rows:
        rows["planned_stops"] = rows["planned_stops"].astype(float)
    history_columns = tuple(c for c in FEATURE_COLUMNS_BY_GROUP["delay_dynamics"] if c in extra_inputs)
    if prediction_time in CURRENT_DELAY_TIMES and "current_delay" not in history_columns:
        history_columns += ("current_delay",)
    if history_columns:
        rows = merge_inputs(rows, read_inputs(inputs_dir / f"history_{split}.parquet", history_columns))
    if prediction_time not in CURRENT_DELAY_TIMES:
        rows["current_delay"] = float("nan")
    for group, name in (("recent_reliability", "reliability"),):
        columns = tuple(c for c in FEATURE_COLUMNS_BY_GROUP[group] if c in extra_inputs)
        if columns:
            rows = merge_inputs(rows, read_inputs(inputs_dir / f"{name}_{split}.parquet", columns))
    if SEQUENCE_COLUMN in extra_inputs:
        rows = merge_inputs(rows, read_inputs(inputs_dir / f"sequence_{split}.parquet", (SEQUENCE_COLUMN,)))
    if weather:
        weather_columns = ("weather_hour_utc", *WEATHER,
                           *(c for c in WEATHER_LAG if c in extra_inputs))
        rows = merge_inputs(rows, read_inputs(inputs_dir / f"weather_{split}.parquet", weather_columns))
        rows["snow_on_route_24h"] = rows["snow_on_route_24h"].astype(float)
    return add_holiday_features(rows) if set(HOLIDAYS).intersection(extra_inputs) else rows
