"""Load the modelling rows for one prediction time.

The rows are data/prediction_rows_<split>.parquet for that prediction time,
joined with the journey columns the features need and with the derived
columns below. Missing `current_delay` values are kept; the models decide
how to treat them.

Derived columns:
- `planned_travel_min`: planned München arrival minus planned Frankfurt
  departure, in minutes;
- `hour`, `weekday` (0 = Monday), `month`: of the planned Frankfurt departure.

CV rows keep their `fold` (see frankfurt_munich/folds.py). The hold-out set is
not to be looked at yet, so loading it needs allow_holdout=True.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE.parent))
from folds import TIME_COLUMN  # noqa: E402  frankfurt_munich/folds.py

PREDICTION_TIMES = ["24h", "6h", "1h", "20min", "0min"]
JOURNEY_COLUMNS = ["journey_id", "train_type", "starts_at_origin",
                   "destination_arrival_planned_local", "observed_stops",
                   "recorded_route", "unobserved_sequence_positions"]


def load_rows(prediction_time: str, split: str = "cv",
              allow_holdout: bool = False) -> pd.DataFrame:
    if prediction_time not in PREDICTION_TIMES:
        raise ValueError(f"unknown prediction time {prediction_time!r}")
    if split not in ("cv", "holdout"):
        raise ValueError(f"unknown split {split!r}")
    if split == "holdout" and not allow_holdout:
        raise PermissionError("The hold-out set is not to be looked at yet; "
                              "pass allow_holdout=True for the final evaluation.")

    rows = pd.read_parquet(DATA / f"prediction_rows_{split}.parquet")
    rows = rows[rows["prediction_time"] == prediction_time]
    journeys = pd.read_parquet(DATA / f"journeys_{split}.parquet",
                               columns=JOURNEY_COLUMNS)
    rows = rows.merge(journeys, on="journey_id", how="left", validate="one_to_one")

    departure = rows[TIME_COLUMN]
    rows["planned_travel_min"] = (
        rows["destination_arrival_planned_local"] - departure).dt.total_seconds() / 60
    rows["hour"] = departure.dt.hour
    rows["weekday"] = departure.dt.dayofweek
    rows["month"] = departure.dt.month
    return rows.reset_index(drop=True)
