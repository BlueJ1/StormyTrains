"""Load the modelling rows for one prediction time.

The rows are data/prediction_rows_<split>.parquet for that prediction time,
joined with the journey columns the features need and with the derived
columns below. Missing `current_delay` values are kept; the models decide
how to treat them.

Derived columns:
- `planned_travel_min`: planned München arrival minus planned Frankfurt
  departure, in minutes;
- `hour`, `weekday` (0 = Monday), `month`: of the planned Frankfurt departure;
- `outcome_known_at`: when the journey's outcome was known, the latest of the
  reported München arrival, the planned München arrival and
  `cancellation_announced`. Used for the gap between training and validation.

The hold-out year is not to be looked at yet, so loading it needs
allow_holdout=True.
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
LEAD = {"24h": pd.Timedelta(hours=24), "6h": pd.Timedelta(hours=6),
        "1h": pd.Timedelta(hours=1), "20min": pd.Timedelta(minutes=20),
        "0min": pd.Timedelta(0)}
JOURNEY_COLUMNS = ["journey_id", "train_type", "starts_at_origin",
                   "destination_arrival_planned_local",
                   "destination_arrival_reported_local", "observed_stops",
                   "recorded_route", "unobserved_sequence_positions",
                   "cancellation_announced"]


def load_rows(prediction_time: str, split: str = "cv",
              allow_holdout: bool = False) -> pd.DataFrame:
    if prediction_time not in LEAD:
        raise ValueError(f"unknown prediction time {prediction_time!r}")
    if split not in ("cv", "holdout"):
        raise ValueError(f"unknown split {split!r}")
    if split == "holdout" and not allow_holdout:
        raise PermissionError("The hold-out year is not to be looked at yet; "
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
    rows["outcome_known_at"] = rows[["destination_arrival_reported_local",
                                     "destination_arrival_planned_local",
                                     "cancellation_announced"]].max(axis=1)
    return rows.reset_index(drop=True)
