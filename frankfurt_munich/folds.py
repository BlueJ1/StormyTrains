"""Hold-out set and cross-validation folds.

All journeys from July 2024 to September 2026 are split at random by calendar
day of the planned Frankfurt departure, so all journeys of one day land in the
same part: journeys on the same day share weather, strikes and network
disruptions. 20% of the days form the hold-out set; the other 80% are the
cross-validation set, divided into 5 folds of days in the same way. Both splits
are stratified by `severity_class` (scikit-learn's StratifiedGroupKFold), so
every part has about the same class shares. The seed is fixed, so a rebuild
gives the same split.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5
HOLDOUT_PARTS = 5  # one of 5 parts of the days, i.e. 20%, is the hold-out set
TIME_COLUMN = "origin_departure_planned_local"
# Not a split: the data-quality reports compare the first 15 months with the
# last 12, from this day on, to describe changes in the source over time.
LATER_PERIOD_START = pd.Timestamp("2025-10-01")


def _day_parts(journeys: pd.DataFrame, n_parts: int) -> pd.Series:
    """Part number 1..n_parts for each journey, grouped by day, stratified by class."""
    day = journeys[TIME_COLUMN].dt.normalize()
    splitter = StratifiedGroupKFold(n_splits=n_parts, shuffle=True, random_state=SEED)
    part = pd.Series(0, index=journeys.index)
    for number, (_, rows) in enumerate(splitter.split(journeys, journeys.severity_class.astype(str), day), start=1):
        part.iloc[rows] = number
    return part


def assign_splits(journeys: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (is_holdout, fold) for each journey; fold is 1..5 for CV journeys, missing otherwise."""
    is_holdout = _day_parts(journeys, HOLDOUT_PARTS).eq(1)
    fold = _day_parts(journeys[~is_holdout], N_FOLDS).reindex(journeys.index).astype("Int64")
    return is_holdout, fold


def cv_folds(journeys: pd.DataFrame):
    """Yield (fold number, training mask, validation mask) for the CV journeys."""
    for number in range(1, N_FOLDS + 1):
        validation = journeys.fold.eq(number)
        yield number, ~validation, validation
