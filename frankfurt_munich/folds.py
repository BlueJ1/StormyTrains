"""Hold-out set and cross-validation folds.

Journeys are split by ISO week (Monday to Sunday) of the planned Frankfurt
departure, so all journeys of one week land in the same part: neighbouring days
share weather, strikes and network disruptions, which often last several days.

The weeks were assigned once by scikit-learn's StratifiedGroupKFold (seed 42),
stratified by `severity_class`: one of 5 parts of the weeks (about 20%) is the
hold-out set, and the remaining weeks are divided into 5 cross-validation folds
in the same way. This assignment is greedy, not random: the weeks are sorted by
how uneven their class counts are and each is put where it best balances the
class shares, so the seed changes little.

The assignment is stored in split_weeks.csv and committed, so that a rebuild of
the data keeps every week in its part. build_dataset.py reads it and fails on
weeks that are not in it; `build_dataset.py --new-split` makes a new one.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5
HOLDOUT_PARTS = 5  # one of 5 parts of the weeks, about 20%, is the hold-out set
TIME_COLUMN = "origin_departure_planned_local"
ASSIGNMENT_FILE = Path(__file__).resolve().parent / "split_weeks.csv"

# The calendar of the data: planned Frankfurt departures from 1 July 2024 to
# 30 September 2026.
CALENDAR_START = pd.Timestamp("2024-07-01")
CALENDAR_END = pd.Timestamp("2026-10-01")
# Not a split: the data-quality reports compare the first 15 months with the
# last 12, from this day on, to describe changes in the source over time.
LATER_PERIOD_START = pd.Timestamp("2025-10-01")
EARLIER_PERIOD, LATER_PERIOD = "before_2025_10", "from_2025_10"


def week_of(times: pd.Series) -> pd.Series:
    """The Monday that starts each time's ISO week."""
    day = times.dt.normalize()
    return day - pd.to_timedelta(day.dt.dayofweek, unit="D")


def period_of(times: pd.Series) -> pd.Series:
    """EARLIER_PERIOD or LATER_PERIOD for each time."""
    return times.ge(LATER_PERIOD_START).map({False: EARLIER_PERIOD, True: LATER_PERIOD})


def _week_parts(journeys: pd.DataFrame, n_parts: int) -> pd.Series:
    """Part number 1..n_parts for each journey, grouped by week, stratified by class."""
    splitter = StratifiedGroupKFold(n_splits=n_parts, shuffle=True, random_state=SEED)
    part = pd.Series(0, index=journeys.index)
    groups = week_of(journeys[TIME_COLUMN])
    for number, (_, rows) in enumerate(
            splitter.split(journeys, journeys.severity_class.astype(str), groups), start=1):
        part.iloc[rows] = number
    return part


def new_assignment(journeys: pd.DataFrame) -> pd.DataFrame:
    """Assign every week to the hold-out set or a fold; one row per week."""
    is_holdout = _week_parts(journeys, HOLDOUT_PARTS).eq(1)
    fold = _week_parts(journeys[~is_holdout], N_FOLDS).reindex(journeys.index)
    weeks = pd.DataFrame({"week_start": week_of(journeys[TIME_COLUMN]),
                          "split": is_holdout.map({True: "holdout", False: "cv"}),
                          "fold": fold.astype("Int64")})
    weeks = weeks.drop_duplicates("week_start").sort_values("week_start")
    assert weeks.week_start.is_unique
    return weeks.reset_index(drop=True)


def read_assignment() -> pd.DataFrame:
    weeks = pd.read_csv(ASSIGNMENT_FILE, parse_dates=["week_start"])
    weeks["fold"] = weeks.fold.astype("Int64")
    return weeks


def assign_splits(journeys: pd.DataFrame, weeks: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (is_holdout, fold) for each journey from a week assignment;
    fold is 1..5 for CV journeys, missing otherwise."""
    w = weeks.set_index("week_start")
    week = week_of(journeys[TIME_COLUMN])
    unknown = ~week.isin(w.index)
    if unknown.any():
        raise ValueError(f"weeks not in {ASSIGNMENT_FILE.name}: {sorted(week[unknown].dt.date.unique())}; "
                         "run build_dataset.py --new-split to make a new assignment")
    is_holdout = week.map(w.split).eq("holdout")
    fold = week.map(w.fold).astype("Int64").where(~is_holdout)
    return is_holdout, fold


def cv_folds(journeys: pd.DataFrame):
    """Yield (fold number, training mask, validation mask) for the CV journeys."""
    for number in range(1, N_FOLDS + 1):
        validation = journeys.fold.eq(number)
        yield number, ~validation, validation
