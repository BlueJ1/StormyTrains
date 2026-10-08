"""Hold-out period and chronological cross-validation folds.

The hold-out set is one full year, October 2025 to September 2026. The
cross-validation period before it (July 2024 to September 2025) gives 5 folds
with a fixed validation window of 2/12 of the period. Fold k trains on the
first k/12 and validates on the following 2/12, for k = 6, 7, 8, 9, 10, so
consecutive validation windows overlap by 1/12 and the last one ends with the
period. Fractions are of calendar time, cut at midnight, by planned Frankfurt
departure.
"""
from __future__ import annotations

import pandas as pd

CV_START = pd.Timestamp("2024-07-01")
HOLDOUT_START = pd.Timestamp("2025-10-01")
HOLDOUT_END = pd.Timestamp("2026-10-01")
TRAIN_TWELFTHS = [6, 7, 8, 9, 10]
VALIDATION_TWELFTHS = 2
TIME_COLUMN = "origin_departure_planned_local"


def fold_bounds() -> list[dict]:
    span = HOLDOUT_START - CV_START
    folds = []
    for number, twelfths in enumerate(TRAIN_TWELFTHS, start=1):
        cut = (CV_START + span * twelfths / 12).floor("D")
        end = (CV_START + span * (twelfths + VALIDATION_TWELFTHS) / 12).floor("D")
        folds.append({"fold": number, "train_start": CV_START, "train_end": cut,
                      "validation_start": cut, "validation_end": end})
    return folds


def cv_folds(journeys: pd.DataFrame):
    """Yield (fold number, training mask, validation mask) for the CV journeys."""
    when = journeys[TIME_COLUMN]
    for f in fold_bounds():
        yield (f["fold"], when.ge(f["train_start"]) & when.lt(f["train_end"]),
               when.ge(f["validation_start"]) & when.lt(f["validation_end"]))
