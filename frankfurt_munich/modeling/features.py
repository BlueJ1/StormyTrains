"""Feature columns per prediction time.

Only information known at the prediction time is used: the timetable,
calendar features of the planned departure, the recorded route, and
`current_delay`. `current_delay` is used from 1 h on; at 24 h it is always
missing and at 6 h 93% missing.

Left out because they describe the outcome: reported times, the cancellation
columns, `cancellation_announced`, `cancellation_withdrawn`,
`shared_departure`, `is_replacement_train` and the targets.

Caveat: `recorded_route`, `observed_stops` and `unobserved_sequence_positions`
come from the recorded stops, not the planned ones.
"""
from __future__ import annotations

NUMERIC = ["planned_travel_min", "observed_stops", "unobserved_sequence_positions"]
CATEGORICAL = ["train_type", "starts_at_origin", "recorded_route",
               "hour", "weekday", "month"]
CURRENT_DELAY_TIMES = {"1h", "20min", "0min"}


def feature_columns(prediction_time: str) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature columns for a prediction time."""
    numeric = list(NUMERIC)
    if prediction_time in CURRENT_DELAY_TIMES:
        numeric.append("current_delay")
    return numeric, list(CATEGORICAL)
