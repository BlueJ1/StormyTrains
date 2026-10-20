"""Feature columns per prediction time.

Only information known at the prediction time is used: the timetable,
calendar features of the planned departure, the recorded route,
`current_delay`, and, for the weather models, the route weather at the
prediction time. `current_delay` is used from 1 h on; at 24 h it is always
missing and at 6 h 95% missing (CV set).

Left out because they describe the outcome: reported times, the cancellation
columns, `cancellation_announced`, `cancellation_withdrawn`,
`shared_departure`, `is_replacement_train` and the targets.

Caveat: `recorded_route`, `observed_stops` and `unobserved_sequence_positions`
come from the recorded stops, not the planned ones. The weather features do
not: they use a fixed route per branch, chosen by train number.

Weather (frankfurt_munich/weather_features.py, see the README's Weather
section): the same columns at every prediction time, all numeric
(`snow_on_route_24h` as 0/1). WEATHER is that script's FEATURE_COLUMNS, so a
new weather column reaches the weather models. `weather_hour_utc`, the DWD
hour used, is a timestamp and not a feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather_features import FEATURE_COLUMNS as WEATHER  # noqa: E402  frankfurt_munich/weather_features.py

NUMERIC = ["planned_travel_min", "observed_stops", "unobserved_sequence_positions"]
CATEGORICAL = ["train_type", "starts_at_origin", "recorded_route",
               "hour", "weekday", "month"]
CURRENT_DELAY_TIMES = {"1h", "20min", "0min"}


def feature_columns(prediction_time: str, weather: bool = False) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature columns for a prediction time.

    With weather=True, the weather columns are added to the numeric ones.
    """
    numeric = list(NUMERIC)
    if prediction_time in CURRENT_DELAY_TIMES:
        numeric.append("current_delay")
    if weather:
        numeric += WEATHER
    return numeric, list(CATEGORICAL)
