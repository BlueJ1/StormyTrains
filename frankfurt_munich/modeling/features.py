"""Named feature sets. Outcome-dependent columns are diagnostic-only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "advanced_modeling"))
from history_features import FEATURE_COLUMNS_BY_GROUP  # noqa: E402
from holiday_features import FEATURES as HOLIDAYS  # noqa: E402
from weather_features import FEATURE_COLUMNS as WEATHER  # noqa: E402
from forecast_columns import FORECAST  # noqa: E402

TIMETABLE = (("planned_travel_min",), ("train_type", "starts_at_origin"))
CALENDAR = ((), ("hour", "weekday", "month"))
RECORDED_ROUTE = (("observed_stops", "unobserved_sequence_positions"), ("recorded_route",))
CURRENT_DELAY_TIMES = {"1h", "20min", "0min"}
PLANNED_TIMETABLE = (("planned_stops", "upstream_planned_min"), ("route_id",))
PHASE2_GROUPS = {"planned_timetable": PLANNED_TIMETABLE, "holidays": (tuple(HOLIDAYS), ())}
PHASE2_GROUPS.update({group: (tuple(c for c in columns if c != "current_delay"), ())
                      for group, columns in FEATURE_COLUMNS_BY_GROUP.items()})


def feature_groups(prediction_time: str, weather: bool = False, forecast: bool = False):
    """Disjoint groups for the full Phase 2 set, suitable for Phase 3 ablations.

    weather adds the observed (DWD) groups and forecast the GFS forecast groups
    (D17). The two never share a group, so the ablations (3.1) and the group
    selections (0.8 automatic groups, 3.3) keep or drop them separately."""
    groups = {
        "timetable": TIMETABLE,
        "calendar": (tuple(HOLIDAYS), CALENDAR[1]),
        "route": PLANNED_TIMETABLE,
        "recent_reliability": PHASE2_GROUPS["recent_reliability"],
    }
    if prediction_time in CURRENT_DELAY_TIMES:
        groups["delay"] = (("current_delay", *PHASE2_GROUPS["delay_dynamics"][0]), ())
    if weather:
        groups.update({
            "temperature": (tuple(c for c in WEATHER if "temp" in c), ()),
            "precipitation": (tuple(c for c in WEATHER if "precip" in c), ()),
            "wind": (tuple(c for c in WEATHER if "wind" in c or "gust" in c), ()),
            "snow": (tuple(c for c in WEATHER if "snow" in c), ()),
        })
    if forecast:
        groups.update({
            "forecast_temperature": (tuple(c for c in FORECAST if "temp" in c), ()),
            "forecast_precipitation": (tuple(c for c in FORECAST if "precip" in c), ()),
            "forecast_wind": (tuple(c for c in FORECAST if "wind" in c or "gust" in c), ()),
            "forecast_snow": (tuple(c for c in FORECAST if "snow" in c), ()),
        })
    return groups


def feature_columns(prediction_time: str, weather: bool = False, *,
                    feature_set: str = "base", forecast: bool = False) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns for a named feature set."""
    if feature_set not in ("base", "recorded_route_diagnostic", "all_groups", *PHASE2_GROUPS):
        raise ValueError(f"unknown feature set {feature_set!r}")
    numeric, categorical = map(list, TIMETABLE)
    if feature_set == "recorded_route_diagnostic":
        numeric.extend(RECORDED_ROUTE[0])
        categorical.extend(RECORDED_ROUTE[1])
    categorical.extend(CALENDAR[1])
    if prediction_time in CURRENT_DELAY_TIMES:
        numeric.append("current_delay")
    if weather:
        numeric.extend(WEATHER)
    if forecast:
        numeric.extend(FORECAST)
    added = PHASE2_GROUPS if feature_set == "all_groups" else (feature_set,)
    for group in added:
        if group == "delay_dynamics" and prediction_time not in CURRENT_DELAY_TIMES:
            continue
        extra_numeric, extra_categorical = PHASE2_GROUPS.get(group, ((), ()))
        numeric.extend(extra_numeric)
        categorical.extend(extra_categorical)
    return numeric, categorical
