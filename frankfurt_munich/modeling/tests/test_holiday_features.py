from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from holiday_features import FEATURES, add_holiday_features  # noqa: E402


def test_public_holidays_follow_the_three_state_rules():
    rows = pd.DataFrame({
        "origin_departure_planned_local": pd.to_datetime([
            "2025-01-06",  # Epiphany, Baden-Württemberg and Bavaria
            "2025-06-19",  # Corpus Christi, all three states
            "2025-11-01",  # All Saints' Day, Baden-Württemberg and Bavaria
            "2025-08-15",  # Bavaria's municipality-specific holiday is excluded
        ])
    })

    result = add_holiday_features(rows)

    assert result["public_holiday_he"].tolist() == [False, True, False, False]
    assert result["public_holiday_bw"].tolist() == [True, True, True, False]
    assert result["public_holiday_by"].tolist() == [True, True, True, False]


def test_school_break_ranges_are_inclusive_and_state_specific():
    rows = pd.DataFrame({
        "origin_departure_planned_local": pd.to_datetime([
            "2025-07-30",
            "2025-07-31",
            "2025-08-03",
            "2025-08-04",
            "2025-08-15",
            "2025-08-16",
        ])
    })

    result = add_holiday_features(rows)

    assert result["school_holiday_he"].tolist() == [True, True, True, True, True, False]
    assert result["school_holiday_bw"].tolist() == [False, True, True, True, True, True]
    assert result["school_holiday_by"].tolist() == [False, False, False, True, True, True]


def test_holiday_adjacency_and_bridge_day_definitions():
    rows = pd.DataFrame({
        "origin_departure_planned_local": pd.to_datetime([
            "2025-05-28",  # day before Ascension Day
            "2025-05-29",  # Ascension Day
            "2025-05-30",  # Friday bridge day after Ascension Day
            "2026-01-05",  # Monday bridge day before Tuesday Epiphany
            "2026-01-06",  # Epiphany, Baden-Württemberg and Bavaria
        ])
    })

    result = add_holiday_features(rows)

    assert result["day_before_holiday"].tolist() == [True, False, False, True, False]
    assert result["day_after_holiday"].tolist() == [False, False, True, False, False]
    assert result["bridge_day"].tolist() == [False, False, True, True, False]


def test_adds_only_the_declared_boolean_features_without_mutating_input():
    rows = pd.DataFrame({
        "origin_departure_planned_local": pd.to_datetime(["2025-05-30"]),
        "existing": ["kept"],
    })
    original = rows.copy(deep=True)

    result = add_holiday_features(rows)

    pd.testing.assert_frame_equal(rows, original)
    assert result["existing"].tolist() == ["kept"]
    assert set(result.columns) == {"origin_departure_planned_local", "existing", *FEATURES}
    assert result[FEATURES].dtypes.eq(bool).all()


@pytest.mark.parametrize("outside", ["2024-06-30", "2026-10-01"])
def test_rejects_dates_outside_the_cv_coverage(outside):
    rows = pd.DataFrame({"origin_departure_planned_local": pd.to_datetime([outside])})

    with pytest.raises(ValueError, match="CV date span"):
        add_holiday_features(rows)


def test_rejects_missing_or_timezone_aware_departures():
    missing = pd.DataFrame({"origin_departure_planned_local": [pd.NaT]})
    timezone_aware = pd.DataFrame({
        "origin_departure_planned_local": pd.to_datetime(["2025-05-30T10:00:00+02:00"])
    })

    with pytest.raises(ValueError, match="missing dates"):
        add_holiday_features(missing)
    with pytest.raises(ValueError, match="timezone-naive"):
        add_holiday_features(timezone_aware)


def test_rejects_a_missing_date_column():
    with pytest.raises(KeyError, match="origin_departure_planned_local"):
        add_holiday_features(pd.DataFrame({"other": []}))
