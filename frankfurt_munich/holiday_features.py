"""Calendar features for the Frankfurt--München CV departure dates.

The function uses the planned local Frankfurt departure date. It does not
read journey outcomes or prediction timestamps.

Coverage is the CV set's observed span, 2024-07-01 through 2026-09-30. Public
holidays are calculated for Hesse, Baden-Württemberg and Bavaria from the
state holiday rules. School-break dates are pinned below to the KMK calendars
for school years 2023/24, 2024/25 and 2025/26, plus the KMK 2025--2030 summer
schedule for summer 2026. Ranges include both endpoints.

Sources:
- Hesse holidays: https://innen.hessen.de/Buerger-Staat/Feiertage
- Baden-Württemberg Feiertagsgesetz, § 1:
  https://www.landesrecht-bw.de/bsbw/?aiz=1&docId=jlr-NNLBW00008DAE
- Bavarian Feiertagsgesetz, Art. 1:
  https://www.gesetze-bayern.de/Content/Document/BayFTG/True
- KMK school calendar, 2023/24:
  https://www.kmk.org/fileadmin/Dateien/pdf/Ferienkalender/FER2023_24-neu.pdf
- KMK school calendar, 2024/25:
  https://www.kmk.org/fileadmin/Dateien/pdf/Ferienkalender/FER2024_25.pdf
- KMK school calendar, 2025/26:
  https://www.kmk.org/fileadmin/Dateien/pdf/Ferienkalender/FER2025_26.pdf
- KMK summer schedule, 2025--2030:
  https://www.kmk.org/fileadmin/Dateien/pdf/Ferienkalender/Sommerferienregelung_2025-2030_2022-09-21.pdf

Scope limits: dates are state-wide holidays only. Bavaria's Assumption Day
and Augsburg Peace Festival vary by municipality and are excluded. KMK's
movable school-free days are not included unless the calendar lists their
dates. Local school calendars can therefore differ from these state ranges.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

FEATURES = [
    "public_holiday_he",
    "public_holiday_bw",
    "public_holiday_by",
    "school_holiday_he",
    "school_holiday_bw",
    "school_holiday_by",
    "bridge_day",
    "day_before_holiday",
    "day_after_holiday",
]

SUPPORTED_START = date(2024, 7, 1)
SUPPORTED_END = date(2026, 9, 30)
_STATES = ("he", "bw", "by")


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _public_holidays(year: int) -> dict[str, set[date]]:
    easter = _easter_sunday(year)
    shared = {
        date(year, 1, 1),
        easter - timedelta(days=2),  # Good Friday
        easter + timedelta(days=1),  # Easter Monday
        date(year, 5, 1),
        easter + timedelta(days=39),  # Ascension Day
        easter + timedelta(days=50),  # Whit Monday
        date(year, 10, 3),
        date(year, 12, 25),
        date(year, 12, 26),
    }
    regional = {
        "he": set(shared) | {easter + timedelta(days=60)},  # Corpus Christi
        "bw": set(shared) | {
            date(year, 1, 6),  # Epiphany
            easter + timedelta(days=60),  # Corpus Christi
            date(year, 11, 1),  # All Saints' Day
        },
        "by": set(shared) | {
            date(year, 1, 6),  # Epiphany
            easter + timedelta(days=60),  # Corpus Christi
            date(year, 11, 1),  # All Saints' Day
        },
    }
    return regional


# KMK dates as YYYY-MM-DD inclusive ranges. The 2024/25 calendar lists
# BW's 31 October school-free day separately; it is kept as a one-day range.
_SCHOOL_BREAKS = {
    "he": [
        ("2024-07-15", "2024-08-23"),
        ("2024-10-14", "2024-10-25"),
        ("2024-12-23", "2025-01-10"),
        ("2025-04-07", "2025-04-21"),
        ("2025-07-07", "2025-08-15"),
        ("2025-10-06", "2025-10-18"),
        ("2025-12-22", "2026-01-10"),
        ("2026-03-30", "2026-04-10"),
        ("2026-06-29", "2026-08-07"),
    ],
    "bw": [
        ("2024-07-25", "2024-09-07"),
        ("2024-10-28", "2024-10-31"),
        ("2024-12-23", "2025-01-04"),
        ("2025-04-14", "2025-04-26"),
        ("2025-06-10", "2025-06-20"),
        ("2025-07-31", "2025-09-13"),
        ("2025-10-27", "2025-10-31"),
        ("2025-12-22", "2026-01-05"),
        ("2026-03-30", "2026-04-11"),
        ("2026-05-26", "2026-06-05"),
        ("2026-07-30", "2026-09-12"),
    ],
    "by": [
        ("2024-07-29", "2024-09-09"),
        ("2024-10-28", "2024-10-31"),
        ("2024-12-23", "2025-01-03"),
        ("2025-03-03", "2025-03-07"),
        ("2025-04-14", "2025-04-25"),
        ("2025-06-10", "2025-06-20"),
        ("2025-08-04", "2025-09-15"),
        ("2025-11-03", "2025-11-07"),
        ("2025-12-22", "2026-01-05"),
        ("2026-02-16", "2026-02-20"),
        ("2026-03-30", "2026-04-10"),
        ("2026-05-26", "2026-06-05"),
        ("2026-08-03", "2026-09-14"),
    ],
}


def _school_dates(state: str) -> set[date]:
    dates: set[date] = set()
    for start_text, end_text in _SCHOOL_BREAKS[state]:
        start = date.fromisoformat(start_text)
        end = date.fromisoformat(end_text)
        dates.update(start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return dates


def _holiday_dates(years: set[int]) -> dict[str, set[date]]:
    result = {state: set() for state in _STATES}
    for year in years:
        for state, days in _public_holidays(year).items():
            result[state].update(days)
    return result


def add_holiday_features(
    rows: pd.DataFrame,
    date_column: str = "origin_departure_planned_local",
) -> pd.DataFrame:
    """Return ``rows`` with the deterministic holiday columns added.

    ``date_column`` must contain local, timezone-naive departure timestamps in
    the CV coverage interval. The input is not modified. Holiday-adjacency
    features use the union of state-wide public holidays across the three
    states. ``bridge_day`` is a Friday after a Thursday holiday or a Monday
    before a Tuesday holiday.
    """
    if date_column not in rows:
        raise KeyError(f"missing holiday date column {date_column!r}")
    departure = pd.to_datetime(rows[date_column], errors="raise")
    if departure.isna().any():
        raise ValueError(f"{date_column!r} contains missing dates")
    if departure.dt.tz is not None:
        raise ValueError(f"{date_column!r} must contain local timezone-naive dates")

    days = departure.dt.date
    if not days.empty and (min(days) < SUPPORTED_START or max(days) > SUPPORTED_END):
        raise ValueError(
            "holiday calendar supports only the CV date span "
            f"{SUPPORTED_START.isoformat()} through {SUPPORTED_END.isoformat()}"
        )

    years = {day.year for day in days}
    public_by_state = _holiday_dates(years)
    public_any = set().union(*public_by_state.values())
    school_by_state = {state: _school_dates(state) for state in _STATES}
    result = rows.copy()
    for state in _STATES:
        result[f"public_holiday_{state}"] = days.isin(public_by_state[state]).to_numpy(dtype=bool)
        result[f"school_holiday_{state}"] = days.isin(school_by_state[state]).to_numpy(dtype=bool)

    # Friday after Thursday, or Monday before Tuesday, with a public holiday
    # observed in at least one of the three states.
    bridge = []
    day_before = []
    day_after = []
    for day in days:
        bridge.append(
            (day.weekday() == 4 and day - timedelta(days=1) in public_any)
            or (day.weekday() == 0 and day + timedelta(days=1) in public_any)
        )
        day_before.append(day + timedelta(days=1) in public_any)
        day_after.append(day - timedelta(days=1) in public_any)
    result["bridge_day"] = bridge
    result["day_before_holiday"] = day_before
    result["day_after_holiday"] = day_after
    return result
