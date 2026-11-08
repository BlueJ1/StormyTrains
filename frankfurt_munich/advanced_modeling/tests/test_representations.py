import numpy as np
import pandas as pd
import pytest

from cv import PROBA_COLUMNS, cross_validate
from data import TIME_COLUMN, load_rows
from features import feature_columns
from models import make_logistic_regression
from representations import REPRESENTATIONS, WEATHER_LAG, PeriodicCalendar, RouteFrequency, RouteTarget


def test_calendar_splines_wrap_at_each_clock_boundary():
    rows = load_rows("1h")[["hour", "weekday", "month"]].head(20)
    calendar = PeriodicCalendar().fit(rows)
    original = calendar.transform(rows)
    shifted = calendar.transform(rows + [24, 7, 12])
    np.testing.assert_allclose(shifted, original, atol=1e-12)


@pytest.mark.parametrize("variant", REPRESENTATIONS)
def test_representation_fits_and_predicts_without_changing_the_cohort(variant):
    weather = variant.startswith("weather")
    rows = load_rows("1h", weather=weather).groupby("fold", group_keys=False).head(150)
    if variant == "weather_change":
        rows = rows.assign(**dict.fromkeys(WEATHER_LAG, 0.0))
    if variant == "route_onehot":
        pytest.skip("one-hot is already the linear reference; this variant is for boosters")
    numeric, categorical = feature_columns("1h", weather)
    if variant == "route_frequency":
        rows["route_id"] = rows["recorded_route"]
        categorical.append("route_id")
    build, numeric, categorical = REPRESENTATIONS[variant].wrap(make_logistic_regression,
                                                               numeric, categorical)
    _, predictions = cross_validate(rows, "1h", "severity", build, numeric, categorical)
    assert set(predictions["journey_id"]) == set(rows["journey_id"])
    np.testing.assert_allclose(predictions[PROBA_COLUMNS].sum(axis=1), 1)


def test_route_target_encoding_never_uses_its_own_week():
    weeks = pd.date_range("2025-01-06", periods=4, freq="W-MON")
    X = pd.DataFrame({TIME_COLUMN: weeks.repeat(2),
                      "route_id": [f"route_{i}" for i in range(4) for _ in range(2)]})
    y = np.repeat([0.0, 10.0, 20.0, 30.0], 2)
    encoding = RouteTarget("delay").fit_transform(X, y)["route_target"]
    np.testing.assert_allclose(encoding, (60 - y) / 3)


def test_missing_route_does_not_encode_fold_priors():
    weeks = pd.date_range("2025-01-06", periods=4, freq="W-MON")
    X = pd.DataFrame({TIME_COLUMN: weeks.repeat(2), "route_id": "unknown"})
    y = np.repeat([0.0, 10.0, 20.0, 30.0], 2)
    encoder = RouteTarget("delay")
    assert encoder.fit_transform(X, y)["route_target"].isna().all()
    assert encoder.transform(X)["route_target"].isna().all()


def test_route_frequency_uses_numpy_dtype_with_arrow_routes():
    routes = pd.DataFrame({"route_id": pd.array(["A", "A", "B"], dtype="string[pyarrow]")})
    encoder = RouteFrequency().fit(routes)
    unseen = pd.DataFrame({"route_id": pd.array(["A", "C"], dtype="string[pyarrow]")})
    frequency = encoder.transform(unseen)["route_frequency"]
    assert frequency.dtype == np.dtype("float64")
    np.testing.assert_allclose(frequency, [2 / 3, 0])
