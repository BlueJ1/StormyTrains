"""Pre-listed Phase 1 representations and their model input columns."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, TargetEncoder

from data import TIME_COLUMN
from features import WEATHER
from folds import N_FOLDS, SEED, week_of
from metrics import CLASSES

CALENDAR = ("month", "hour", "weekday")
PERIODS = {"month": 12, "hour": 24, "weekday": 7}
ORIGINS = {"month": 1, "hour": 0, "weekday": 0}
EXTREMES = {
    "weather_heat": ("route_temp_max_24h_c", 30),
    "weather_gust": ("route_gust_max_24h_ms", 20),
    "weather_snow": ("snow_on_route_24h", 0),
    # DWD's first hourly heavy-rain warning threshold (1 l/m² = 1 mm).
    "weather_heavy_rain": ("route_precip_max_mm", 15),
}
WEATHER_LAG = tuple(f"{column}_lag24h" for column in WEATHER)
WEATHER_CHANGE = tuple(f"{column}_change24h" for column in WEATHER)


@dataclass(frozen=True)
class Representation:
    inputs: tuple[str, ...]
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    transformer: object = None
    extra_inputs: tuple[str, ...] = ()

    def wrap(self, factory, numeric, categorical):
        """Return a factory and the input columns needed before encoding."""
        inputs = set(self.inputs)
        if not inputs <= set(numeric + categorical):
            raise ValueError(f"representation inputs are missing: {inputs - set(numeric + categorical)}")
        encoded_numeric = [c for c in numeric if c not in inputs] + list(self.numeric)
        encoded_categorical = [c for c in categorical if c not in inputs] + list(self.categorical)

        def build(_numeric, _categorical, **params):
            return make_pipeline(self.transformer(), factory(encoded_numeric, encoded_categorical, **params))

        return build, numeric + list(self.extra_inputs), categorical


def stateless(function, **kwargs):
    return partial(FunctionTransformer, function, kw_args=kwargs)


def cyclic_columns(columns):
    return tuple(f"{column}_{wave}" for column in columns for wave in ("sin", "cos"))


def cyclic(X, *, keep_categories):
    result = X.copy()
    for column in CALENDAR:
        angle = 2 * np.pi * (X[column] - ORIGINS[column]) / PERIODS[column]
        result[f"{column}_sin"] = np.sin(angle)
        result[f"{column}_cos"] = np.cos(angle)
    return result if keep_categories else result.drop(columns=list(CALENDAR))


def spline_columns(column):
    return [f"{column}_spline_{i}" for i in range(6)]


class PeriodicCalendar(TransformerMixin, BaseEstimator):
    """Six evenly spaced periodic spline basis functions for each clock."""

    def fit(self, X, y=None):
        self.splines_ = {
            c: SplineTransformer(knots=np.linspace(ORIGINS[c], ORIGINS[c] + PERIODS[c], 7)[:, None],
                                 extrapolation="periodic").fit(X[[c]])
            for c in CALENDAR
        }
        return self

    def transform(self, X):
        result = X.drop(columns=list(CALENDAR)).copy()
        for column, spline in self.splines_.items():
            result[spline_columns(column)] = spline.transform(X[[column]])
        return result


def finer_time(X, *, variant):
    departure = X[TIME_COLUMN].dt
    X = X.drop(columns=TIME_COLUMN).copy()
    if variant == "minute":
        return X.drop(columns="hour").assign(minute_of_day=departure.hour * 60 + departure.minute)
    days = np.where(departure.is_leap_year, 366, 365)
    angle = 2 * np.pi * (departure.dayofyear - 1) / days
    return X.drop(columns="month").assign(day_sin=np.sin(angle), day_cos=np.cos(angle))


def hour_weekday(X):
    return X.assign(hour_weekday=X["hour"] * 7 + X["weekday"])


def delay_transform(X, *, variant):
    delay = X["current_delay"]
    if variant == "clip":
        return X.assign(current_delay=delay.clip(lower=0))
    if variant == "log":
        return X.assign(current_delay=np.log1p(delay.clip(lower=0)))
    if variant == "bins":
        bins = pd.cut(delay, [-np.inf, 0, 5, 20, 60, np.inf], right=False, labels=False)
        return X.drop(columns="current_delay").assign(current_delay_bin=bins.fillna(-1))
    return X.assign(current_delay_missing=delay.isna().astype(float))


def extreme_weather(X):
    result = X.drop(columns=WEATHER).copy()
    for name, (column, threshold) in EXTREMES.items():
        values = X[column]
        result[name] = values.gt(threshold).astype(float).where(values.notna())
    return result


def weather_change(X):
    result = X.drop(columns=[*WEATHER, *WEATHER_LAG]).copy()
    for current, lag, change in zip(WEATHER, WEATHER_LAG, WEATHER_CHANGE):
        result[change] = X[current].astype(float) - X[lag].astype(float)
    return result


class RouteFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.frequencies_ = X["route_id"].value_counts(normalize=True)
        return self

    def transform(self, X):
        return X.drop(columns="route_id").assign(
            route_frequency=X["route_id"].map(self.frequencies_).fillna(0).astype(float))


class RouteOneHot(TransformerMixin, BaseEstimator):
    """One-hot route for a booster that would ordinarily use native categories."""

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[["route_id"]])
        return self

    def transform(self, X):
        values = self.encoder_.transform(X[["route_id"]])
        columns = [f"route_onehot_{i}" for i in range(values.shape[1])]
        return X.drop(columns="route_id").join(pd.DataFrame(values, index=X.index, columns=columns))


class RouteTarget(TransformerMixin, BaseEstimator):
    """Encode routes from other weeks for fit, and all fit rows for prediction."""

    def __init__(self, task):
        self.task = task

    def _encoder(self, X, y):
        weeks = week_of(X[TIME_COLUMN])
        n_splits = N_FOLDS - 1
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)
        strata = y if self.task == "severity" else np.digitize(y, [20, 60])
        target_type = "multiclass" if self.task == "severity" else "continuous"
        return TargetEncoder(cv=list(splitter.split(X, strata, weeks)), target_type=target_type)

    def fit(self, X, y):
        self.encoder_ = self._encoder(X, y).fit(X[["route_id"]], y)
        return self

    def fit_transform(self, X, y):
        self.encoder_ = self._encoder(X, y)
        return self._format(X, self.encoder_.fit_transform(X[["route_id"]], y))

    def transform(self, X):
        return self._format(X, self.encoder_.transform(X[["route_id"]]))

    def _format(self, X, values):
        if self.task == "severity":
            classes = list(self.encoder_.classes_)
            values = values[:, [classes.index(c) for c in CLASSES]]
        columns = route_target_columns(self.task)
        # A missing route carries no route-specific outcome information. Its
        # cross-fitted class prior would otherwise vary with the training fold.
        available = X["route_id"].notna() & X["route_id"].ne("unknown")
        return X.drop(columns=["route_id", TIME_COLUMN]).join(
            pd.DataFrame(values, index=X.index, columns=columns).where(available, axis=0))


def route_target_columns(task):
    return tuple(f"route_target_{label}" for label in CLASSES) if task == "severity" else ("route_target",)


def route_representation(variant, task):
    if variant == "route_target":
        return Representation(("route_id",), route_target_columns(task),
                              transformer=partial(RouteTarget, task), extra_inputs=(TIME_COLUMN,))
    return REPRESENTATIONS[variant]


REPRESENTATIONS = {
    "calendar_integer": Representation(CALENDAR, CALENDAR, transformer=FunctionTransformer),
    "calendar_cyclic": Representation(CALENDAR, cyclic_columns(CALENDAR),
                                      transformer=stateless(cyclic, keep_categories=False)),
    "calendar_splines": Representation(CALENDAR, tuple(c for col in CALENDAR for c in spline_columns(col)),
                                        transformer=PeriodicCalendar),
    "calendar_combined": Representation(CALENDAR, cyclic_columns(CALENDAR), CALENDAR,
                                         stateless(cyclic, keep_categories=True)),
    "minute_of_day": Representation(("hour",), ("minute_of_day",),
                                     transformer=stateless(finer_time, variant="minute"),
                                     extra_inputs=(TIME_COLUMN,)),
    "day_of_year": Representation(("month",), ("day_sin", "day_cos"),
                                   transformer=stateless(finer_time, variant="day"),
                                   extra_inputs=(TIME_COLUMN,)),
    "hour_weekday": Representation(("hour", "weekday"), categorical=("hour", "weekday", "hour_weekday"),
                                    transformer=stateless(hour_weekday)),
    "delay_clipped": Representation(("current_delay",), ("current_delay",),
                                     transformer=stateless(delay_transform, variant="clip")),
    "delay_log": Representation(("current_delay",), ("current_delay",),
                                 transformer=stateless(delay_transform, variant="log")),
    "delay_bins": Representation(("current_delay",), categorical=("current_delay_bin",),
                                  transformer=stateless(delay_transform, variant="bins")),
    "delay_missing": Representation(("current_delay",), ("current_delay", "current_delay_missing"),
                                     transformer=stateless(delay_transform, variant="missing")),
    "weather_extremes": Representation(tuple(WEATHER), tuple(EXTREMES),
                                       transformer=stateless(extreme_weather)),
    "weather_change": Representation(tuple(WEATHER), WEATHER_CHANGE,
                                     transformer=stateless(weather_change), extra_inputs=WEATHER_LAG),
    "route_frequency": Representation(("route_id",), ("route_frequency",), transformer=RouteFrequency),
    "route_onehot": Representation(("route_id",), transformer=RouteOneHot),
}
