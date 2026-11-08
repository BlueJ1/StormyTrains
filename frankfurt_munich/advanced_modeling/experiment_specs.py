"""Registered experiment names and pre-listed variants."""
from __future__ import annotations

from dataclasses import dataclass

import _paths  # noqa: F401  shared modelling code and dataset builders
from corridor_transfer import METHODS as CORRIDOR_METHODS
from experiment_models import CATBOOST, DISTRIBUTION_CLASSIFIERS, DISTRIBUTIONS, EBM, LIGHTGBM, LINEAR, NEURAL, XGBOOST
from features import CURRENT_DELAY_TIMES, PHASE2_GROUPS, feature_columns, feature_groups
from group_selection import ablations
from representations import route_representation

DECOMPOSED = {"4.1": (LIGHTGBM[1], "cancellation_then_class"),
              "4.3": (LIGHTGBM[1], "cancellation_then_ordinal"),
              "4.4": (LIGHTGBM[0], "mixture_of_experts"),
              "4.5": (LIGHTGBM[0], "change_then_level")}
SEARCH_CONFIGS = 20


@dataclass(frozen=True)
class Run:
    step: str
    model: str
    weather: bool = False
    variant: str = ""
    diagnostic: bool = False
    nested: bool = False
    calendar: str = ""
    n_configs: int = SEARCH_CONFIGS

    @property
    def name(self):
        suffix = ("_weather" if self.weather else "") + (f"_{self.variant}" if self.variant else "")
        if self.calendar:
            suffix += f"_{self.calendar}"
        return f"advanced_{self.step.replace('.', '_')}_{self.model}{suffix}"


def registered_runs(step: str):
    if step == "0.7":
        for model in ("class_shares", "training_mean"):
            yield Run(step, model)
        for model in (*LINEAR, *LIGHTGBM, *XGBOOST):
            for weather in (False, True):
                yield Run(step, model, weather)
                yield Run(step, model, weather, variant="recorded_route", diagnostic=True)
        return
    if step == "0.8":
        for model in (*LIGHTGBM, *XGBOOST):
            for weather in (False, True):
                yield Run(step, model, weather, nested=True)
        return
    if step == "3.1":
        variants = [name for name, _ in ablations(feature_groups("1h", weather=True))]
        for model in (*LINEAR, *LIGHTGBM):
            for variant in ("full", "timetable", *variants):
                yield Run(step, model, weather=True, variant=variant)
        return
    if step == "3.2":
        for model in LIGHTGBM:
            yield Run(step, model, weather=True)
        return
    if step == "3.3":
        for model in LINEAR:
            variants = ("full", "forward", "backward", "l1") if model == "logistic_regression" else (
                "full", "forward", "backward")
            for weather in (False, True):
                for variant in variants:
                    yield Run(step, model, weather=weather, variant=variant)
        return
    if step == "3.4":
        for model in LIGHTGBM:
            for variant in ("full", "temperature", "wind", "snow", "weather_extremes"):
                yield Run(step, model, weather=True, variant=variant)
        return
    if step in DECOMPOSED:
        model, variant = DECOMPOSED[step]
        for weather in (False, True):
            yield Run(step, model, weather=weather, variant=variant)
        return
    if step in ("4.2", "6.5"):
        for model in DISTRIBUTION_CLASSIFIERS if step == "4.2" else DISTRIBUTIONS:
            for weather in (False, True):
                yield Run(step, model, weather)
        return
    if step.startswith("5.") or step == "6.2":
        yield from ensemble_runs(step)
        return
    if step in ("6.1", "6.3", "6.6"):
        models = {"6.1": CATBOOST, "6.3": EBM, "6.6": LIGHTGBM[:1]}[step]
        variants = ("huber", "log_shift") if step == "6.6" else ("",)
        for model in models:
            for weather in (False, True):
                for variant in variants:
                    yield Run(step, model, weather=weather, variant=variant, nested=step == "6.1")
        return
    if step in NEURAL:
        for model in NEURAL[step]:
            for weather in (False, True):
                yield Run(step, model, weather=weather, nested=step in ("7.1", "7.2", "7.5"))
        return
    if step in ("8.2", "8.4"):
        for model in LIGHTGBM:
            for weather in (False, True):
                yield Run(step, model, weather, variant="pooled" if step == "8.2" else "network_transfer")
        return
    if step == "8.3":
        for model in LIGHTGBM:
            for method in CORRIDOR_METHODS:
                yield Run(step, model, variant=method)
        return
    models = LIGHTGBM if step == "1.5" else (*LINEAR, *LIGHTGBM)
    for variant in VARIANTS[step]:
        for model in models:
            if variant != "hour_weekday" or model in LINEAR:
                if variant == "route_onehot" and model in LINEAR:
                    continue
                if variant == "delay_missing" and model in LINEAR:
                    continue
                yield Run(step, model, weather=step == "1.5", variant=variant)


VARIANTS = {
    "1.1": ("calendar_integer", "calendar_cyclic", "calendar_splines", "calendar_combined"),
    "1.2": ("minute_of_day", "day_of_year", "hour_weekday"),
    "1.3": ("route_native", "route_onehot", "route_frequency", "route_target"),
    "1.4": ("delay_clipped", "delay_log", "delay_bins", "delay_missing"),
    "1.5": ("weather_extremes", "weather_change"),
    "2.1": ("recent_reliability",),
    "2.2": ("planned_timetable",),
    "2.3": ("holidays",),
    "2.4": ("delay_dynamics",),
    "2.5": ("network_state",),
    "2.all": ("all_groups",),
}


def ensemble_runs(step):
    for weather in (False, True):
        if step == "6.2":
            for model in (LIGHTGBM[1], XGBOOST[1], CATBOOST[1]):
                for variant in ("temperature", "isotonic"):
                    yield Run(step, model, weather, variant)
            continue
        if step == "5.2":
            for model in (*LIGHTGBM, *XGBOOST):
                for variant in ("seeds", "top5_flat", "top5_nested"):
                    yield Run(step, model, weather, variant)
            continue
        for model in LIGHTGBM:
            variants = ("mean", "plus_linear", "plus_catboost", "plus_neural") if step == "5.1" else (
                "greedy",) if step == "5.3" else ("stack",)
            for variant in variants:
                yield Run(step, model, weather, variant)
            if step == "5.1" and model == LIGHTGBM[1]:
                yield Run(step, model, weather, "geometric")


def latest_run(results_dir, run):
    root = results_dir / run.name
    paths = [p for p in root.glob("*") if (p / "config.json").is_file()]
    if not paths:
        raise FileNotFoundError(f"run the reference first: {root}")
    return max(paths)


def tuning_reference(run):
    """The single-booster nested reference for bagging or calibration."""
    step = "6.1" if run.model in CATBOOST else "0.8"
    return Run(step, run.model, run.weather, nested=True)


LATE_STEPS = ("1.4", "2.4", "4.5", "7.5", "8.4")
LATE_VARIANTS = (*VARIANTS["1.4"], "delay_dynamics")


AUTO_VARIANTS = ("auto_groups", "auto_representations")


def nested_feature_variants(run, prediction_time, task, factory):
    if run.variant not in AUTO_VARIANTS:
        return None
    numeric, categorical = feature_columns(prediction_time, run.weather)
    variants = {"base": (factory, numeric, categorical)}
    late = prediction_time in CURRENT_DELAY_TIMES
    if run.variant == "auto_groups":
        groups = [group for group in PHASE2_GROUPS if late or group != "delay_dynamics"]
        for group in (*groups, "all_groups"):
            num, cat = feature_columns(prediction_time, run.weather, feature_set=group)
            variants[group] = factory, num, cat
    else:
        for step in ("1.1", "1.2", "1.3", "1.4"):
            if step == "1.4" and not late:
                continue
            for name in VARIANTS[step]:
                if name == "hour_weekday":
                    continue  # This comparison is registered only for linear models.
                cat = [*categorical, "route_id"] if step == "1.3" else categorical
                variants[name] = ((factory, numeric, cat) if name == "route_native"
                                  else route_representation(name, task).wrap(factory, numeric, cat))
    return variants
