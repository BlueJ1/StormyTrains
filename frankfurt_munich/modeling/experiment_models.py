"""Factories and pinned flat-search inputs for the advanced experiments."""
from __future__ import annotations

import json
from functools import cache, partial

import pandas as pd

from cv import RESULTS
from decomposed import (make_cancellation_then_class, make_cancellation_then_distribution,
                        make_cancellation_then_ordinal, make_change_then_level, make_mixture_of_experts)
from distributions import make_lightgbmlss, make_ngboost, make_quantile_forest
from extra_models import log_shifted, make_catboost, make_ebm
from models import make_lightgbm, make_lightgbm_classifier
from run_linear_models import RUNS
from run_tuned_models import MODELS as TUNED_MODELS
from tuning import SCORE, SPACES, fixed_configurations

LINEAR = ("linear_regression", "logistic_regression")
LIGHTGBM = ("lightgbm", "lightgbm_classifier")
XGBOOST = ("xgboost", "xgboost_classifier")
CATBOOST = ("catboost", "catboost_classifier")
EBM = ("ebm", "ebm_classifier")
DISTRIBUTIONS = {"ngboost": make_ngboost, "quantile_forest": make_quantile_forest,
                 "lightgbmlss": make_lightgbmlss}
DISTRIBUTION_CLASSIFIERS = tuple(f"{model}_severity" for model in DISTRIBUTIONS)
NEURAL = {"7.1": ("mlp", "mlp_classifier"), "7.2": ("multitask", "multitask_classifier"),
          "7.3": ("tabgns", "tabgns_classifier"),
          "7.4": ("realmlp", "realmlp_classifier", "tabm", "tabm_classifier"),
          "7.5": ("sequence_gru", "sequence_gru_classifier")}

FLAT_TIMESTAMPS = {
    "lightgbm": "20261026-092616",
    "lightgbm_weather": "20261026-093239",
    "lightgbm_classifier": "20261031-120618",
    "lightgbm_classifier_weather": "20261031-122533",
    "xgboost": "20261026-093649",
    "xgboost_weather": "20261026-094318",
    "xgboost_classifier": "20261031-123804",
    "xgboost_classifier_weather": "20261031-125613",
}

def make_neural(numeric, categorical, *, family, task, **params):
    if family == "tabgns":
        from gated_neural import GatedMLP
        return GatedMLP(numeric, categorical, task=task, **params)
    if family == "sequence_gru":
        from sequence_model import SequenceMLP
        return SequenceMLP(numeric, categorical, task=task, **params)
    from neural import MultiTaskMLP, TabKitModel, TorchMLP
    if family in ("mlp", "multitask"):
        cls = TorchMLP if family == "mlp" else MultiTaskMLP
        return cls(numeric, categorical, task=task, **params)
    return TabKitModel(numeric, categorical, task=task, family=family, **params)


MODELS = {
    **{name: (task, model, None) for name, task, model, weather in RUNS if not weather},
    **{name: TUNED_MODELS[name] for name in (*LIGHTGBM, *XGBOOST)},
    **{name: (task, partial(factory, task=task), space)
       for names, factory, space in [(CATBOOST, make_catboost, SPACES["catboost"]),
                                     (EBM, make_ebm, None)]
       for name, task in zip(names, ("delay", "severity"))},
    **{name: ("delay", factory, None) for name, factory in DISTRIBUTIONS.items()},
    **{f"{name}_severity": ("severity", partial(make_cancellation_then_distribution,
                                               delay_factory=factory), None)
       for name, factory in DISTRIBUTIONS.items()},
}
for models in NEURAL.values():
    for name in models:
        family = name.removesuffix("_classifier")
        task = "delay" if family == name else "severity"
        space = SPACES["mlp"] if family in ("mlp", "multitask", "sequence_gru") else None
        MODELS[name] = task, partial(make_neural, family=family, task=task), space


def flat_search_path(model, weather):
    key = model + ("_weather" if weather else "")
    return RESULTS / key / FLAT_TIMESTAMPS[key] if key in FLAT_TIMESTAMPS else None


@cache
def flat_choices(model, weather):
    source = flat_search_path(model, weather)
    if source is None:
        return {}
    task, factory, _ = MODELS[model]
    return fixed_configurations(source / "config.json", task=task, model=factory.__name__,
                                weather=weather)


def fixed_factory(model, weather, prediction_time):
    choices = flat_choices(model, weather)
    factory = MODELS[model][1]
    return partial(factory, **choices[prediction_time]["params"]) if choices else factory


def top_flat_factories(model, weather, prediction_time, *, count=5):
    path = flat_search_path(model, weather) / "search_results.csv"
    search = pd.read_csv(path)
    task, factory, _ = MODELS[model]
    top = search.loc[search["prediction_time"].eq(prediction_time)].nsmallest(count, f"{SCORE[task]}_mean")
    if len(top) != count:
        raise ValueError(f"{path} has fewer than {count} configurations for {prediction_time}")
    return [partial(factory, **json.loads(params)) for params in top["params"]]


def variant_model(variant, fixed_params, numeric, categorical, **sampled_params):
    params = fixed_params | sampled_params
    classifier = partial(make_lightgbm_classifier, **params, objective=None)
    regressor = partial(make_lightgbm, **params)
    if variant == "cancellation_then_class":
        return make_cancellation_then_class(numeric, categorical, classifier)
    if variant == "cancellation_then_ordinal":
        return make_cancellation_then_ordinal(numeric, categorical, classifier)
    if variant == "mixture_of_experts":
        return make_mixture_of_experts(numeric, categorical, classifier, regressor)
    if variant == "change_then_level":
        return make_change_then_level(numeric, categorical, regressor)
    if variant == "huber":
        return make_lightgbm(numeric, categorical, **params, objective="huber")
    if variant == "log_shift":
        return log_shifted(regressor, numeric, categorical)
    raise ValueError(f"unknown model variant {variant!r}")
