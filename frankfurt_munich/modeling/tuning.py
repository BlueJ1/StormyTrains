"""Random search over hyperparameters, scored by the shared cross-validation.

For each prediction time:

1. N_CONFIGS configurations are drawn from the model's search space with
   scikit-learn's ParameterSampler (seed SEED). The draw depends only on the
   space and the seed, so the railway-only and weather runs, and every
   prediction time, try the same configurations.
2. Every configuration is cross-validated on the 5 week folds exactly as in
   cv.py (cross_validate), so its score is the mean of its 5 fold scores.
3. The configuration with the lowest mean score (RMSE for `delay`, log loss
   for `severity`) is chosen and its fold scores and validation predictions
   are the run's result.

This is a flat search: the folds that choose the configuration also score it,
so the chosen configuration's score is optimistic (the best of N_CONFIGS on
the same folds). search_results.csv keeps every configuration's fold scores so
that the size of that optimism can be judged.

run_search writes results/<name>/<timestamp>/ like cv.run_cv, plus
search_results.csv (one row per prediction time and configuration). Its
config.json holds the search space, the chosen configuration per prediction
time and the features.
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler

from cv import RESULTS, TARGETS, cross_validate, fit_fold, save_run
from data import PREDICTION_TIMES, load_rows
from features import feature_columns
from folds import cv_folds, inner_folds

N_CONFIGS = 50
FEATURE_SCREEN_CONFIGS = 3
SEED = 42
SCORE = {"delay": "rmse", "severity": "log_loss"}  # lower is better


class LogUniformInt:
    """Integers drawn log-uniformly from [low, high]."""

    def __init__(self, low: int, high: int):
        self.low, self.high = low, high

    def rvs(self, random_state=None):
        return int(round(loguniform(self.low, self.high).rvs(random_state=random_state)))

    def __repr__(self):
        return f"LogUniformInt({self.low}, {self.high})"


def between(low: float, high: float):
    """Uniform on [low, high] (scipy's uniform takes low and width)."""
    return uniform(low, high - low)


SPACES = {
    "mlp": {
        "depth": [2, 3, 4],
        "width": [64, 128, 256],
        "dropout": [0.0, 0.1, 0.2],
        "weight_decay": loguniform(1e-6, 1e-2),
        "learning_rate": loguniform(3e-4, 3e-3),
        "epochs": [12, 24, 48],
    },
    "catboost": {
        "iterations": LogUniformInt(100, 1500),
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": loguniform(0.005, 0.3),
        "l2_leaf_reg": loguniform(0.1, 100),
        "random_strength": loguniform(0.01, 10),
        "bagging_temperature": between(0, 2),
    },
    "random_forest": {
        "max_features": between(0.1, 1.0),
        "min_samples_leaf": LogUniformInt(1, 50),
        "max_depth": [None, 8, 16, 32],
        "max_samples": between(0.5, 1.0),
    },
    "xgboost": {
        "n_estimators": LogUniformInt(100, 2000),
        "learning_rate": loguniform(0.005, 0.3),
        "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "min_child_weight": loguniform(1, 100),
        "subsample": between(0.5, 1.0),
        "colsample_bytree": between(0.3, 1.0),
        "reg_lambda": loguniform(1e-3, 100),
        "reg_alpha": loguniform(1e-3, 10),
    },
    "lightgbm": {
        "n_estimators": LogUniformInt(100, 2000),
        "learning_rate": loguniform(0.005, 0.3),
        "num_leaves": LogUniformInt(4, 256),
        "min_child_samples": LogUniformInt(5, 200),
        "subsample": between(0.5, 1.0),
        "colsample_bytree": between(0.3, 1.0),
        "reg_lambda": loguniform(1e-3, 100),
        "reg_alpha": loguniform(1e-3, 10),
        "cat_smooth": loguniform(1, 100),
    },
    "svr": {
        "C": loguniform(0.1, 100),
        "epsilon": loguniform(0.01, 1),
        "gamma": loguniform(1e-4, 1),
    },
}


def sample_configs(space: dict, n: int = N_CONFIGS, seed: int = SEED) -> list[dict]:
    """The n configurations of a search space, the same on every call."""
    configs = list(ParameterSampler(space, n_iter=n, random_state=seed))
    # numpy scalars -> plain Python, so configurations can be written as JSON
    return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in c.items()}
            for c in configs]


def search(rows: pd.DataFrame, prediction_time: str, task: str, factory, configs: list[dict],
           numeric: list[str], categorical: list[str], *, splits=cv_folds, prior=None):
    """Cross-validate every configuration at one prediction time.

    Return (search table, fold metrics of the chosen configuration, its
    validation predictions). The search table has one row per configuration:
    its parameters (JSON), the score of each fold, their mean and standard
    deviation, the seconds taken, and `chosen`.

    `prior` is the result of an earlier search of the same rows, model, columns
    and splits over the first configurations of `configs`; those are reused
    instead of being fitted again.
    """
    score = SCORE[task]
    table = []
    folds_for_rows = tuple(splits(rows))
    best_score = np.inf
    if prior is not None:
        prior_table, best_metrics, best_predictions = prior
        if prior_table["params"].tolist() != [json.dumps(c) for c in configs[:len(prior_table)]]:
            raise ValueError("prior search must cover the first configurations of this search")
        table = prior_table.drop(columns="chosen").to_dict("records")
        best = int(prior_table.loc[prior_table["chosen"], "config_id"].item())
        best_score = float(prior_table.loc[prior_table["chosen"], f"{score}_mean"].item())
    for config_id, params in enumerate(configs[len(table):], start=len(table)):
        start = time.perf_counter()
        cpu_start = time.process_time()
        metrics, predictions = cross_validate(rows, prediction_time, task,
                                              partial(factory, **params), numeric, categorical,
                                              splits=lambda _: folds_for_rows)
        seconds = time.perf_counter() - start
        folds = metrics.set_index("fold")[score]
        mean = folds.mean()
        if mean < best_score:
            best_score = mean
            best = config_id
            best_metrics, best_predictions = metrics, predictions
        table.append({"prediction_time": prediction_time, "config_id": config_id,
                      "params": json.dumps(configs[config_id]),
                      **{f"{score}_fold{f}": v for f, v in folds.items()},
                      f"{score}_mean": mean, f"{score}_std": folds.std(),
                      "seconds": round(seconds, 1), "cpu_seconds": time.process_time() - cpu_start})
    if not np.isfinite(best_score):
        raise ValueError("search needs at least one configuration with a finite score")
    table = pd.DataFrame(table)
    table["chosen"] = table["config_id"] == best
    return table, best_metrics, best_predictions.assign(config_id=best)


def nested_search(rows, prediction_time, task, factory, configs, numeric, categorical, *,
                  variants=None):
    """Search features and configurations on inner weeks, then score the outer fold."""
    tables, metrics, predictions = [], [], []
    for fold, train, validation in cv_folds(rows):
        outer_train = rows.loc[train]
        inner = tuple(inner_folds(outer_train))
        splits = lambda _: inner
        chosen_variant = None
        model, num, cat = factory, numeric, categorical
        screen = None
        if variants:
            screens = screen_feature_variants(outer_train, prediction_time, task,
                                              variants, configs, splits)
            chosen_variant = best_variant(screens, task)
            model, num, cat = variants[chosen_variant]
            screen = screens[chosen_variant]
        table, _, _ = search(outer_train, prediction_time, task, model, configs,
                             num, cat, splits=splits, prior=screen)
        best = int(table.loc[table["chosen"], "config_id"].item())
        scores, out = fit_fold(outer_train, rows.loc[validation], prediction_time, fold,
                               task, partial(model, **configs[best]), num, cat)
        if chosen_variant is not None:
            table["feature_variant"] = chosen_variant
        tables.append(table.assign(outer_fold=fold))
        metrics.append(scores)
        predictions.append(out.assign(config_id=best))
    return pd.concat(tables, ignore_index=True), pd.DataFrame(metrics), pd.concat(
        predictions, ignore_index=True)


def screen_feature_variants(rows, prediction_time, task, variants, configs, splits):
    """Search each variant over the same first three configurations."""
    screening_configs = configs[:FEATURE_SCREEN_CONFIGS]
    return {name: search(rows, prediction_time, task, factory, screening_configs,
                         numeric, categorical, splits=splits)
            for name, (factory, numeric, categorical) in variants.items()}


def best_variant(screens, task):
    scores = {name: table[f"{SCORE[task]}_mean"].min() for name, (table, _, _) in screens.items()}
    return min(scores, key=scores.__getitem__)


def select_feature_variant(rows, prediction_time, task, variants, configs, splits):
    """Rank each variant by its best inner score over the same three configurations."""
    return best_variant(screen_feature_variants(rows, prediction_time, task, variants,
                                                configs, splits), task)


def run_search(name: str, task: str, factory, space: dict, *, weather: bool = False,
               prediction_times=PREDICTION_TIMES, n_configs: int = N_CONFIGS,
               feature_set: str = "base", results_dir=RESULTS) -> pd.DataFrame:
    """Random search at every prediction time; save the chosen configurations' results.

    weather=True adds the weather features (features.WEATHER); it is keyword-only.
    """
    configs = sample_configs(space, n_configs)
    tables, all_metrics, all_predictions, chosen, config_features = [], [], [], {}, {}
    for prediction_time in prediction_times:
        numeric, categorical = feature_columns(prediction_time, weather, feature_set=feature_set)
        config_features[prediction_time] = {"numeric": numeric, "categorical": categorical}
        rows = load_rows(prediction_time, weather=weather)
        table, metrics, predictions = search(rows, prediction_time, task, factory, configs,
                                             numeric, categorical)
        best = int(table.loc[table["chosen"], "config_id"].item())
        chosen[prediction_time] = {"config_id": best, "params": configs[best]}
        tables.append(table)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        print(f"  {prediction_time}: {SCORE[task]} {metrics[SCORE[task]].mean():.3f}, "
              f"{table['seconds'].sum() / 60:.1f} min searching", flush=True)

    config = {"name": name, "task": task, "target": TARGETS[task],
              "model": factory.__name__, "weather": weather, "feature_set": feature_set,
              "evaluation": "exploratory",
              "search": {"n_configs": n_configs, "seed": SEED,
                                             "space": {k: repr(v) for k, v in space.items()},
                         "selection": f"lowest mean {SCORE[task]} over the folds"},
              "chosen": chosen, "features": config_features}
    out, summary = save_run(results_dir, name, config, pd.concat(all_metrics, ignore_index=True),
                            pd.concat(all_predictions, ignore_index=True))
    pd.concat(tables, ignore_index=True).to_csv(out / "search_results.csv", index=False)
    return summary


def fixed_configurations(path: Path, *, task: str, model: str, weather: bool) -> dict:
    """Read the per-time configurations from a matching flat search."""
    source = json.loads(path.read_text())
    expected = {"task": task, "model": model, "weather": weather}
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{path} does not match the model, task and weather arm")
    chosen = source["chosen"]
    if any("params" not in value for value in chosen.values()):
        raise ValueError(f"{path} does not contain a flat search")
    return chosen
