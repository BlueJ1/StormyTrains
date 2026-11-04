"""Ensemble prediction and calibration using only outer-training week folds."""
from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

from cv import PROBA_COLUMNS, RESULTS, TARGETS, class_probabilities, score_fold
from data import TIME_COLUMN
from experiment_models import (CATBOOST, LIGHTGBM, LINEAR, MODELS, NEURAL, XGBOOST, fixed_factory,
                               flat_search_path, top_flat_factories)
from experiment_specs import Run, latest_run, tuning_reference
from folds import cv_folds, inner_folds
from metrics import CLASSES, EPS, class_log_loss, rmse
from tuning import SCORE


def predict_members(train, validation, task, factories, numeric, categorical):
    """Fit each member and return an array of shape (rows, models[, classes])."""
    features = numeric + categorical
    target = TARGETS[task]
    train = train[train[target].notna()]
    estimates = []
    for factory in factories:
        model = factory(numeric, categorical)
        model.fit(train[features], train[target])
        if task == "severity":
            prediction = class_probabilities(model, validation[features])
        else:
            prediction = model.predict(validation[features])
        estimates.append(prediction)
    return np.stack(estimates, axis=1)


def cross_fitted_members(train, task, factories, numeric, categorical):
    """Prediction and inner fold for each eligible outer-training row."""
    eligible = train[TARGETS[task]].notna()
    rows = train.loc[eligible]
    shape = (len(rows), len(factories))
    if task == "severity":
        shape += (len(CLASSES),)
    out = np.empty(shape)
    inner_fold = np.zeros(len(rows), dtype=int)
    for fold, training, validation in inner_folds(train):
        valid = train.loc[validation & eligible]
        index = rows.index.get_indexer(valid.index)
        out[index] = predict_members(train.loc[training], valid, task, factories,
                                     numeric, categorical)
        inner_fold[index] = fold
    if (inner_fold == 0).any():
        raise ValueError("some rows received no cross-fitted predictions")
    return rows[TARGETS[task]], out, inner_fold


def mean_prediction(predictions, *, geometric=False):
    if geometric:
        return softmax(np.log(np.clip(predictions, EPS, 1)).mean(axis=1), axis=1)
    return predictions.mean(axis=1)


def greedy_weights(y, predictions, inner_fold, task, *, iterations=30):
    """Caruana selection with replacement, judged by equally weighted fold scores."""
    score = class_log_loss if task == "severity" else rmse
    folds = [inner_fold == fold for fold in np.unique(inner_fold)]
    y = np.asarray(y)
    counts = np.zeros(predictions.shape[1], dtype=int)
    total = np.zeros_like(predictions[:, 0])
    for step in range(1, iterations + 1):
        candidates = (total[:, None, ...] + predictions) / step
        scores = [np.mean([score(y[fold], candidate[fold]) for fold in folds])
                  for candidate in candidates.swapaxes(0, 1)]
        best = int(np.argmin(scores))
        total += predictions[:, best]
        counts[best] += 1
    return counts / counts.sum()


def flatten_predictions(predictions):
    return predictions.reshape(len(predictions), -1)


def fit_stacker(y, predictions, task):
    model = LogisticRegression(max_iter=1000) if task == "severity" else LinearRegression()
    return model.fit(flatten_predictions(predictions), y)


def fit_temperature(y, proba):
    logits = np.log(np.clip(proba, EPS, 1))
    result = minimize_scalar(
        lambda log_t: class_log_loss(y, softmax(logits / np.exp(log_t), axis=1)),
        bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(proba, temperature):
    return softmax(np.log(np.clip(proba, EPS, 1)) / temperature, axis=1)


def fit_isotonic(y, proba):
    return [IsotonicRegression(out_of_bounds="clip").fit(proba[:, k], np.asarray(y) == label)
            for k, label in enumerate(CLASSES)]


def apply_isotonic(proba, models):
    calibrated = np.column_stack([model.predict(proba[:, k]) for k, model in enumerate(models)])
    total = calibrated.sum(axis=1, keepdims=True)
    return np.divide(calibrated, total, out=proba.copy(), where=total > 0)


def cv_ensemble(rows, prediction_time, task, factories_for_fold, numeric, categorical, method):
    """Cross-validate a combination of prespecified or previously tuned members."""
    metrics, predictions, fitted = [], [], {}
    target = TARGETS[task]
    for fold, training, validation in cv_folds(rows):
        train = rows.loc[training]
        valid = rows.loc[validation & rows[target].notna()]
        factories = factories_for_fold(fold)
        members = predict_members(train, valid, task, factories, numeric, categorical)
        if method in ("mean", "geometric"):
            prediction = mean_prediction(members, geometric=method == "geometric")
        else:
            y, inner, inner_fold = cross_fitted_members(train, task, factories, numeric, categorical)
            prediction, details = fit_combiner(method, y, inner, members, inner_fold, task)
            fitted[str(fold)] = details
        row, out = score_fold(valid, prediction_time, fold, task, int(train[target].notna().sum()),
                              prediction)
        metrics.append(row)
        predictions.append(out)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), fitted


def fit_combiner(method, y, inner, members, inner_fold, task):
    if method == "greedy":
        weights = greedy_weights(y, inner, inner_fold, task)
        return np.average(members, axis=1, weights=weights), {"weights": weights.tolist()}
    if method == "stack":
        stacker = fit_stacker(y, inner, task)
        inputs = flatten_predictions(members)
        prediction = (class_probabilities(stacker, inputs) if task == "severity"
                      else stacker.predict(inputs))
        return prediction, {"weights": stacker.coef_.tolist(),
                            "intercept": np.asarray(stacker.intercept_).tolist()}
    if method == "temperature":
        temperature = fit_temperature(y, inner[:, 0])
        return apply_temperature(members[:, 0], temperature), {"temperature": temperature}
    if method == "isotonic":
        models = fit_isotonic(y, inner[:, 0])
        return apply_isotonic(members[:, 0], models), {}
    raise ValueError(f"unknown ensemble method {method!r}")


def ensemble_recipe(run, prediction_time, results_dir):
    variant = run.variant
    model, weather = run.model, run.weather
    if variant == "seeds":
        factory = fixed_factory(model, weather, prediction_time)
        seeds = list(range(5))
        return "mean", lambda _: [partial(factory, seed=seed) for seed in seeds], {"seeds": seeds}
    if variant == "top5_flat":
        members = top_flat_factories(model, weather, prediction_time)
        source = flat_search_path(model, weather) / "search_results.csv"
        return "mean", lambda _: members, {"top": 5, "source": str(source.relative_to(RESULTS))}
    if variant == "top5_nested" or run.step == "6.2":
        source = latest_run(results_dir, tuning_reference(run))
        task, factory, _ = MODELS[model]
        details = {"source": str(source.relative_to(results_dir))}
        if variant == "top5_nested":
            table = pd.read_csv(source / "search_results.csv")
            table = table[table["prediction_time"].eq(prediction_time)]
            selected = {
                fold: rows.nsmallest(5, f"{SCORE[task]}_mean")[["config_id", "params"]]
                for fold, rows in table.groupby("outer_fold")
            }
            if any(len(configs) != 5 for configs in selected.values()):
                raise ValueError("nested bagging needs five configurations from each outer training set")
            details["config_ids"] = {str(fold): configs["config_id"].tolist()
                                     for fold, configs in selected.items()}
            return "mean", lambda fold: [partial(factory, **json.loads(p))
                                          for p in selected[fold]["params"]], details
        chosen = json.loads((source / "config.json").read_text())["chosen"][prediction_time]
        details["config_ids"] = {fold: choice["config_id"] for fold, choice in chosen.items()}
        return variant, lambda fold: [partial(factory, **chosen[str(fold)]["params"])], details

    task_index = LIGHTGBM.index(model)
    names = [model, XGBOOST[task_index]]
    if variant in ("greedy", "stack"):
        names.append(LINEAR[task_index])
    base = [fixed_factory(name, weather, prediction_time) for name in names]
    return variant, lambda _: base, {"members": names}


def saved_average(run, results_dir, prediction_times):
    """Combine the matched CV predictions directly, with no extra model fitting."""
    index = LIGHTGBM.index(run.model)
    task = MODELS[run.model][0]
    names = [("0.7", run.model), ("0.7", XGBOOST[index])]
    if run.variant in ("plus_linear", "plus_catboost", "plus_neural"):
        names.append(("0.7", LINEAR[index]))
    if run.variant in ("plus_catboost", "plus_neural"):
        names.append(("6.1", CATBOOST[index]))
    if run.variant == "plus_neural":
        names += [("7.1", NEURAL["7.1"][index]), ("7.4", NEURAL["7.4"][index]),
                  ("7.4", NEURAL["7.4"][index + 2])]
    sources = [latest_run(results_dir, Run(step, model, run.weather,
                                          calendar=run.calendar if step.startswith("7.") else ""))
               for step, model in names]
    row_key = ["prediction_time", "fold", "journey_id"]
    cohort = [*row_key, TIME_COLUMN, TARGETS[task]]
    prediction_columns = PROBA_COLUMNS if task == "severity" else ["predicted_delay"]
    estimates = []
    base = None
    for source in sources:
        frame = pd.read_parquet(source / "oof_predictions.parquet", columns=[*cohort, *prediction_columns])
        frame = frame[frame["prediction_time"].isin(prediction_times)].sort_values(row_key).reset_index(drop=True)
        if base is None:
            base = frame[cohort]
        elif not frame[cohort].equals(base):
            raise ValueError("ensemble members do not have identical CV prediction rows")
        estimates.append(frame[prediction_columns].to_numpy())
    averaged = mean_prediction(np.stack(estimates, axis=1), geometric=run.variant == "geometric")
    base = base.assign(**dict(zip(prediction_columns, averaged.T)))
    train_sizes = pd.read_csv(sources[0] / "fold_metrics.csv").set_index(["prediction_time", "fold"])["train_rows"]
    metrics, predictions = [], []
    for (pt, fold), frame in base.groupby(["prediction_time", "fold"]):
        estimate = frame[prediction_columns].to_numpy()
        estimate = estimate[:, 0] if task == "delay" else estimate
        row, out = score_fold(frame, pt, fold, task, int(train_sizes.loc[(pt, fold)]), estimate)
        metrics.append(row)
        predictions.append(out)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), sources
