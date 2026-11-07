"""Whole-feature-group ablations and selection inside the outer training weeks."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from cv import TARGETS, class_probabilities, cross_validate, fit_fold, score_fold
from folds import cv_folds, inner_folds
from metrics import class_log_loss, rmse
from tuning import SCORE


def columns_for(groups, selected):
    """Return the two feature lists for named, disjoint groups."""
    numeric, categorical = [], []
    for group, (num, cat) in groups.items():
        if group in selected:
            numeric.extend(num)
            categorical.extend(cat)
    return numeric, categorical


def ablations(groups):
    """Add one group to the timetable or remove one group from the full set."""
    all_groups = frozenset(groups)
    for group in groups:
        if group != "timetable":
            yield f"add_{group}", frozenset(("timetable", group))
        yield f"drop_{group}", all_groups - {group}


def greedy_groups(groups, evaluate: Callable, direction: str):
    """Keep timetable and stop when the next addition/removal fails to improve."""
    if direction not in ("forward", "backward"):
        raise ValueError(f"invalid direction {direction!r}")
    all_groups = frozenset(groups)
    selected = frozenset(("timetable",)) if direction == "forward" else all_groups
    current_score = evaluate(selected)
    while True:
        if direction == "forward":
            candidates = [selected | {g} for g in groups if g not in selected]
        else:
            candidates = [selected - {g} for g in groups if g in selected and g != "timetable"]
        if not candidates:
            return selected
        score, best = min((evaluate(candidate), i) for i, candidate in enumerate(candidates))
        if score >= current_score:
            return selected
        selected, current_score = candidates[best], score


def nested_groups(rows, prediction_time, task, factory, groups, direction):
    """Return outer scores, predictions and inner choices for one direction."""
    metrics, predictions, choices = [], [], []
    for fold, training, validation in cv_folds(rows):
        outer_train = rows.loc[training]
        splits = tuple(inner_folds(outer_train))
        scores = {}

        def evaluate(selected):
            if selected not in scores:
                columns = columns_for(groups, selected)
                inner, _ = cross_validate(outer_train, prediction_time, task, factory,
                                          *columns, splits=lambda _: splits)
                scores[selected] = inner[SCORE[task]].mean()
            return scores[selected]

        selected = greedy_groups(groups, evaluate, direction)
        columns = columns_for(groups, selected)
        row, out = fit_fold(outer_train, rows.loc[validation], prediction_time, fold,
                            task, factory, *columns)
        metrics.append(row)
        predictions.append(out)
        choices.append({"prediction_time": prediction_time, "outer_fold": fold,
                        "groups": sorted(selected), "inner_score": evaluate(selected)})
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), choices


def group_permutation(model, X, y, task, groups, *, repeats=5, seed=42):
    """Jointly permute columns of one group on validation rows, leaving its correlations intact."""
    def score(frame):
        if task == "severity":
            return class_log_loss(y, class_probabilities(model, frame))
        return rmse(y, model.predict(frame))

    base = score(X)
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(X)) for _ in range(repeats)]
    output = []
    for group, (numeric, categorical) in groups.items():
        columns = list(numeric) + list(categorical)
        perturbed = X.copy()
        for repeat, perm in enumerate(permutations):
            shuffled = X.iloc[perm][columns].set_axis(X.index)
            perturbed[columns] = shuffled
            output.append({"group": group, "repeat": repeat,
                           "score_increase": score(perturbed) - base})
    return pd.DataFrame(output)


def lightgbm_contributions(model, X):
    """Per-feature mean absolute TreeSHAP values over validation rows and classes."""
    encoded = model.named_steps["categories"].transform(X)
    booster = model.named_steps["model"].booster_
    contributions = np.asarray(booster.predict(encoded, pred_contrib=True, num_threads=1))
    by_class = contributions.reshape(len(X), -1, encoded.shape[1] + 1)
    return pd.DataFrame({"feature": encoded.columns,
                         "mean_abs_shap": np.abs(by_class[:, :, :-1]).mean(axis=(0, 1))})


def cv_importance(rows, prediction_time, task, factory, groups):
    metrics, predictions, permutations, contributions = [], [], [], []
    target = TARGETS[task]
    rows = rows[rows[target].notna()]
    numeric, categorical = columns_for(groups, groups)
    columns = numeric + categorical
    for fold, training, validation in cv_folds(rows):
        train, valid = rows.loc[training], rows.loc[validation]
        model = factory(numeric, categorical).fit(train[columns], train[target])
        X = valid[columns]
        prediction = class_probabilities(model, X) if task == "severity" else model.predict(X)
        row, out = score_fold(valid, prediction_time, fold, task, len(train), prediction)
        metrics.append(row)
        predictions.append(out)
        permutations.append(group_permutation(model, X, valid[target], task, groups)
                            .assign(prediction_time=prediction_time, fold=fold))
        contributions.append(lightgbm_contributions(model, X)
                             .assign(prediction_time=prediction_time, fold=fold))
    return (pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True),
            pd.concat(permutations, ignore_index=True), pd.concat(contributions, ignore_index=True))
