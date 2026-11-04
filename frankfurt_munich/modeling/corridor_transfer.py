"""CV-only transfer experiments across supplied corridor prediction rows.

The module does not load or extract data.  Callers supply the target corridor's
CV rows and already-built CV rows for the other corridors.  Every outer and
inner source fit is restricted to the weeks present in the corresponding
target training partition; source journeys matching validation journey IDs are
removed even when their corridor-specific origin falls in a different week.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from cv import TARGETS, class_probabilities, score_fold
from data import PREDICTION_TIMES, TIME_COLUMN
from folds import cv_folds, inner_folds, week_of
from metrics import CLASSES
from models import CategoryCodes, make_lightgbm, make_lightgbm_classifier

METHODS = ("pooled", "init_model", "stacked")
BASE_KEYS = ("journey_id", "prediction_time", TIME_COLUMN)


def _week_keys(values: pd.Series) -> pd.Series:
    """Normalize local dates or supplied week stamps to comparable Monday keys."""
    if isinstance(values.dtype, pd.PeriodDtype):
        times = values.dt.to_timestamp()
    else:
        times = pd.to_datetime(values, errors="coerce")
    if times.dt.tz is not None:
        times = times.dt.tz_localize(None)
    weeks = week_of(times)
    return weeks.dt.strftime("%Y-%m-%d").astype("string")


def _check_rows(target_rows: pd.DataFrame, source_rows: pd.DataFrame,
                prediction_time: str, task: str, numeric: list[str],
                categorical: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task not in TARGETS:
        raise ValueError("task must be 'severity' or 'delay'")
    if prediction_time not in PREDICTION_TIMES:
        raise ValueError(f"unknown prediction time {prediction_time!r}")
    if set(numeric) & set(categorical):
        raise ValueError("numeric and categorical feature columns must be disjoint")
    feature_columns = [*numeric, *categorical]
    required = {*BASE_KEYS, "corridor", *feature_columns, *TARGETS.values()}
    missing_target = (required | {"fold"}) - set(target_rows.columns)
    if missing_target:
        raise ValueError(f"target rows are missing columns: {sorted(missing_target)}")
    missing_source = required - set(source_rows.columns)
    if missing_source:
        raise ValueError(f"source rows are missing columns: {sorted(missing_source)}")

    target = target_rows.loc[target_rows["prediction_time"].astype(str).eq(prediction_time)].copy()
    source = source_rows.loc[source_rows["prediction_time"].astype(str).eq(prediction_time)].copy()
    if target.empty:
        raise ValueError(f"no target rows for prediction time {prediction_time!r}")
    if target.duplicated(["journey_id", "prediction_time"]).any():
        raise ValueError("target rows must have one row per journey and prediction time")
    if target["fold"].isna().any():
        raise ValueError("target CV rows must have assigned folds")
    if target["journey_id"].isna().any() or source["journey_id"].isna().any():
        raise ValueError("journey_id cannot be missing")
    if task == "severity":
        unexpected = (set(target[TARGETS[task]].dropna().astype(str))
                      | set(source[TARGETS[task]].dropna().astype(str))) - set(CLASSES)
        if unexpected:
            raise ValueError(
                f"severity labels are outside the fixed class order: {sorted(unexpected)}"
            )
    return target.reset_index(drop=True), source.reset_index(drop=True)


def _source_training_rows(source_rows: pd.DataFrame, target_training: pd.DataFrame,
                          target_validation: pd.DataFrame, prediction_time: str,
                          task: str) -> pd.DataFrame:
    """Select source records available under this target training partition."""
    allowed_weeks = set(_week_keys(target_training[TIME_COLUMN]).dropna())
    source_weeks = _week_keys(source_rows[TIME_COLUMN])
    selected = source_rows.loc[
        source_rows["prediction_time"].astype(str).eq(prediction_time)
        & source_weeks.isin(allowed_weeks)
    ].copy()

    # Auxiliary source labels can arrive in a later week than the run itself.
    # If the builder provides these timestamps as week fields, all must also
    # belong to weeks represented in the current target training partition.
    for column in ("run_week", "destination_week", "outcome_week"):
        if column in selected:
            selected = selected.loc[_week_keys(selected[column]).isin(allowed_weeks)]

    validation_ids = set(target_validation["journey_id"].astype("string").dropna())
    duplicate = selected["journey_id"].astype("string").isin(validation_ids)

    # A run can cross a week boundary between corridor origins.  When both
    # extracts carry the canonical run week, also exclude source rows from a
    # validation run-week, even if their corridor-specific journey key differs.
    if "run_week" in selected and "run_week" in target_validation:
        validation_run_weeks = set(_week_keys(target_validation["run_week"]).dropna())
        duplicate |= _week_keys(selected["run_week"]).isin(validation_run_weeks)

    selected = selected.loc[~duplicate]
    label = TARGETS[task]
    return selected.loc[selected[label].notna()].reset_index(drop=True)


def _check_severity_classes(rows: pd.DataFrame, *, name: str) -> None:
    present = set(rows[TARGETS["severity"]].dropna().astype(str))
    missing = set(CLASSES) - present
    if missing:
        raise ValueError(f"{name} severity training rows are missing classes: {sorted(missing)}")


def _fit(model, rows: pd.DataFrame, features: list[str], task: str):
    label = TARGETS[task]
    fit_rows = rows.loc[rows[label].notna()]
    if fit_rows.empty:
        raise ValueError("a corridor transfer fit has no labeled training rows")
    if task == "severity":
        _check_severity_classes(fit_rows, name="source/target")
    model.fit(fit_rows[features], fit_rows[label])
    return model, len(fit_rows)


def _predict(model, rows: pd.DataFrame, features: list[str], task: str) -> np.ndarray:
    X = rows[features]
    return class_probabilities(model, X) if task == "severity" else np.asarray(model.predict(X))


def _fit_with_init_model(source_rows: pd.DataFrame, target_training: pd.DataFrame,
                         numeric: list[str], categorical: list[str], params: Mapping,
                         task: str, factory: Callable) -> tuple[object, int, int]:
    """Pretrain and continue with a shared, training-fitted category schema."""
    label = TARGETS[task]
    source_fit = source_rows.loc[source_rows[label].notna()]
    target_fit = target_training.loc[target_training[label].notna()]
    if source_fit.empty or target_fit.empty:
        raise ValueError("init_model transfer needs labeled source and target training rows")
    if task == "severity":
        _check_severity_classes(source_fit, name="source")
        _check_severity_classes(target_fit, name="target")

    features = [*numeric, *categorical]
    codes = CategoryCodes(categorical).fit(pd.concat(
        [source_fit[features], target_fit[features]], ignore_index=True))
    source_X = codes.transform(source_fit[features])
    target_X = codes.transform(target_fit[features])

    # Use the project factory's LightGBM estimator while fitting the one shared
    # CategoryCodes transformer explicitly; otherwise each pipeline would
    # independently learn category codes and LightGBM's init_model schema could
    # differ between source and target.
    source_pipeline = factory(numeric, categorical, **dict(params))
    target_pipeline = factory(numeric, categorical, **dict(params))
    if "categories" not in source_pipeline.named_steps or "model" not in source_pipeline.named_steps:
        raise TypeError("init_model transfer requires a LightGBM categories/model pipeline")
    source_model = source_pipeline.named_steps["model"]
    target_model = target_pipeline.named_steps["model"]
    source_model.fit(source_X, source_fit[label])
    if task == "severity":
        source_classes = np.asarray(source_model.classes_)
        if not np.array_equal(source_classes, np.asarray(sorted(CLASSES))):
            raise ValueError("source LightGBM class order does not match the fixed class set")
    target_model.fit(target_X, target_fit[label], init_model=source_model.booster_)
    if task == "severity" and not np.array_equal(source_model.classes_, target_model.classes_):
        raise ValueError("source and target LightGBM class orders differ")
    # Keep the shared transformer attached for predictions, too.  Returning the
    # raw estimator would pass unencoded strings to LightGBM at validation time.
    target_pipeline.steps[0] = ("categories", codes)
    return target_pipeline, len(target_fit), len(source_fit)


def _source_prediction_columns(task: str) -> list[str]:
    return ([f"source_p_{label}" for label in CLASSES]
            if task == "severity" else ["source_predicted_delay"])


def _source_predictions_for_outer_train(target_training: pd.DataFrame,
                                        outer_validation: pd.DataFrame,
                                        source_rows: pd.DataFrame,
                                        prediction_time: str, task: str,
                                        numeric: list[str], categorical: list[str],
                                        params: Mapping, factory: Callable,
                                        inner_splitter: Callable) -> tuple[pd.DataFrame, int]:
    """Make source-model predictions for target training rows by inner weeks."""
    label = TARGETS[task]
    pred_columns = _source_prediction_columns(task)
    output = pd.DataFrame(index=target_training.index, columns=pred_columns, dtype=float)
    source_rows_used = 0
    for _, inner_train_mask, inner_valid_mask in inner_splitter(target_training):
        inner_training = target_training.loc[inner_train_mask]
        inner_validation = target_training.loc[inner_valid_mask]
        unavailable = pd.concat([inner_validation, outer_validation], ignore_index=True)
        source_fit = _source_training_rows(
            source_rows, inner_training, unavailable, prediction_time, task)
        if source_fit.empty:
            raise ValueError("an inner week split has no eligible source-corridor training rows")
        source_model, count = _fit(factory(numeric, categorical, **dict(params)),
                                   source_fit, numeric + categorical, task)
        source_rows_used += count
        eligible_valid = inner_validation.loc[inner_validation[label].notna()]
        if eligible_valid.empty:
            continue
        prediction = _predict(source_model, eligible_valid, numeric + categorical, task)
        output.loc[eligible_valid.index, pred_columns] = np.asarray(prediction).reshape(
            len(eligible_valid), len(pred_columns))
    eligible_training = target_training.loc[target_training[label].notna()]
    if output.loc[eligible_training.index].isna().any().any():
        raise ValueError("inner source folds did not predict every labeled target training row")
    return output, source_rows_used


def evaluate_corridor_transfer(
    target_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    *,
    prediction_time: str,
    task: str,
    method: str,
    numeric: list[str],
    categorical: list[str],
    params: Mapping,
    splits: Callable = cv_folds,
    inner_splitter: Callable = inner_folds,
    factory: Callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one transfer method on target CV folds.

    Parameters
    ----------
    target_rows, source_rows:
        Caller-supplied CV prediction rows.  Both include journey ID, prediction
        time, planned origin departure, corridor, requested base features and
        the two outcome columns.  Target rows also include the committed CV
        ``fold`` assignment.  Source rows are never loaded by this module.
    method:
        ``pooled`` combines eligible source/target training rows and adds the
        corridor category; ``init_model`` pretrains a source LightGBM booster
        then continues it on target rows; ``stacked`` uses inner-cross-fitted
        source predictions as target-model features.

    Returns ``(fold_metrics, validation_predictions)`` in the shared ``cv.py``
    output schema, with a ``method`` column on each table.  ``factory`` exists
    for small tests; production defaults are the project's LightGBM factories.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    target, source = _check_rows(target_rows, source_rows, prediction_time, task,
                                 numeric, categorical)
    base_numeric, base_categorical = list(numeric), list(categorical)
    model_factory = factory or (make_lightgbm_classifier if task == "severity" else make_lightgbm)
    metrics, predictions = [], []
    label = TARGETS[task]

    if method == "pooled":
        if "corridor" in base_numeric:
            raise ValueError("corridor must be categorical for pooled transfer")
        pooled_categorical = base_categorical + ([] if "corridor" in base_categorical else ["corridor"])
        pooled_features = [*base_numeric, *pooled_categorical]
        for fold, training_mask, validation_mask in splits(target):
            training = target.loc[training_mask]
            validation = target.loc[validation_mask]
            source_fit = _source_training_rows(source, training, validation,
                                               prediction_time, task)
            target_fit = training.loc[training[label].notna()]
            combined = pd.concat([target_fit, source_fit], ignore_index=True, sort=False)
            if combined.empty:
                raise ValueError("pooled transfer has no labeled training rows")
            if "corridor" in combined:
                combined["corridor"] = combined["corridor"].astype("string").fillna("unknown")
            if task == "severity":
                _check_severity_classes(combined, name="pooled")
            model, _ = _fit(model_factory(base_numeric, pooled_categorical, **dict(params)),
                            combined, pooled_features, task)
            valid = validation.loc[validation[label].notna()]
            if "corridor" in valid:
                valid = valid.copy()
                valid["corridor"] = valid["corridor"].astype("string").fillna("unknown")
            pred = _predict(model, valid, pooled_features, task)
            scores, out = score_fold(valid, prediction_time, fold, task,
                                     len(target_fit) + len(source_fit), pred)
            scores.update(method=method, target_train_rows=len(target_fit),
                          source_train_rows=len(source_fit))
            metrics.append(scores)
            predictions.append(out.assign(method=method))

    elif method == "init_model":
        features = [*base_numeric, *base_categorical]
        for fold, training_mask, validation_mask in splits(target):
            training = target.loc[training_mask]
            validation = target.loc[validation_mask]
            source_fit = _source_training_rows(source, training, validation,
                                               prediction_time, task)
            model, target_count, source_count = _fit_with_init_model(
                source_fit, training, base_numeric, base_categorical, params,
                task, model_factory)
            valid = validation.loc[validation[label].notna()]
            pred = _predict(model, valid, features, task)
            scores, out = score_fold(valid, prediction_time, fold, task,
                                     target_count + source_count, pred)
            scores.update(method=method, target_train_rows=target_count,
                          source_train_rows=source_count)
            metrics.append(scores)
            predictions.append(out.assign(method=method))

    else:  # stacked
        features = [*base_numeric, *base_categorical]
        source_pred_columns = _source_prediction_columns(task)
        target_numeric = [*base_numeric, *source_pred_columns]
        for fold, training_mask, validation_mask in splits(target):
            training = target.loc[training_mask]
            validation = target.loc[validation_mask]
            source_oof, inner_source_rows = _source_predictions_for_outer_train(
                training, validation, source, prediction_time, task, base_numeric,
                base_categorical, params, model_factory, inner_splitter)
            target_training = training.loc[training[label].notna()].copy()
            target_training[source_pred_columns] = source_oof.loc[target_training.index]

            source_fit = _source_training_rows(source, training, validation,
                                               prediction_time, task)
            if source_fit.empty:
                raise ValueError("an outer week split has no eligible source-corridor training rows")
            source_model, source_count = _fit(
                model_factory(base_numeric, base_categorical, **dict(params)),
                source_fit, features, task)
            valid = validation.loc[validation[label].notna()].copy()
            valid[source_pred_columns] = _predict(source_model, valid, features, task).reshape(
                len(valid), len(source_pred_columns))
            target_model, _ = _fit(
                model_factory(target_numeric, base_categorical, **dict(params)),
                target_training, [*target_numeric, *base_categorical], task)
            pred = _predict(target_model, valid, [*target_numeric, *base_categorical], task)
            scores, out = score_fold(valid, prediction_time, fold, task,
                                     len(target_training) + source_count, pred)
            scores.update(method=method, target_train_rows=len(target_training),
                          source_train_rows=source_count,
                          source_inner_fit_rows=inner_source_rows)
            metrics.append(scores)
            predictions.append(out.assign(method=method))

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)
