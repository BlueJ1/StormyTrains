"""Analysis helpers for paired final-model predictions.

This module is deliberately data-free at import time. Functions consume
prediction and metadata frames supplied by the runner; none open project data
or decide which hold-out rows to load.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _paths  # noqa: F401  shared modelling code and dataset builders
from cv import PROBA_COLUMNS, TARGETS
from data import PREDICTION_TIMES, TIME_COLUMN
from folds import SEED, period_of, week_of
from metrics import CLASSES, EPS, class_log_loss, macro_f1, rmse
from representations import EXTREMES

KEYS = ["journey_id", "prediction_time", TIME_COLUMN]
WEATHER_BRANCH = "weather_branch"
WEATHER_EXTREMES = tuple(EXTREMES)
HORIZONS = tuple(PREDICTION_TIMES)


def _task(task: str) -> str:
    if task not in TARGETS:
        raise ValueError("task must be 'severity' or 'delay'")
    return TARGETS[task]


def _validate_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    target = _task(task)
    predictions = PROBA_COLUMNS if task == "severity" else ["predicted_delay"]
    required = [*KEYS, target, *predictions]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    rows = frame[required].copy()
    if rows[KEYS].isna().any().any():
        raise ValueError("prediction keys cannot be missing")
    if pd.to_datetime(rows[KEYS[2]], errors="coerce").isna().any():
        raise ValueError("planned departure times must be valid timestamps")
    if rows.duplicated(KEYS).any():
        raise ValueError("prediction keys must be unique")
    if pd.isna(rows[target]).any():
        raise ValueError("prediction targets cannot be missing")

    if task == "severity":
        labels = set(rows[target].astype(str))
        if not labels <= set(CLASSES):
            raise ValueError(f"severity target contains unknown classes: {sorted(labels - set(CLASSES))}")
        probabilities = rows[PROBA_COLUMNS].to_numpy(dtype=float)
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0)):
            raise ValueError("severity probabilities must be finite, nonnegative and sum to one")
    else:
        target_values = rows[target].to_numpy(dtype=float)
        predictions_values = rows["predicted_delay"].to_numpy(dtype=float)
        if not np.isfinite(target_values).all() or not np.isfinite(predictions_values).all():
            raise ValueError("delay targets and predictions must be finite")
    return rows


def _align(left: pd.DataFrame, right: pd.DataFrame, task: str) -> pd.DataFrame:
    """Align one-to-one on the journey/time keys and check paired targets."""
    target = _task(task)
    prediction_columns = PROBA_COLUMNS if task == "severity" else ["predicted_delay"]
    left_rows = _validate_frame(left, task).rename(
        columns={target: "_target_left", **{c: f"_left_{c}" for c in prediction_columns}})
    right_rows = _validate_frame(right, task).rename(
        columns={target: "_target_right", **{c: f"_right_{c}" for c in prediction_columns}})
    paired = left_rows.merge(right_rows, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("paired prediction frames must contain exactly the same keys")
    target_equal = paired["_target_left"].eq(paired["_target_right"]).fillna(False)
    if not target_equal.all():
        raise ValueError("paired prediction frames must have identical targets")
    paired = paired.rename(columns={"_target_left": target}).drop(columns=["_target_right", "_merge"])
    return paired


def _week_labels(rows: pd.DataFrame) -> pd.Series:
    departures = pd.to_datetime(rows[KEYS[2]])
    return week_of(departures)


def _score(rows: pd.DataFrame, task: str, side: str) -> tuple[float, float | None]:
    target = _task(task)
    y = rows[target]
    if task == "severity":
        probabilities = rows[[f"{side}_{c}" for c in PROBA_COLUMNS]].to_numpy(dtype=float)
        return class_log_loss(y, probabilities), macro_f1(y, probabilities)
    predictions = rows[f"{side}_predicted_delay"].to_numpy(dtype=float)
    return rmse(y, predictions), None


def week_loss_sums(rows, task):
    """Count rows and sum row losses separately for each arm and ISO week."""
    target = rows[TARGETS[task]]
    data = {"week": rows["_week"], "count": 1}
    for arm in ("_left", "_right"):
        if task == "severity":
            index = target.map(dict(zip(CLASSES, range(len(CLASSES))))).to_numpy(dtype=int)
            proba = rows[[f"{arm}_{column}" for column in PROBA_COLUMNS]].to_numpy()
            loss = -np.log(np.clip(proba[np.arange(len(rows)), index], EPS, 1))
        else:
            loss = (rows[f"{arm}_predicted_delay"] - target) ** 2
        data[arm] = loss
    return pd.DataFrame(data).groupby("week", sort=True).sum()


def paired_bootstrap_ci(left: pd.DataFrame, right: pd.DataFrame, task: str, *,
                        n_resamples: int = 2000, confidence: float = 0.95,
                        seed: int = SEED) -> pd.DataFrame:
    """Return paired whole-ISO-week bootstrap intervals for log loss or RMSE.

    ``delta`` is right minus left, so negative values favor the right-hand
    model for log loss and RMSE. Severity macro-F1 deltas are included as
    descriptive point estimates and are not bootstrapped.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    paired = _align(left, right, task)
    paired["_week"] = _week_labels(paired)
    if paired["_week"].isna().any():
        raise ValueError("planned departure times must be valid timestamps")

    rng = np.random.default_rng(seed)
    output = []
    metric = "log_loss" if task == "severity" else "rmse"
    for prediction_time, rows in paired.groupby("prediction_time", sort=True):
        sums = week_loss_sums(rows, task)
        n_weeks = len(sums)
        if n_weeks < 2:
            raise ValueError("paired week bootstrap needs at least two ISO weeks")
        left_score, left_f1 = _score(rows, task, "_left")
        right_score, right_f1 = _score(rows, task, "_right")
        sampled = sums[["count", "_left", "_right"]].to_numpy()[
            rng.integers(n_weeks, size=(n_resamples, n_weeks))].sum(axis=1)
        scores = sampled[:, 1:] / sampled[:, :1]
        if task == "delay":
            scores = np.sqrt(scores)
        deltas = scores[:, 1] - scores[:, 0]
        alpha = (1 - confidence) / 2
        lower, upper = np.quantile(deltas, [alpha, 1 - alpha])
        result = {
            "prediction_time": prediction_time,
            "metric": metric,
            "left_score": left_score,
            "right_score": right_score,
            "delta": right_score - left_score,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence": confidence,
            "n_rows": len(rows),
            "n_weeks": n_weeks,
            "n_resamples": n_resamples,
        }
        if task == "severity":
            result.update({
                "left_macro_f1": left_f1,
                "right_macro_f1": right_f1,
                "delta_macro_f1": right_f1 - left_f1,
            })
        output.append(result)
    return pd.DataFrame(output)


def _metadata_keys(metadata: pd.DataFrame) -> list[str]:
    keys = [key for key in KEYS if key in metadata.columns]
    if "journey_id" not in keys:
        raise ValueError("metadata must include journey_id")
    if metadata.duplicated(keys).any():
        raise ValueError(f"metadata must be unique by its available keys: {keys}")
    return keys


def _attach_metadata(rows: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None:
        return rows.copy()
    keys = _metadata_keys(metadata)
    # Keep any target or key already present in the predictions from being
    # shadowed by redundant metadata columns.
    values = metadata.drop(columns=[c for c in metadata if c in rows.columns and c not in keys],
                           errors="ignore")
    merged = rows.merge(values, on=keys, how="left", validate="many_to_one", indicator="_metadata_match")
    if not merged["_metadata_match"].eq("both").all():
        raise ValueError("metadata must cover every prediction row")
    return merged.drop(columns="_metadata_match")


def _classes_for_errors(rows: pd.DataFrame, task: str, metadata: pd.DataFrame | None) -> pd.Series:
    if task == "severity":
        return rows["severity_class"].astype(str)
    if metadata is not None and "severity_class" in metadata.columns:
        joined = _attach_metadata(rows[[*KEYS]], metadata)
        labels = joined["severity_class"].astype(str)
        if not set(labels) <= set(CLASSES[:3]):
            raise ValueError("delay metadata severity_class must label completed journeys")
        return labels
    y = rows["arrival_delay_min"].to_numpy(dtype=float)
    return pd.Series(np.select([y < 20, y < 60], CLASSES[:2], default=CLASSES[2]), index=rows.index)


def _extreme_group(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any() or not set(numeric.dropna().unique()) <= {0, 1, 0.0, 1.0}:
        raise ValueError("weather extreme flags must be 0, 1, or missing")
    levels = pd.Series("unknown", index=values.index, dtype="string")
    known = numeric.notna()
    levels.loc[known] = np.where(numeric.loc[known].astype(bool), "extreme", "not_extreme")
    return levels


def _group_values(rows: pd.DataFrame, group: str, task: str,
                  metadata: pd.DataFrame | None) -> pd.Series:
    if group == "class":
        return _classes_for_errors(rows, task, metadata)
    if group == "period":
        return period_of(pd.to_datetime(rows[KEYS[2]])).astype("string")
    if group == WEATHER_BRANCH:
        if group not in rows:
            raise ValueError(f"metadata is missing {group!r}")
        return rows[group].astype("string").fillna("unknown")
    if group in WEATHER_EXTREMES:
        if group not in rows:
            raise ValueError(f"metadata is missing {group!r}")
        return _extreme_group(rows[group])
    raise ValueError(f"unknown analysis group {group!r}")


def paired_error_table(left: pd.DataFrame, right: pd.DataFrame, task: str, *,
                       metadata: pd.DataFrame | None = None,
                       groups: tuple[str, ...] = (
                           "class", "period", WEATHER_BRANCH, *WEATHER_EXTREMES)) -> pd.DataFrame:
    """Compare existing task metrics by class, period, branch, and weather flag.

    The class rows use conditional log loss for severity or RMSE for delay.
    Macro-F1 is reported only for broader groups where it summarizes multiple
    realized classes. Missing branch/extreme metadata is shown as ``unknown``.
    """
    if not groups:
        raise ValueError("at least one analysis group is required")
    paired = _align(left, right, task)
    paired = _attach_metadata(paired, metadata)
    output = []
    for group in groups:
        paired["_group"] = _group_values(paired, group, task, metadata)
        for value, rows in paired.groupby("_group", sort=False, dropna=False):
            left_score, left_f1 = _score(rows, task, "_left")
            right_score, right_f1 = _score(rows, task, "_right")
            metric = "log_loss" if task == "severity" else "rmse"
            result = {
                "task": task,
                "grouping": group,
                "group": value,
                "n_rows": len(rows),
                "metric": metric,
                "left_score": left_score,
                "right_score": right_score,
                "delta": right_score - left_score,
            }
            if task == "severity" and group != "class":
                result.update({
                    "left_macro_f1": left_f1,
                    "right_macro_f1": right_f1,
                    "delta_macro_f1": right_f1 - left_f1,
                })
            output.append(result)
    return pd.DataFrame(output)


def calibration_reliability(predictions: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Return one-vs-rest reliability bin counts for severity probabilities.

    Output contains only the ingredients of a reliability plot (bin count,
    mean predicted probability, and observed frequency), not a calibration
    score. Equal-width bins cover [0, 1], with probability 1 in the final bin.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    rows = _validate_frame(predictions, "severity")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    output = []
    for prediction_time, time_rows in rows.groupby("prediction_time", sort=False, dropna=False):
        y = time_rows["severity_class"].astype(str).to_numpy()
        for label in CLASSES:
            probability = time_rows[f"p_{label}"].to_numpy(dtype=float)
            bins = np.searchsorted(edges, probability, side="right") - 1
            bins = np.clip(bins, 0, n_bins - 1)
            observed = (y == label).astype(float)
            for bin_number in range(n_bins):
                selected = bins == bin_number
                count = int(selected.sum())
                output.append({
                    "prediction_time": prediction_time,
                    "class": label,
                    "bin": bin_number,
                    "bin_lower": float(edges[bin_number]),
                    "bin_upper": float(edges[bin_number + 1]),
                    "n_rows": count,
                    "mean_predicted_probability": float(probability[selected].mean()) if count else np.nan,
                    "observed_frequency": float(observed[selected].mean()) if count else np.nan,
                })
    return pd.DataFrame(output)


def example_risk_trajectories(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select the earliest complete five-horizon journey in each realized class.

    Selection is fixed by realized severity, planned local departure, then
    journey ID; predicted success or risk never affects selection. The output
    includes ``p_severe_or_canceled`` at each horizon.
    """
    rows = _validate_frame(predictions, "severity")
    unknown_times = set(rows["prediction_time"].astype(str)) - set(HORIZONS)
    if unknown_times:
        raise ValueError(f"unknown prediction horizons: {sorted(unknown_times)}")
    complete = []
    for (journey_id, departure), group in rows.groupby(["journey_id", KEYS[2]], sort=False):
        times = group["prediction_time"].astype(str).tolist()
        if len(times) != len(HORIZONS) or set(times) != set(HORIZONS):
            continue
        labels = group["severity_class"].astype(str).unique()
        if len(labels) != 1:
            raise ValueError(f"journey {journey_id!r} has inconsistent severity targets across horizons")
        complete.append({"journey_id": journey_id, KEYS[2]: departure,
                         "severity_class": labels[0]})
    candidates = pd.DataFrame(complete)
    if candidates.empty:
        return pd.DataFrame(columns=[*KEYS[:1], KEYS[2], "severity_class", "prediction_time",
                                     *PROBA_COLUMNS, "p_severe_or_canceled"])
    candidates = candidates.sort_values([KEYS[2], "journey_id"], kind="stable")
    selected = candidates.drop_duplicates("severity_class", keep="first")
    chosen = pd.MultiIndex.from_frame(selected[["journey_id", KEYS[2]]])
    row_ids = pd.MultiIndex.from_frame(rows[["journey_id", KEYS[2]]])
    trajectories = rows.loc[row_ids.isin(chosen)].copy()
    trajectories["_horizon_order"] = trajectories["prediction_time"].astype(str).map(
        {time: i for i, time in enumerate(HORIZONS)})
    trajectories["_class_order"] = trajectories["severity_class"].astype(str).map(
        {label: index for index, label in enumerate(CLASSES)})
    trajectories = trajectories.sort_values(["_class_order", KEYS[2], "journey_id", "_horizon_order"],
                                            kind="stable")
    trajectories["p_severe_or_canceled"] = trajectories["p_severe"] + trajectories["p_canceled"]
    return trajectories.drop(columns=["_horizon_order", "_class_order"]).reset_index(drop=True)
