"""Cross-validation shared by every model.

For each prediction time and each fold of frankfurt_munich/folds.py:

1. the validation rows are the CV rows with that `fold`, the training rows
   all other CV rows (folds are groups of whole weeks, see folds.py);
2. a fresh model is fitted on the training rows and predicts the validation
   rows;
3. the fold is scored: log loss and macro-F1 for `severity`, RMSE for
   `delay`. The `delay` task uses completed journeys only.

Only the mean and standard deviation of the fold scores are reported; the
validation predictions are saved but not pooled into one score.

run_cv writes results/<name>/<timestamp>/:
- config.json: task, model, whether weather is used, features, git commit;
- fold_metrics.csv: one row per prediction time and fold;
- summary.csv: mean and standard deviation over the folds;
- oof_predictions.parquet: every validation prediction.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import process_time

import numpy as np
import pandas as pd

from data import HERE, PREDICTION_TIMES, TIME_COLUMN, load_rows
from features import feature_columns
from folds import cv_folds
from metrics import CLASSES, class_log_loss, macro_f1, rmse

RESULTS = HERE / "results"
TARGETS = {"severity": "severity_class", "delay": "arrival_delay_min"}
PROBA_COLUMNS = [f"p_{c}" for c in CLASSES]
MODEL_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm",
                  "xgboost", "catboost", "interpret-core", "torch", "pytabkit",
                  "ngboost", "quantile-forest", "lightgbmlss")


def class_probabilities(model, X) -> np.ndarray:
    """predict_proba with its columns put in CLASSES order."""
    classes = list(model.classes_)
    missing = set(CLASSES) - set(classes)
    if missing:
        raise ValueError(f"classes missing from the training rows: {sorted(missing)}")
    return model.predict_proba(X)[:, [classes.index(c) for c in CLASSES]]


def cross_validate(rows: pd.DataFrame, prediction_time: str, task: str,
                   make_model, numeric: list[str], categorical: list[str], *,
                   splits=cv_folds):
    """Return (fold metrics, validation predictions) for one prediction time."""
    if set(rows["prediction_time"].astype(str)) != {prediction_time}:
        raise ValueError(f"rows must all be for prediction time {prediction_time}")
    metrics, predictions = [], []
    for fold, train, validation in splits(rows):
        scores, out = fit_fold(rows.loc[train], rows.loc[validation], prediction_time, fold,
                               task, make_model, numeric, categorical)
        metrics.append(scores)
        predictions.append(out)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def fit_fold(train, validation, prediction_time, fold, task, make_model, numeric, categorical):
    """Fit once on the training rows and score on the supplied validation rows."""
    target = TARGETS[task]
    features = numeric + categorical
    model = make_model(numeric, categorical)
    if task == "delay":
        validation = validation.loc[validation[target].notna()]
    start = process_time()
    if hasattr(model, "fit_joint"):
        model.fit_joint(train[features], train[TARGETS["severity"]], train[TARGETS["delay"]])
    else:
        train = train.loc[train[target].notna()]
        model.fit(train[features], train[target])
    fit_cpu_seconds = process_time() - start
    X = validation[features]
    prediction = class_probabilities(model, X) if task == "severity" else model.predict(X)
    metrics, out = score_fold(validation, prediction_time, fold, task, len(train), prediction)
    if hasattr(model, "n_parameters_"):
        metrics.update(parameters=model.n_parameters_, widths=json.dumps(model.widths),
                       fit_cpu_seconds=fit_cpu_seconds)
    estimator = getattr(model, "named_steps", {}).get("model")
    if getattr(estimator, "solver", None) == "saga":
        iterations = int(np.max(estimator.n_iter_))
        metrics.update(fit_iterations=iterations, converged=iterations < estimator.max_iter)
    return metrics, out


def score_fold(validation, prediction_time, fold, task, train_rows, prediction):
    """Write fold metrics and aligned predictions for any model."""
    target = TARGETS[task]
    y = validation[target]
    out = validation[["journey_id", TIME_COLUMN, target]].assign(
        prediction_time=prediction_time, fold=fold)
    prediction = np.asarray(prediction)
    shape = (len(validation), len(CLASSES)) if task == "severity" else (len(validation),)
    if prediction.shape != shape or not np.isfinite(prediction).all():
        raise ValueError("predictions must align with validation rows and be finite")
    if task == "severity":
        if (prediction < 0).any() or not np.allclose(prediction.sum(axis=1), 1):
            raise ValueError("class probabilities must be nonnegative and sum to one")
        out[PROBA_COLUMNS] = prediction
        scores = {"log_loss": class_log_loss(y, prediction), "macro_f1": macro_f1(y, prediction)}
    else:
        out["predicted_delay"] = prediction
        scores = {"rmse": rmse(y, prediction)}
    return {"prediction_time": prediction_time, "fold": fold, "train_rows": train_rows,
            "validation_rows": len(validation), **scores}, out


def pooled_cv(rows, task, factory, numeric, categorical):
    """Fit one model per fold using all horizons, then score each horizon separately."""
    target = TARGETS[task]
    rows = rows[rows[target].notna()]
    features = numeric + categorical
    metrics, predictions = [], []
    for fold, training, validation in cv_folds(rows):
        train = rows.loc[training]
        model = factory(numeric, categorical).fit(train[features], train[target])
        for prediction_time, valid in rows.loc[validation].groupby("prediction_time", observed=True):
            X = valid[features]
            prediction = class_probabilities(model, X) if task == "severity" else model.predict(X)
            row, out = score_fold(valid, prediction_time, fold, task, len(train), prediction)
            metrics.append(row)
            predictions.append(out)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    scores = [c for c in fold_metrics if c in ("log_loss", "macro_f1", "rmse")]
    summary = fold_metrics.groupby("prediction_time", sort=False)[scores].agg(["mean", "std"])
    summary.columns = [f"{score}_{stat}" for score, stat in summary.columns]
    return summary.reset_index()


def git_commit() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    # Results written earlier in the same run don't count as changes.
    changed = git("status", "--porcelain", "--", ":/", ":(exclude,glob)**/results/**")
    return commit + (" (uncommitted changes)" if changed else "")


def runtime_versions():
    versions = {"python": platform.python_version(), "architecture": platform.machine()}
    for package in MODEL_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def run_cv(name: str, task: str, make_model, *, weather: bool = False,
           feature_set: str = "base", prediction_times=PREDICTION_TIMES,
           results_dir: Path = RESULTS) -> pd.DataFrame:
    """Cross-validate one model at every prediction time and save the results.

    weather=True adds the weather features (features.WEATHER); it is keyword-only.
    """
    all_metrics, all_predictions, config_features = [], [], {}
    for prediction_time in prediction_times:
        numeric, categorical = feature_columns(prediction_time, weather, feature_set=feature_set)
        config_features[prediction_time] = {"numeric": numeric, "categorical": categorical}
        rows = load_rows(prediction_time, weather=weather)
        metrics, predictions = cross_validate(rows, prediction_time, task, make_model,
                                              numeric, categorical)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
    first = config_features[prediction_times[0]]
    config = {"name": name, "task": task, "target": TARGETS[task],
              "model": str(make_model(first["numeric"], first["categorical"])),
              "weather": weather, "feature_set": feature_set, "features": config_features}
    _, summary = save_run(results_dir, name, config, pd.concat(all_metrics, ignore_index=True),
                          pd.concat(all_predictions, ignore_index=True))
    return summary


def save_run(results_dir: Path, name: str, config: dict, fold_metrics: pd.DataFrame,
             predictions: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    """Write config.json (plus the git commit and time), fold_metrics.csv,
    summary.csv and oof_predictions.parquet to results/<name>/<timestamp>/.

    Return that directory and the summary.
    """
    out = results_dir / name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    config = {**config, "git_commit": config.get("git_commit") or git_commit(),
              "runtime": runtime_versions(),
              "created": datetime.now().isoformat(timespec="seconds")}
    summary = summarize(fold_metrics)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    predictions.to_parquet(out / "oof_predictions.parquet", index=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out, summary
