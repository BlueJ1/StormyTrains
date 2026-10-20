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
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data import HERE, PREDICTION_TIMES, TIME_COLUMN, load_rows
from features import feature_columns
from folds import cv_folds
from metrics import CLASSES, class_log_loss, macro_f1, rmse

RESULTS = HERE / "results"
TARGETS = {"severity": "severity_class", "delay": "arrival_delay_min"}
PROBA_COLUMNS = [f"p_{c}" for c in CLASSES]


def class_probabilities(model, X) -> np.ndarray:
    """predict_proba with its columns put in CLASSES order."""
    classes = list(model.classes_)
    missing = set(CLASSES) - set(classes)
    if missing:
        raise ValueError(f"classes missing from the training rows: {sorted(missing)}")
    return model.predict_proba(X)[:, [classes.index(c) for c in CLASSES]]


def cross_validate(rows: pd.DataFrame, prediction_time: str, task: str,
                   make_model, numeric: list[str], categorical: list[str]):
    """Return (fold metrics, validation predictions) for one prediction time."""
    if set(rows["prediction_time"].astype(str)) != {prediction_time}:
        raise ValueError(f"rows must all be for prediction time {prediction_time}")
    target = TARGETS[task]
    if task == "delay":
        rows = rows[rows[target].notna()]
    features = numeric + categorical

    metrics, predictions = [], []
    for fold, train, validation in cv_folds(rows):
        model = make_model(numeric, categorical)
        model.fit(rows.loc[train, features], rows.loc[train, target])
        X, y = rows.loc[validation, features], rows.loc[validation, target]
        out = rows.loc[validation, ["journey_id", TIME_COLUMN, target]].assign(
            prediction_time=prediction_time, fold=fold)
        if task == "severity":
            proba = class_probabilities(model, X)
            out[PROBA_COLUMNS] = proba
            scores = {"log_loss": class_log_loss(y, proba), "macro_f1": macro_f1(y, proba)}
        else:
            out["predicted_delay"] = model.predict(X)
            scores = {"rmse": rmse(y, out["predicted_delay"])}
        metrics.append({"prediction_time": prediction_time, "fold": fold,
                        "train_rows": int(train.sum()),
                        "validation_rows": int(validation.sum()), **scores})
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
    changed = git("status", "--porcelain", "--", ":/", ":(exclude)results")
    return commit + (" (uncommitted changes)" if changed else "")


def run_cv(name: str, task: str, make_model, *, weather: bool = False,
           prediction_times=PREDICTION_TIMES, results_dir: Path = RESULTS) -> pd.DataFrame:
    """Cross-validate one model at every prediction time and save the results.

    weather=True adds the weather features (features.WEATHER); it is keyword-only.
    """
    all_metrics, all_predictions, config_features = [], [], {}
    for prediction_time in prediction_times:
        numeric, categorical = feature_columns(prediction_time, weather)
        config_features[prediction_time] = {"numeric": numeric, "categorical": categorical}
        rows = load_rows(prediction_time, weather=weather)
        metrics, predictions = cross_validate(rows, prediction_time, task, make_model,
                                              numeric, categorical)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
    fold_metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize(fold_metrics)

    out = results_dir / name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    config = {"name": name, "task": task, "target": TARGETS[task],
              "model": str(make_model(*feature_columns(prediction_times[0], weather))),
              "weather": weather, "features": config_features, "git_commit": git_commit(),
              "created": datetime.now().isoformat(timespec="seconds")}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(
        out / "oof_predictions.parquet", index=False)
    return summary
