"""Descriptive differences between runs on identical validation rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import PREDICTION_TIMES, TIME_COLUMN
from folds import N_FOLDS
from tuning import SCORE

MINIMUM_GAIN = {"log_loss": 0.003, "rmse": 0.1}
FOLD_KEY = ["prediction_time", "fold"]
ROW_KEY = [*FOLD_KEY, "journey_id"]


def paired_differences(reference, candidate, score):
    """Negative differences favor the candidate. The mean gives each fold equal weight."""
    left, right = (frame.set_index(FOLD_KEY).sort_index() for frame in (reference, candidate))
    if not left.index.is_unique or not left.index.equals(right.index):
        raise ValueError("runs must contain the same unique prediction times and folds")
    if not left["validation_rows"].equals(right["validation_rows"]):
        raise ValueError("validation row counts differ")
    output = []
    for prediction_time in PREDICTION_TIMES:
        if prediction_time not in left.index.get_level_values("prediction_time"):
            continue
        delta = right.loc[prediction_time, score] - left.loc[prediction_time, score]
        if delta.index.tolist() != list(range(1, N_FOLDS + 1)):
            raise ValueError(f"{prediction_time} must contain all {N_FOLDS} folds")
        if not np.isfinite(delta).all():
            raise ValueError(f"{prediction_time} has non-finite score differences")
        mean = delta.mean()
        output.append({"prediction_time": prediction_time,
                       **{f"fold_{fold}": value for fold, value in delta.items()},
                       "mean": mean, "folds_improved": int(delta.lt(0).sum()),
                       "reaches_minimum_gain": bool(-mean >= MINIMUM_GAIN[score] - 1e-12)})
    if len(output) != left.index.get_level_values("prediction_time").nunique() or not output:
        raise ValueError("comparison has missing or unknown prediction times")
    return pd.DataFrame(output)


def compare_runs(reference: Path, candidate: Path, task: str, *, subset_times=False):
    target = {"severity": "severity_class", "delay": "arrival_delay_min"}[task]
    columns = [*ROW_KEY, TIME_COLUMN, target]
    left, right = (
        pd.read_parquet(path / "oof_predictions.parquet", columns=columns)
        for path in (reference, candidate)
    )
    times = right["prediction_time"].unique()
    if subset_times:
        left = left[left["prediction_time"].isin(times)]
    left, right = (frame.sort_values(ROW_KEY).reset_index(drop=True) for frame in (left, right))
    if left.duplicated(ROW_KEY).any() or not left.equals(right):
        raise ValueError("runs must have the same validation rows and targets")
    reference_metrics = pd.read_csv(reference / "fold_metrics.csv")
    if subset_times:
        reference_metrics = reference_metrics[reference_metrics["prediction_time"].isin(times)]
    return paired_differences(reference_metrics, pd.read_csv(candidate / "fold_metrics.csv"),
                              SCORE[task])


def markdown(comparison, *, signed=True):
    columns = list(comparison)
    style = "+.6f" if signed else ".6f"
    rows = [[format(value, style) if isinstance(value, float) else str(value)
             for value in row] for row in comparison.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(row) + " |"
                     for row in [columns, ["---"] * len(columns), *rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("task", choices=SCORE)
    args = parser.parse_args()
    print(markdown(compare_runs(args.reference, args.candidate, args.task)))


if __name__ == "__main__":
    main()
