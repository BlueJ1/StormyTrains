"""Compare the four weather arms of the screen (run_weather_fe_screen.py).

For each model and prediction time: the mean score of each arm over the 5 week
folds, and for each engineered arm its paired differences against the base
weather arm and against the railway arm: the mean difference and the number of
folds (of 5) in which the engineered arm is better. Lower is better for both
scores, so a negative difference favours the engineered arm. `d12` marks a mean
gain of at least 0.1 min RMSE or 0.003 log loss (the advanced plan's D12), as a
reference only.

Writes results_weather_fe/comparison.csv and prints Markdown tables.
Usage: compare_weather_fe.py
"""
from __future__ import annotations

import pandas as pd

from data import PREDICTION_TIMES
from run_weather_fe_screen import ARMS, LINEAR, SCREEN, TUNED, run_name
from tuning import SCORE

D12 = {"rmse": 0.1, "log_loss": 0.003}


def fold_scores(model: str, arm: str, metric: str) -> pd.DataFrame | None:
    runs = sorted((SCREEN / run_name(model, arm)).glob("*/fold_metrics.csv"))
    if not runs:
        return None
    return pd.read_csv(runs[-1]).pivot(index="prediction_time", columns="fold", values=metric)


def compare() -> pd.DataFrame:
    rows = []
    for model, (task, _) in {**LINEAR, **TUNED}.items():
        metric = SCORE[task]
        scores = {arm: fold_scores(model, arm, metric) for arm in ARMS}
        if any(s is None for s in scores.values()):
            continue
        for pt in PREDICTION_TIMES:
            row = {"task": task, "model": model, "prediction_time": pt, "metric": metric,
                   **{arm: scores[arm].loc[pt].mean() for arm in ARMS}}
            for arm in ("fe_replace", "fe_add"):
                for reference in ("base", "railway"):
                    diff = scores[arm].loc[pt] - scores[reference].loc[pt]
                    row[f"{arm}_vs_{reference}"] = diff.mean()
                    row[f"{arm}_vs_{reference}_folds"] = int((diff < 0).sum())
                    row[f"{arm}_vs_{reference}_d12"] = bool(diff.mean() <= -D12[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def markdown(table: pd.DataFrame, task: str) -> str:
    part = table[table.task == task]
    digits = 2 if task == "delay" else 4
    lines = ["| Model | Time | Railway | Base | FE replace | FE add | Replace − base (folds) | "
             "Add − base (folds) | Replace − railway (folds) | Add − railway (folds) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in part.iterrows():
        cells = [f"{r[a]:.{digits}f}" for a in ARMS]
        for arm, ref in (("fe_replace", "base"), ("fe_add", "base"),
                         ("fe_replace", "railway"), ("fe_add", "railway")):
            key = f"{arm}_vs_{ref}"
            mark = " **D12**" if r[f"{key}_d12"] else ""
            cells.append(f"{r[key]:+.{digits + 1}f} ({r[f'{key}_folds']}/5){mark}")
        lines.append(f"| {r.model} | {r.prediction_time} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    table = compare()
    table.to_csv(SCREEN / "comparison.csv", index=False)
    for task in ("delay", "severity"):
        metric = "RMSE (min)" if task == "delay" else "log loss"
        print(f"\n### {task}: {metric}, mean over 5 folds; differences negative = engineered better\n")
        print(markdown(table, task))
