"""Summarize completed advanced CV runs in the modelling README."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from compare import markdown
from cv import RESULTS
from data import HERE, PREDICTION_TIMES
from tuning import SCORE

BEGIN = "<!-- BEGIN ADVANCED CV -->"
END = "<!-- END ADVANCED CV -->"


def run_index(results_dir: Path):
    """Read each completed run with its saved configuration and summary."""
    for root in sorted(results_dir.glob("advanced_*")):
        ready = [path for path in root.iterdir()
                 if (path / "config.json").is_file() and (path / "summary.csv").is_file()]
        if not ready:
            continue
        path = max(ready)
        config = json.loads((path / "config.json").read_text())
        yield path, config, pd.read_csv(path / "summary.csv")


def reference_table(runs, step):
    records = []
    for path, config, summary in runs:
        if config["step"] != step or config.get("variant"):
            continue
        score = SCORE[config["task"]]
        scores = summary.set_index("prediction_time")[f"{score}_mean"]
        model = config["model"] + (" + weather" if config["weather"] else "")
        records.append({"Model": f"[{model}](results/{path.parent.name}/{path.name})",
                        "Search configs": config.get("search", {}).get("n_configs", "fixed"),
                        "Score": score, **{time: scores.get(time, float("nan"))
                                           for time in PREDICTION_TIMES}})
    return pd.DataFrame(records)


def comparison_table(runs):
    comparisons = []
    for path, config, _ in runs:
        file = path / "paired_vs_base.csv"
        if not file.is_file():
            continue
        table = pd.read_csv(file)
        evaluation = table["evaluation"] if "evaluation" in table else config["evaluation"]
        variant = config.get("variant", "")
        contrast = ""
        if config["step"] == "3.1":
            contrast = "timetable-only" if variant.startswith("add_") else "full"
        comparisons.append(table.assign(
            step=config["step"], task=config["task"], method=config["model"],
            variant=variant, evaluation=evaluation, weather=config["weather"], contrast=contrast,
            result=f"[folds](results/{path.parent.name}/{path.name}/paired_vs_base.csv)"))
    if not comparisons:
        return pd.DataFrame()
    table = pd.concat(comparisons, ignore_index=True)
    best = table.groupby(["step", "task", "evaluation", "prediction_time", "contrast"])["mean"].idxmin()
    return table.loc[best, ["step", "task", "evaluation", "prediction_time", "method", "variant", "weather", "contrast",
                            "mean", "folds_improved", "reaches_minimum_gain", "result"]]


def selection_table(runs):
    rows = []
    for _, config, _ in runs:
        for choice in config.get("group_choices", []):
            if choice["prediction_time"] not in ("24h", "0min"):
                continue
            for group in choice["groups"]:
                rows.append({"task": config["task"], "weather": config["weather"],
                             "direction": config["variant"], "prediction_time": choice["prediction_time"],
                             "group": group})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).value_counts(sort=False).rename("folds_selected").reset_index()


def importance_table(runs):
    tables = []
    for path, config, _ in runs:
        if config["step"] != "3.2" or not (path / "shap.csv").is_file():
            continue
        group = pd.read_csv(path / "permutation.csv").groupby(
            ["prediction_time", "group"], as_index=False)["score_increase"].mean()
        shap = pd.read_csv(path / "shap.csv").groupby(
            ["prediction_time", "feature"], as_index=False)["mean_abs_shap"].mean()
        top_group = group.loc[group.groupby("prediction_time")["score_increase"].idxmax()]
        top_shap = shap.loc[shap.groupby("prediction_time")["mean_abs_shap"].idxmax()]
        table = top_group.merge(top_shap, on="prediction_time").assign(task=config["task"])
        table["details"] = f"[folds](results/{path.parent.name}/{path.name})"
        tables.append(table[["task", "prediction_time", "group", "score_increase",
                             "feature", "mean_abs_shap", "details"]])
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def render(results_dir):
    runs = list(run_index(results_dir))
    output = ["### Advanced CV results", "",
              "Scores use the five committed CV week folds. Positive paired differences favor the reference."]
    tables = [("Step 0.7 reference scores", reference_table(runs, "0.7"), False),
              ("Step 0.8 nested reference scores", reference_table(runs, "0.8"), False),
              ("Smallest paired differences by step, task and horizon", comparison_table(runs), True),
              ("Step 3.2 leading permutation groups and TreeSHAP features", importance_table(runs), False),
              ("Step 3.5 group choices at 24 h and 0 min", selection_table(runs), False)]
    for title, table, signed in tables:
        if not table.empty:
            output.extend(["", f"#### {title}", "", markdown(table, signed=signed)])
    if not runs:
        output.extend(["", "Advanced CV results are pending."])
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    report = render(args.results_dir)
    if args.write_readme:
        readme = HERE / "README.md"
        before, sep, remainder = readme.read_text().partition(BEGIN)
        _, close, after = remainder.partition(END)
        if not sep or not close:
            raise ValueError("the README needs both advanced CV report markers")
        readme.write_text(before + BEGIN + "\n\n" + report + "\n\n" + END + after)
    else:
        print(report)


if __name__ == "__main__":
    main()
