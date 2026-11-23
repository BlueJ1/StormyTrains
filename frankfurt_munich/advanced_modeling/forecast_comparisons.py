"""Forecast arms against their matched arms (step 10.9, decision D17).

Each forecast run is paired with the run of the same step, model and variant in another arm,
on the same rows and folds, and compared with compare.py (descriptive, minimum gain D12):

    forecast_vs_railway            forecasts alone against railway only
    forecast_vs_weather            forecasts alone against observed weather
    weather_forecast_vs_weather    observations plus forecasts against observations
    weather_forecast_vs_forecast   observations plus forecasts against forecasts alone

A pair is skipped when its reference arm is not registered (railway only has no 1.5, 3.2 or
3.4 arm, and the 3.1 forecast ablations exist only in the forecast arms). Negative means favor
the forecast run (the candidate).

The CSV keeps every pair; the summary leaves out the rows marked in `excluded`: the
recorded-route leakage diagnostic of 0.7, and pairs whose two runs have the same inputs at every
horizon (3.1 variants that offer neither arm's weather, such as timetable-only or a railway
add-one), which test nothing. Ties between runs with different inputs are kept.

`configuration` says how the two runs were configured: "tuned per arm" (nested searches),
"same configuration", or "different flat configuration": fixed-configuration boosters take the
flat search of their own arm (D2), so a forecast run uses the observed-weather configuration and
its railway-only counterpart the railway one. Such a pair compares two pipelines, not inputs alone.

Usage:
    .venv/bin/python frankfurt_munich/advanced_modeling/forecast_comparisons.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import _paths  # noqa: F401  shared modelling code and dataset builders
from advanced_data import ADVANCED_RESULTS
from audit_cv_results import STEPS, active_arms
from compare import compare_runs
from experiment_models import MODELS
from experiment_specs import latest_run

CONTRASTS = {
    "forecast_vs_railway": ({"forecast": True, "weather": False}, {"forecast": False, "weather": False}),
    "forecast_vs_weather": ({"forecast": True, "weather": False}, {"forecast": False, "weather": True}),
    "weather_forecast_vs_weather": ({"forecast": True, "weather": True}, {"forecast": False, "weather": True}),
    "weather_forecast_vs_forecast": ({"forecast": True, "weather": True}, {"forecast": True, "weather": False}),
}
OUTPUT = ADVANCED_RESULTS / "forecast_arm_comparisons.csv"


def evaluation(*paths):
    kinds = {json.loads((path / "config.json").read_text())["evaluation"] for path in paths}
    if not kinds <= {"nested", "prespecified"}:
        return "exploratory"
    return "nested" if "nested" in kinds else "prespecified"


def pairs():
    """(contrast, candidate run, reference run) for every registered matched pair."""
    registered = {run.name: run for run in active_arms(STEPS)}
    for contrast, (candidate_arm, reference_arm) in CONTRASTS.items():
        for run in registered.values():
            if (run.forecast, run.weather) != (candidate_arm["forecast"], candidate_arm["weather"]):
                continue
            reference = replace(run, **reference_arm)
            if reference.name in registered:
                yield contrast, run, reference


def configuration_sources(path, results_dir):
    """The flat searches a run's configuration comes from, through saved ensemble members too."""
    config = json.loads((path / "config.json").read_text())
    sources = {config["configuration_source"]} if config.get("configuration_source") else set()
    for member in config.get("members", []):
        sources |= configuration_sources(results_dir / member, results_dir)
    return sources


def configuration(candidate_path, reference_path, results_dir):
    configs = [json.loads((path / "config.json").read_text()) for path in (candidate_path, reference_path)]
    if all("search" in config for config in configs):
        return "tuned per arm"
    same = (configuration_sources(candidate_path, results_dir)
            == configuration_sources(reference_path, results_dir))
    return "same configuration" if same else "different flat configuration"


def inputs(features):
    """The saved input columns per horizon, as a comparable set."""
    def columns(value):
        if isinstance(value, dict):
            return frozenset(column for part in value.values() for column in part)
        return frozenset(value)
    return {pt: columns(value) for pt, value in features.items()}


def excluded(run, candidate_path, reference_path):
    if run.diagnostic:
        return "recorded-route diagnostic"
    features = [json.loads((path / "config.json").read_text()).get("features")
                for path in (candidate_path, reference_path)]
    if all(features) and inputs(features[0]) == inputs(features[1]):
        return "same inputs"
    return ""


def comparisons(results_dir=ADVANCED_RESULTS):
    tables = []
    for contrast, run, reference in pairs():
        candidate_path, reference_path = (latest_run(results_dir, r) for r in (run, reference))
        task = MODELS[run.model][0]
        table = compare_runs(reference_path, candidate_path, task)
        tables.append(table.assign(
            contrast=contrast, step=run.step, task=task, model=run.model, variant=run.variant,
            evaluation=evaluation(reference_path, candidate_path),
            configuration=configuration(candidate_path, reference_path, results_dir),
            excluded=excluded(run, candidate_path, reference_path),
            candidate=str(candidate_path.relative_to(results_dir)),
            reference=str(reference_path.relative_to(results_dir))))
    columns = ["contrast", "step", "task", "model", "variant", "evaluation", "configuration", "excluded",
               "prediction_time"]
    table = pd.concat(tables, ignore_index=True)
    return table[columns + [c for c in table if c not in columns]]


def family(row):
    if row.model == "lightgbmlss" or row.model.startswith("lightgbmlss"):
        return "lightgbmlss (unstable)"
    if str(row.step) in ("4.2", "6.5"):
        return "distribution models"
    if str(row.step) in ("6.3", "7.1", "7.2", "7.3", "7.4", "7.5"):
        return "EBM and neural networks"
    return "linear and boosters"


def summary(table, by=("contrast", "family", "task")):
    """Comparisons and how many reach D12 in each direction, without the excluded rows."""
    table = table[table["excluded"].fillna("").eq("")]
    threshold = table["task"].map({"severity": 0.003, "delay": 0.1})
    table = table.assign(family=table.apply(family, axis=1),
                         reference_better=table["mean"] >= threshold - 1e-12)
    return table.groupby(list(by)).agg(
        comparisons=("mean", "size"), candidate_reaches_d12=("reaches_minimum_gain", "sum"),
        reference_reaches_d12=("reference_better", "sum"),
        candidate_lower=("mean", lambda m: int((m < 0).sum()))).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ADVANCED_RESULTS)
    args = parser.parse_args()
    table = comparisons(args.results_dir)
    table.to_csv(args.results_dir / OUTPUT.name, index=False)
    print(summary(table).to_string(index=False))


if __name__ == "__main__":
    main()
