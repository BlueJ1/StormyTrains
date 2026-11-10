"""Screen the engineered weather features on the preliminary models.

Four arms per model, on the same rows and folds, with the recorded-route
diagnostic feature set of the preliminary runs:

- railway: railway features only;
- base: railway + the 14 base weather columns;
- fe_replace: railway + the two endpoint temperatures + the engineered columns;
- fe_add: railway + the 14 base columns + the engineered columns.

Linear and logistic regression have no hyperparameters and are simply
cross-validated. The tuned models are not searched again: each arm uses, per
prediction time, the configuration that the preliminary flat search chose for
it (railway arm: the railway search; the three weather arms: the base-weather
search). This gives the base arm a small head start over the engineered arms.

The railway and base arms are rerun as well, so that every arm comes from the
same code, and their scores are checked against the saved preliminary results.
Everything is written to results_weather_fe/, not to results/.

Usage: run_weather_fe_screen.py [model ...]   (default: all nine)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cv import RESULTS, run_cv
from features import feature_columns
from models import (make_lightgbm, make_lightgbm_classifier, make_linear_regression,
                    make_logistic_regression, make_random_forest, make_random_forest_classifier,
                    make_svr, make_xgboost, make_xgboost_classifier)
from tuning import SCORE, fixed_configurations, run_fixed

SCREEN = Path(__file__).resolve().parent / "results_weather_fe"
FEATURE_SET = "recorded_route_diagnostic"
ARMS = {"railway": False, "base": True, "fe_replace": "fe_replace", "fe_add": "fe_add"}
LINEAR = {"linear_regression": ("delay", make_linear_regression),
          "logistic_regression": ("severity", make_logistic_regression)}
TUNED = {"lightgbm": ("delay", make_lightgbm), "xgboost": ("delay", make_xgboost),
         "random_forest": ("delay", make_random_forest), "svr": ("delay", make_svr),
         "lightgbm_classifier": ("severity", make_lightgbm_classifier),
         "xgboost_classifier": ("severity", make_xgboost_classifier),
         "random_forest_classifier": ("severity", make_random_forest_classifier)}


def run_name(model: str, arm: str) -> str:
    return model + {"railway": "", "base": "_weather", "fe_replace": "_weather_fe_replace",
                    "fe_add": "_weather_fe_add"}[arm]


def latest(name: str, file: str) -> Path:
    """The newest preliminary run of `name` that has `file`."""
    runs = sorted(p for p in (RESULTS / name).glob(f"*/{file}"))
    if not runs:
        raise FileNotFoundError(f"no preliminary run of {name} with {file}")
    return runs[-1]


def check_reproduced(model: str, arm: str, summary: pd.DataFrame, task: str) -> None:
    """The rerun railway and base arms must match the saved preliminary scores."""
    saved = pd.read_csv(latest(run_name(model, arm), "summary.csv")).set_index("prediction_time")
    column = f"{SCORE[task]}_mean"
    rerun = summary.set_index("prediction_time")[column]
    if not np.allclose(rerun.to_numpy(), saved.loc[rerun.index, column].to_numpy(), rtol=0, atol=1e-9):
        raise AssertionError(f"{run_name(model, arm)} does not reproduce its preliminary scores:\n"
                             f"{pd.DataFrame({'rerun': rerun, 'saved': saved[column]})}")


def screen(model: str) -> None:
    linear = model in LINEAR
    task, factory = (LINEAR if linear else TUNED)[model]
    for arm, weather in ARMS.items():
        name = run_name(model, arm)
        start = time.perf_counter()
        print(f"\n{name} ({task})", flush=True)
        if linear:
            summary = run_cv(name, task, factory, weather=weather, feature_set=FEATURE_SET,
                             results_dir=SCREEN)
        else:
            source = latest(run_name(model, "railway" if arm == "railway" else "base"), "config.json")
            chosen = fixed_configurations(source, task=task, model=factory.__name__,
                                          weather=arm != "railway")
            if arm in ("railway", "base"):  # the preliminary features must be the screen's
                saved = json.loads(source.read_text())["features"]
                for pt, columns in saved.items():
                    numeric, categorical = feature_columns(pt, weather, feature_set=FEATURE_SET)
                    assert columns == {"numeric": numeric, "categorical": categorical}, (name, pt)
            summary = run_fixed(name, task, factory, chosen, weather=weather,
                                source=str(source.relative_to(RESULTS.parent)),
                                feature_set=FEATURE_SET, results_dir=SCREEN)
        if arm in ("railway", "base"):
            check_reproduced(model, arm, summary, task)
        print(summary.round(4).to_string(index=False))
        print(f"  {(time.perf_counter() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    names = sys.argv[1:] or [*LINEAR, *TUNED]
    unknown = set(names) - set(LINEAR) - set(TUNED)
    if unknown:
        sys.exit(f"unknown models: {sorted(unknown)}; choose from {[*LINEAR, *TUNED]}")
    for model in names:
        screen(model)
