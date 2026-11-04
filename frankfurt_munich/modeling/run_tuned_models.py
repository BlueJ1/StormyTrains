"""Tune and cross-validate the non-linear models at every prediction time.

- arrival_delay_min (`delay`): random forest, XGBoost, LightGBM and SVR;
- severity_class (`severity`): random forest, XGBoost and LightGBM classifiers.

Each model is tuned by a random search of tuning.N_CONFIGS configurations
(see tuning.py), once with the railway features only and once with the
railway and weather features, on the same rows, folds and configurations.
A classifier uses the same search space as its regressor.

Usage: run_tuned_models.py [model ...]   (default: all seven, delay first)
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from models import (make_lightgbm, make_lightgbm_classifier, make_random_forest,
                    make_random_forest_classifier, make_svr, make_xgboost,
                    make_xgboost_classifier)
from tuning import SPACES, run_search

# name: (task, factory, search space)
MODELS = {"lightgbm": ("delay", make_lightgbm, SPACES["lightgbm"]),
          "xgboost": ("delay", make_xgboost, SPACES["xgboost"]),
          "random_forest": ("delay", make_random_forest, SPACES["random_forest"]),
          "svr": ("delay", make_svr, SPACES["svr"]),
          "lightgbm_classifier": ("severity", make_lightgbm_classifier, SPACES["lightgbm"]),
          "xgboost_classifier": ("severity", make_xgboost_classifier, SPACES["xgboost"]),
          "random_forest_classifier": ("severity", make_random_forest_classifier,
                                       SPACES["random_forest"])}

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    names = sys.argv[1:] or list(MODELS)
    unknown = set(names) - set(MODELS)
    if unknown:
        sys.exit(f"unknown models: {sorted(unknown)}; choose from {list(MODELS)}")
    for model in names:
        task, factory, space = MODELS[model]
        for weather in (False, True):
            name = model + ("_weather" if weather else "")
            start = time.perf_counter()
            print(f"\n{name} ({task})", flush=True)
            summary = run_search(name, task, factory, space, weather=weather)
            print(summary.round(4).to_string(index=False))
            print(f"  {(time.perf_counter() - start) / 60:.1f} min", flush=True)
