"""Tune and cross-validate the non-linear delay models at every prediction time.

Random forest, XGBoost, LightGBM and SVR for arrival_delay_min, each by a
random search of tuning.N_CONFIGS configurations (see tuning.py), once with
the railway features only and once with the railway and weather features, on
the same rows, folds and configurations.

Usage: run_tuned_models.py [model ...]   (default: all four, fastest first)
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from models import make_lightgbm, make_random_forest, make_svr, make_xgboost
from tuning import SPACES, run_search

MODELS = {"lightgbm": make_lightgbm, "xgboost": make_xgboost,
          "random_forest": make_random_forest, "svr": make_svr}

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    names = sys.argv[1:] or list(MODELS)
    unknown = set(names) - set(MODELS)
    if unknown:
        sys.exit(f"unknown models: {sorted(unknown)}; choose from {list(MODELS)}")
    for model in names:
        for weather in (False, True):
            name = model + ("_weather" if weather else "")
            start = time.perf_counter()
            print(f"\n{name} (delay)", flush=True)
            summary = run_search(name, "delay", MODELS[model], SPACES[model], weather=weather)
            print(summary.round(4).to_string(index=False))
            print(f"  {(time.perf_counter() - start) / 60:.1f} min", flush=True)
