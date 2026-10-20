"""Cross-validate the first models and their references at every prediction time.

- severity_class: multinomial logistic regression, against the training
  class shares;
- arrival_delay_min: linear regression, against the training mean delay.

Both models are run twice: with the railway features only and with the
railway and weather features, on the same rows and folds.
"""
from __future__ import annotations

import pandas as pd

from cv import run_cv
from models import (make_class_shares, make_linear_regression,
                    make_logistic_regression, make_training_mean)

# (name, task, model, weather)
RUNS = [("class_shares", "severity", make_class_shares, False),
        ("logistic_regression", "severity", make_logistic_regression, False),
        ("logistic_regression_weather", "severity", make_logistic_regression, True),
        ("training_mean", "delay", make_training_mean, False),
        ("linear_regression", "delay", make_linear_regression, False),
        ("linear_regression_weather", "delay", make_linear_regression, True)]

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name, task, make_model, weather in RUNS:
        print(f"\n{name} ({task})")
        print(run_cv(name, task, make_model, weather=weather).round(4).to_string(index=False))
