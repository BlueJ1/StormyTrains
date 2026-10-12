"""Cross-validate the first models and their references at every prediction time.

- severity_class: multinomial logistic regression, against the training
  class shares;
- arrival_delay_min: linear regression, against the training mean delay.
"""
from __future__ import annotations

import pandas as pd

from cv import run_cv
from models import (make_class_shares, make_linear_regression,
                    make_logistic_regression, make_training_mean)

RUNS = [("class_shares", "severity", make_class_shares),
        ("logistic_regression", "severity", make_logistic_regression),
        ("training_mean", "delay", make_training_mean),
        ("linear_regression", "delay", make_linear_regression)]

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name, task, make_model in RUNS:
        print(f"\n{name} ({task})")
        print(run_cv(name, task, make_model).round(4).to_string(index=False))
