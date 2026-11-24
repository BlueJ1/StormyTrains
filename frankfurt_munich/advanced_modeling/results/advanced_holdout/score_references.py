"""Score two simple baselines on the hold-out weeks: class shares and the training mean.

The group decided to report these baselines on the hold-out set (finishing plan Q7), but
run_final.py did not score them, so this script scored them after the main hold-out run of
24 November 2026. The baselines use no features, and nothing is chosen with them. Each is fitted
on the labelled rows of the 95 CV weeks and scored on exactly the hold-out rows of the finalists
(read from their saved predictions), with the same metric code.

The script lives here because the code check of the hold-out evaluation covers the Python files
directly inside frankfurt_munich/, modeling/ and advanced_modeling/. Adding it there would have
changed the code that was registered with the finalists.

    .venv/bin/python frankfurt_munich/advanced_modeling/results/advanced_holdout/score_references.py
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import _paths  # noqa: E402,F401  makes the shared modelling code importable
from cv import TARGETS, class_probabilities, score_fold  # noqa: E402
from data import PREDICTION_TIMES, load_rows  # noqa: E402
from models import make_class_shares, make_training_mean  # noqa: E402

# Task -> (name in references.csv, function that builds the baseline model)
BASELINES = {"delay": ("training_mean", make_training_mean), "severity": ("class_shares", make_class_shares)}


def main():
    records = []
    for task, (name, make_baseline) in BASELINES.items():
        target = TARGETS[task]
        # Saved hold-out predictions of the railway-only finalists: they give the scored rows.
        holdout = pd.read_parquet(HERE / f"{task}_railway.parquet")
        for prediction_time in PREDICTION_TIMES:
            train = load_rows(prediction_time)
            train = train[train[target].notna()]
            # No numeric and no categorical features: the model sees only the training labels.
            model = make_baseline([], []).fit(pd.DataFrame(index=train.index), train[target])
            rows = holdout[holdout["prediction_time"].eq(prediction_time)].reset_index(drop=True)
            no_features = pd.DataFrame(index=rows.index)
            prediction = (class_probabilities(model, no_features) if task == "severity"
                          else model.predict(no_features))
            # Fold number 0 is a placeholder; the fold column is dropped below.
            score, _ = score_fold(rows, prediction_time, 0, task, len(train), prediction)
            records.append({"task": task, "model": name, **score})
    out = pd.DataFrame(records).drop(columns="fold")
    out.to_csv(HERE / "references.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
