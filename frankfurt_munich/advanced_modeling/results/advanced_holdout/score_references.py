"""Reference scores on the hold-out weeks (finishing plan Q7): class shares and the training mean.

run_final.py did not score the two references, so they were added after the registered run on
24 November 2026. They choose nothing: each is fitted on the labelled rows of the 95 CV weeks and
scored on exactly the hold-out rows of the registered arms (read from the saved predictions),
with the same metric code. Kept here, outside the fingerprinted source folders, so the
registered source stays unchanged.

    .venv/bin/python frankfurt_munich/advanced_modeling/results/advanced_holdout/score_references.py
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import _paths  # noqa: E402,F401  shared modelling code
from cv import TARGETS, class_probabilities, score_fold  # noqa: E402
from data import PREDICTION_TIMES, load_rows  # noqa: E402
from models import make_class_shares, make_training_mean  # noqa: E402

REFERENCES = {"delay": ("training_mean", make_training_mean), "severity": ("class_shares", make_class_shares)}


def main():
    records = []
    for task, (name, factory) in REFERENCES.items():
        target = TARGETS[task]
        holdout = pd.read_parquet(HERE / f"{task}_railway.parquet")
        for pt in PREDICTION_TIMES:
            train = load_rows(pt)
            train = train[train[target].notna()]
            model = factory([], []).fit(pd.DataFrame(index=train.index), train[target])
            rows = holdout[holdout["prediction_time"].eq(pt)].reset_index(drop=True)
            X = pd.DataFrame(index=rows.index)
            pred = class_probabilities(model, X) if task == "severity" else model.predict(X)
            score, _ = score_fold(rows, pt, 0, task, len(train), pred)
            records.append({"task": task, "model": name, **score})
    out = pd.DataFrame(records).drop(columns="fold")
    out.to_csv(HERE / "references.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
