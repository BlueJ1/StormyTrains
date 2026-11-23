"""Output audit of the CV runs: every active arm, its latest run directory.

Checks per arm: the run files exist; the run was made from the given commit
with a clean tree; the summary has exactly the expected horizons and its means
equal the fold means; each horizon has five folds; the out-of-fold predictions
cover each CV row of the task once (severity: all rows, delay: completed
journeys), in the row's committed fold, and are finite (probabilities
nonnegative, summing to one); and the fold scores recomputed from them equal
fold_metrics.csv.

Usage (finishing plan B4 used commit 7e0e96a for the railway and weather arms;
B17 audits the forecast arms of D17 on their own commit with
--arm forecast weather_forecast):
    .venv/bin/python frankfurt_munich/advanced_modeling/audit_cv_results.py --commit <sha> [--arm ...]

By default only the railway and observed-weather arms are audited. --commit takes several
hashes when arms were rerun at a later commit; each run must record one of them.

--accept-b17-dirty-flag exists for B17 only (students' decision, 20 November 2026):
the forecast runs were made from commit 7300d61, before cv.git_commit anchored its results/
exclusion at the repository top. Unanchored, it applied only below frankfurt_munich/modeling, so
every run started after the first new run folder appeared recorded "<sha> (uncommitted changes)"
although only results/ had changed. With the option the audit accepts that flag on the forecast
arms for that commit only, and only when it is one of the commits given.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  shared modelling code and dataset builders
from advanced_data import ADVANCED_RESULTS
from cv import PROBA_COLUMNS, TARGETS
from data import PREDICTION_TIMES, load_rows
from experiment_models import MODELS, NEURAL
from experiment_specs import ARMS, DECOMPOSED, LATE_STEPS, LATE_VARIANTS, VARIANTS, registered_runs
from features import CURRENT_DELAY_TIMES
from metrics import class_log_loss, rmse

STEPS = ["0.7", "0.8", *VARIANTS, "3.1", "3.2", "3.3", "3.4", *DECOMPOSED, "4.2", "5.1", "5.2",
         "5.3", "5.4", "6.1", "6.2", "6.3", "6.5", "6.6", *NEURAL, "8.2", "8.3", "8.4"]


def active_arms(steps):
    arms = []
    for step in steps:
        for run in registered_runs(step):
            arms.append(run)
            if step == "0.8":
                arms.append(replace(run, variant="auto_groups"))
    return arms


def task_of(run):
    model = run.model
    if model in MODELS:
        return MODELS[model][0]
    return "severity" if "classifier" in model or model.endswith("_severity") else "delay"


ROWS = {}


def cv_rows(pt):
    if pt not in ROWS:
        ROWS[pt] = load_rows(pt)[["journey_id", "fold", "severity_class", "arrival_delay_min"]]
    return ROWS[pt]


DIRTY_FLAG = " (uncommitted changes)"
B17_COMMIT = "7300d61bf69e7c504b15a327c48092b6a8532481"
DEFAULT_ARMS = ("railway", "weather")


def commit_ok(recorded, commits, run, accept_flagged=False):
    commits = (commits,) if isinstance(commits, str) else tuple(commits)
    if recorded in commits:
        return True
    return (accept_flagged and run.forecast and B17_COMMIT in commits
            and recorded == B17_COMMIT + DIRTY_FLAG)


def audit(run, commit, results_dir, accept_flagged=False):
    problems = []
    root = results_dir / run.name
    runs = sorted(p for p in root.glob("*") if (p / "config.json").is_file()) if root.is_dir() else []
    if not runs:
        return None, ["no completed run"]
    out = runs[-1]
    for name in ("config.json", "summary.csv", "fold_metrics.csv"):
        if not (out / name).is_file():
            problems.append(f"missing {name}")
    if problems:
        return out, problems
    config = json.loads((out / "config.json").read_text())
    if not commit_ok(config.get("git_commit"), commit, run, accept_flagged):
        problems.append(f"git_commit {config.get('git_commit')!r}")
    task = config.get("task") or task_of(run)
    # the current-delay group exists only at 1h, 20min and 0min, so its 3.1 ablations cover only those
    late = run.step in LATE_STEPS or run.variant in LATE_VARIANTS or (run.step == "3.1" and run.variant in ("add_delay", "drop_delay"))
    expected = [pt for pt in PREDICTION_TIMES if not late or pt in CURRENT_DELAY_TIMES]
    summary = pd.read_csv(out / "summary.csv").set_index("prediction_time")
    folds = pd.read_csv(out / "fold_metrics.csv")
    if sorted(summary.index) != sorted(expected):
        problems.append(f"horizons {list(summary.index)} != {expected}")
    metrics = ["log_loss", "macro_f1"] if task == "severity" else ["rmse"]
    for pt in summary.index:
        fm = folds[folds.prediction_time.eq(pt)]
        if sorted(fm.fold) != [1, 2, 3, 4, 5]:
            problems.append(f"{pt}: folds {sorted(fm.fold)}")
        for m in metrics:
            if m in fm and not np.isclose(fm[m].mean(), summary.loc[pt, f"{m}_mean"], rtol=0, atol=1e-9):
                problems.append(f"{pt}: summary {m} != fold mean")
    oof_path = out / "oof_predictions.parquet"
    if not oof_path.is_file():
        problems.append("missing oof_predictions.parquet")
        return out, problems
    oof = pd.read_parquet(oof_path)
    target = TARGETS[task]
    for pt in summary.index:
        o = oof[oof.prediction_time.eq(pt)]
        rows = cv_rows(pt)
        if task == "delay":
            rows = rows[rows[target].notna()]
        if o.journey_id.duplicated().any():
            problems.append(f"{pt}: duplicated journeys in OOF")
        if set(o.journey_id) != set(rows.journey_id):
            problems.append(f"{pt}: OOF rows {len(o)} vs CV rows {len(rows)}")
            continue
        merged = o.merge(rows, on="journey_id", suffixes=("", "_ref"))
        if not merged.fold.astype(int).equals(merged.fold_ref.astype(int)):
            problems.append(f"{pt}: OOF fold differs from committed fold")
        if task == "severity":
            p = merged[PROBA_COLUMNS].to_numpy()
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
                problems.append(f"{pt}: invalid probabilities")
                continue
        else:
            if not np.isfinite(merged.predicted_delay).all():
                problems.append(f"{pt}: non-finite delays")
                continue
        fm = folds[folds.prediction_time.eq(pt)].set_index("fold")
        for fold, part in merged.groupby("fold_ref"):
            y = part[f"{target}_ref"]
            if task == "severity":
                score = {"log_loss": class_log_loss(y, part[PROBA_COLUMNS].to_numpy())}
            else:
                score = {"rmse": rmse(y, part.predicted_delay.to_numpy())}
            for m, value in score.items():
                if m in fm and not np.isclose(value, fm.loc[fold, m], rtol=1e-7, atol=1e-7):
                    problems.append(f"{pt} fold {fold}: recomputed {m} {value:.6f} != {fm.loc[fold, m]:.6f}")
    return out, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", default=STEPS)
    parser.add_argument("--commit", nargs="+", required=True,
                        help="full commit hashes; each run must record one of them")
    parser.add_argument("--results-dir", type=Path, default=ADVANCED_RESULTS)
    parser.add_argument("--arm", nargs="+", choices=ARMS, default=DEFAULT_ARMS,
                        help="audit only these arms (default: railway and observed weather)")
    parser.add_argument("--accept-b17-dirty-flag", action="store_true",
                        help=f"B17 only: accept '{B17_COMMIT[:7]} (uncommitted changes)' on the forecast arms")
    args = parser.parse_args()
    arms = [run for run in active_arms(args.steps) if run.arm in args.arm]
    failed = 0
    for run in arms:
        out, problems = audit(run, args.commit, args.results_dir, args.accept_b17_dirty_flag)
        if problems:
            failed += 1
            print(f"FAIL {run.name}: {'; '.join(problems)}")
    print(f"{len(arms) - failed} of {len(arms)} arms passed")


if __name__ == "__main__":
    main()
