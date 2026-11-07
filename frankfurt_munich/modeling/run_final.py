"""Register a matched finalist pair per task and horizon, then score it once."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits

from advanced_data import ADVANCED, load_advanced_rows, read_inputs
from cv import (PROBA_COLUMNS, RESULTS, TARGETS, class_probabilities, cross_validate,
                git_commit, runtime_versions, score_fold)
from data import DATA, HERE, PREDICTION_TIMES, ROW_KEY, TIME_COLUMN, merge_inputs
from ensembles import FittedAverage, apply_isotonic, apply_temperature, fit_isotonic, fit_temperature
from experiment_models import DISTRIBUTION_CLASSIFIERS, DISTRIBUTIONS, MODELS
from experiment_specs import AUTO_VARIANTS, SEARCH_CONFIGS, Run, nested_feature_variants
from features import PHASE2_GROUPS, WEATHER, feature_groups
from folds import cv_folds
from final_analysis import (calibration_reliability, example_risk_trajectories,
                            paired_bootstrap_ci, paired_error_table)
from group_selection import columns_for, greedy_groups, lightgbm_contributions
from holdout_gate import REGISTRY, committed_registration, fingerprint_files, source_fingerprints
from report_cv import run_index
from representations import REPRESENTATIONS, extreme_weather, route_representation
from sequence_inputs import SEQUENCE_COLUMN
from tuning import SCORE, sample_configs, search, select_feature_variant

REFIT_STEPS = {"0.8", "3.3", "5.2", "6.1", "6.2", "6.3", "6.5", "7.1", "7.2", "7.3", "7.4", "7.5"}
FEATURE_FILES = {
    frozenset(PHASE2_GROUPS["recent_reliability"][0]): "reliability",
    frozenset(PHASE2_GROUPS["network_state"][0]): "network",
    frozenset((SEQUENCE_COLUMN,)): "sequence",
}


def matched_candidates(results_dir):
    pairs = {}
    for path, config, scores in run_index(results_dir):
        if config["step"] not in REFIT_STEPS or config["evaluation"] not in ("nested", "prespecified"):
            continue
        if config.get("variant", "").startswith("weather_"):
            continue
        if config["model"] in ("tabm", "tabm_classifier"):
            continue  # PyTabKit 1.8 cannot refit TabM on all CV training rows.
        key = (config["step"], config["model"], config.get("variant", ""), config.get("calendar"))
        pairs.setdefault(key, {})[config["weather"]] = (path, config, scores.set_index("prediction_time"))
    return [pair for pair in pairs.values() if pair.keys() == {False, True}]


def choose_finalists(results_dir):
    """Rank matched pairs by the railway-only CV score, independently by task and horizon."""
    pairs = matched_candidates(results_dir)
    for task in SCORE:
        metric = f"{SCORE[task]}_mean"
        for pt in PREDICTION_TIMES:
            eligible = [pair for pair in pairs if pair[False][1]["task"] == task
                        and all(pt in arm[2].index for arm in pair.values())
                        and matched_features(pair[False][1]["features"][pt],
                                             pair[True][1]["features"][pt])
                        and pair[False][1].get("search") == pair[True][1].get("search")]
            if not eligible:
                raise ValueError(f"no matched finalist candidates for {task} at {pt}")
            best = min(eligible, key=lambda pair: pair[False][2].loc[pt, metric])
            yield task, pt, best


def matched_features(railway, with_weather):
    weather_numeric = [column for column in with_weather["numeric"] if column in WEATHER]
    railway_numeric = [column for column in with_weather["numeric"] if column not in WEATHER]
    return (set(weather_numeric) == set(WEATHER) and railway_numeric == railway["numeric"]
            and with_weather["categorical"] == railway["categorical"])


def registered_factory(spec):
    _, factory, _ = MODELS[spec["model"]]
    if spec.get("calendar") not in (None, "native"):
        factory = partial(factory, calendar=spec["calendar"])
    numeric = list(spec["features"]["numeric"])
    categorical = list(spec["features"]["categorical"])
    variant = spec.get("variant", "")
    if spec["step"] == "3.3" and variant == "l1":
        factory = partial(factory, l1=True)
    if variant in REPRESENTATIONS or variant == "route_target":
        rep = route_representation(variant, spec["task"])
        raw_numeric = [column for column in numeric if column not in rep.extra_inputs]
        factory, numeric, categorical = rep.wrap(factory, raw_numeric, categorical)
    return factory, numeric, categorical


def final_group_selection(config, prediction_time, task, inputs_dir):
    """Select railway groups on CV and reuse them in both registered arms."""
    groups = feature_groups(prediction_time)
    numeric, categorical = columns_for(groups, groups)
    rows = load_advanced_rows(prediction_time, inputs_dir=inputs_dir, extra_inputs=tuple(numeric + categorical))
    factory = MODELS[config["model"]][1]

    def evaluate(selected):
        columns = columns_for(groups, selected)
        metrics, _ = cross_validate(rows, prediction_time, task, factory, *columns)
        return metrics[SCORE[task]].mean()

    return greedy_groups(groups, evaluate, config["variant"])


def tune_finalist(spec, config, inputs_dir):
    task, pt, weather, step = (spec[key] for key in ("task", "prediction_time", "weather", "step"))
    n_configs = SEARCH_CONFIGS if step in ("5.2", "6.2") else config.get("search", {}).get("n_configs")
    if n_configs:
        n_configs = min(n_configs, SEARCH_CONFIGS)
        factory, numeric, categorical = registered_factory(spec)
        configs = sample_configs(MODELS[spec["model"]][2], n=n_configs)
        rows = load_advanced_rows(pt, weather=weather, inputs_dir=inputs_dir, extra_inputs=tuple(numeric + categorical))
        table, _, _ = search(rows, pt, task, factory, configs, numeric, categorical)
        policy = {"method": "five CV week folds", "n_configs": n_configs}
        if step == "5.2":
            ids = table.nsmallest(5, f"{SCORE[task]}_mean").config_id.astype(int).tolist()
            policy["members"] = 5
            return {"params": {}, "member_params": [configs[i] for i in ids], "config_ids": ids,
                    "tuning_policy": policy}
        selected = int(table.loc[table["chosen"], "config_id"].item())
        policy["config_id"] = selected
        if step == "6.2":
            policy["calibration"] = "fit on predictions from the five CV week folds at the chosen configuration"
        return {"params": configs[selected], "tuning_policy": policy}
    if step == "7.3":
        budget = float(np.median(list(config["cpu_budgets"][pt].values()))) * 1.25
        return {"params": {"cpu_budget": budget}, "tuning_policy": {
            "method": "inner-week gate search at 1.25 times the median CV CPU budget"}}
    return {"params": {}, "tuning_policy": {"method": "same as CV"}}


def tune_registered_finalist(record, *, inputs_dir):
    spec, config, feature_policy = record
    with threadpool_limits(limits=1):
        spec.update(tune_finalist(spec, config, inputs_dir))
    if feature_policy:
        spec["tuning_policy"]["feature_selection"] = feature_policy
    return spec


def register(results_dir, inputs_dir, registry, jobs=1):
    """Choose configurations using all CV weeks, with zero access to the hold-out labels."""
    if registry.exists():
        raise FileExistsError(f"registration already exists: {registry}")
    plan = (HERE / "PLAN.md").read_text()
    unfinished = re.findall(r"(?m)^\| ([0-8]\.\d+) \| .* \| (?:todo|running) \|$", plan)
    if unfinished:
        raise PermissionError(f"finish the CV plan before registering finalists: {', '.join(unfinished)}")
    sources = source_fingerprints(HERE.parent)
    records = []
    for task, pt, pair in choose_finalists(results_dir):
        railway = pair[False][1]
        variant = railway.get("variant", "")
        selection = None
        selected_groups = None
        if variant in AUTO_VARIANTS:
            _, factory, space = MODELS[railway["model"]]
            run = Run(railway["step"], railway["model"], variant=variant)
            candidates = nested_feature_variants(run, pt, task, factory)
            inputs = tuple(dict.fromkeys(c for _, num, cat in candidates.values() for c in num + cat))
            rows = load_advanced_rows(pt, inputs_dir=inputs_dir, extra_inputs=inputs)
            selection = select_feature_variant(rows, pt, task, candidates,
                                               sample_configs(space, n=SEARCH_CONFIGS), cv_folds)
        elif railway["step"] == "3.3" and variant in ("forward", "backward"):
            selected_groups = final_group_selection(railway, pt, task, inputs_dir)
        for weather in (False, True):
            source, config, scores = pair[weather]
            spec = {key: config[key] for key in ("model", "task", "weather", "step")}
            spec.update(prediction_time=pt, variant=config.get("variant", ""),
                        calendar=config.get("calendar"), features=config["features"][pt],
                        source=str(source.relative_to(results_dir)),
                        cv_score=float(scores.loc[pt, f"{SCORE[task]}_mean"]))
            if selection is not None:
                run = Run(config["step"], config["model"], weather, variant=variant)
                _, numeric, categorical = nested_feature_variants(run, pt, task, factory)[selection]
                spec.update(variant=selection, features={"numeric": numeric, "categorical": categorical})
            if selected_groups is not None:
                groups = feature_groups(pt, weather)
                weather_groups = groups.keys() - feature_groups(pt).keys()
                numeric, categorical = columns_for(groups, selected_groups | weather_groups)
                spec["features"] = {"numeric": numeric, "categorical": categorical}
            feature_policy = (f"{variant} on the railway CV arm"
                              if selection is not None or selected_groups is not None else None)
            records.append((spec, config, feature_policy))
    development = evaluation_fingerprints({"finalists": [spec for spec, _, _ in records]},
                                          inputs_dir, splits=("cv",))
    tune = partial(tune_registered_finalist, inputs_dir=inputs_dir)
    if jobs == 1:
        finalists = list(map(tune, records))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            finalists = list(pool.map(tune, records))
    for spec in finalists:
        print(spec["task"], spec["prediction_time"], spec["model"],
              "weather" if spec["weather"] else "railway", flush=True)
    registration = {"policy": "one matched pair ranked by railway-only CV score per task and horizon",
                    "git_commit": git_commit(), "runtime": runtime_versions(),
                    "development_fingerprints": development, "finalists": finalists}
    if (source_fingerprints(HERE.parent) != sources
            or evaluation_fingerprints(registration, inputs_dir, splits=("cv",)) != development):
        raise PermissionError("development code or inputs changed during registration")
    with registry.open("x") as file:
        json.dump(registration, file, indent=2)
        file.write("\n")


def committed_registry(path):
    """Refuse to open hold-out labels before the exact registration is committed."""
    registration = committed_registration(path)
    validate_registry(registration)
    return registration


def validate_registry(registration):
    finalists = registration["finalists"]
    for spec in finalists:
        if spec["step"] == "5.2" and (spec["variant"] != "top5_nested" or len(spec.get("member_params", [])) != 5):
            raise ValueError("registered configuration bagging requires five members")
        if spec["step"] == "6.2" and (spec["task"] != "severity" or spec["variant"] not in ("temperature", "isotonic")):
            raise ValueError("registered calibration requires a severity classifier and calibration method")
    keys = [(spec["task"], spec["prediction_time"], spec["weather"]) for spec in finalists]
    expected = {(task, pt, weather) for task in SCORE for pt in PREDICTION_TIMES
                for weather in (False, True)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("registration must have one matched pair for each task and prediction time")
    specs = dict(zip(keys, finalists))
    for task in SCORE:
        for pt in PREDICTION_TIMES:
            a, b = (specs[(task, pt, weather)] for weather in (False, True))
            if (any(a[key] != b[key] for key in ("model", "step", "variant", "calendar"))
                    or not matched_features(a["features"], b["features"])):
                raise ValueError(f"unmatched weather comparison for {task} at {pt}")
            policy = [{key: value for key, value in spec["tuning_policy"].items()
                       if key != "config_id"} for spec in (a, b)]
            if policy[0] != policy[1]:
                raise ValueError(f"unmatched tuning policy for {task} at {pt}")


def fitted_model(spec, rows, factory, numeric, categorical):
    models = [factory(numeric, categorical, **params)
              for params in spec.get("member_params", [spec["params"]])]
    model = models[0]
    features = numeric + categorical
    if hasattr(model, "fit_joint"):
        model.fit_joint(rows[features], rows[TARGETS["severity"]], rows[TARGETS["delay"]])
    else:
        target = TARGETS[spec["task"]]
        rows = rows[rows[target].notna()]
        for member in models:
            member.fit(rows[features], rows[target])
    if len(models) > 1:
        model = FittedAverage(models, spec["task"])
    elif spec["step"] == "6.2":
        fixed = partial(factory, **spec["params"])
        _, oof = cross_validate(rows, spec["prediction_time"], spec["task"], fixed, numeric, categorical)
        labels, probabilities = oof[TARGETS["severity"]], oof[PROBA_COLUMNS].to_numpy()
        if spec["variant"] == "temperature":
            calibrate = partial(apply_temperature, temperature=fit_temperature(labels, probabilities))
        else:
            calibrate = partial(apply_isotonic, models=fit_isotonic(labels, probabilities))
        model = FittedAverage(models, spec["task"], calibrate)
    return model, len(rows)


def explanation(model, X, background, task):
    """Mean absolute TreeSHAP when available, else sampled permutation SHAP."""
    if hasattr(model, "named_steps") and "categories" in model.named_steps:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "booster_"):
            return lightgbm_contributions(model, X)
        if hasattr(estimator, "estimator"):
            estimator = estimator.estimator
        if hasattr(estimator, "get_booster"):
            import xgboost as xgb
            encoded = model.named_steps["categories"].transform(X)
            contribution = estimator.get_booster().predict(
                xgb.DMatrix(encoded, enable_categorical=True), pred_contribs=True)
            values = contribution.reshape(len(X), -1, len(X.columns) + 1)[..., :-1]
            return pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=(0, 1))})
    rng = np.random.default_rng(42)
    sample = X.iloc[rng.choice(len(X), size=min(64, len(X)), replace=False)].reset_index(drop=True)
    columns = list(sample)
    repeats = 8
    contributions = None

    def predict(frame):
        value = class_probabilities(model, frame) if task == "severity" else model.predict(frame)
        return np.asarray(value).reshape(len(frame), -1)

    for _ in range(repeats):
        base = background.iloc[rng.integers(len(background), size=len(sample))].reset_index(drop=True)
        before = predict(base)
        if contributions is None:
            contributions = np.zeros((len(columns), *before.shape))
        for i in rng.permutation(len(columns)):
            column = columns[i]
            base[column] = sample[column]
            after = predict(base)
            contributions[i] += after - before
            before = after
    return pd.DataFrame({"feature": columns,
                         "mean_abs_shap": np.abs(contributions / repeats).mean(axis=(1, 2))})


def score_registered_arm(registry, inputs_dir, output, index):
    """Run one arm in its own process so its native libraries stay isolated."""
    registration = committed_registry(registry)
    if not (output / ".evaluation.lock").is_file() or (output / "completed.txt").exists():
        raise PermissionError("no active registered hold-out evaluation")
    manifest = (output / "registration.json").read_bytes()
    if manifest != registry.read_bytes() or json.loads(manifest) != registration:
        raise PermissionError("evaluation directory has a different registration")
    provenance = json.loads((output / "provenance.json").read_text())
    if provenance != evaluation_fingerprints(registration, inputs_dir):
        raise PermissionError("evaluation code or inputs changed")
    if not 0 <= index < len(registration["finalists"]):
        raise ValueError("registered arm index is out of range")
    spec = registration["finalists"][index]
    arm = output / "_arms" / str(index)
    arm.mkdir(parents=True, exist_ok=True)
    if (arm / "score.json").exists():
        return
    task, pt, weather = (spec[key] for key in ("task", "prediction_time", "weather"))
    factory, numeric, categorical = registered_factory(spec)
    features = numeric + categorical
    options = {"weather": weather, "inputs_dir": inputs_dir, "extra_inputs": tuple(features)}
    cv = load_advanced_rows(pt, **options)
    model, n_train = fitted_model(spec, cv, factory, numeric, categorical)
    holdout = load_advanced_rows(pt, split="holdout", allow_holdout=True, **options)
    rows = holdout[holdout[TARGETS[task]].notna()]
    X = rows[features]
    pred = class_probabilities(model, X) if task == "severity" else model.predict(X)
    score, prediction = score_fold(rows, pt, 0, task, n_train, pred)
    prediction = prediction.drop(columns="fold")
    background = cv.loc[cv[TARGETS[task]].notna(), features]
    shap = explanation(model, X, background, task).assign(task=task, prediction_time=pt, weather=weather)
    metadata = pd.DataFrame()
    if weather:
        timetable = read_inputs(inputs_dir / "timetable_holdout.parquet", ("weather_branch",))
        metadata = merge_inputs(holdout[[*ROW_KEY, TIME_COLUMN, "severity_class"]], timetable)
        flags = extreme_weather(holdout).filter(regex=r"^weather_(heat|gust|snow|heavy_rain)$")
        metadata = metadata.join(flags).drop(columns="predicted_at")
    prediction.to_parquet(arm / "predictions.parquet", index=False)
    shap.to_parquet(arm / "shap.parquet", index=False)
    if spec["weather"]:
        metadata.to_parquet(arm / "metadata.parquet", index=False)
    (arm / "environment.json").write_text(json.dumps({
        "interpreter": sys.executable, "runtime": runtime_versions()}, indent=2) + "\n")
    temporary = arm / "score.tmp"
    temporary.write_text(json.dumps(score))
    temporary.replace(arm / "score.json")


def evaluation_fingerprints(registration, inputs_dir, *, splits=("cv", "holdout")):
    """Fingerprint the pipeline source and requested model-input splits."""
    required = {"timetable", "history", "weather"}
    features = {c for spec in registration["finalists"] for names in spec["features"].values() for c in names}
    for group, filename in FEATURE_FILES.items():
        if features & group:
            required.add(filename)
    paths = {}
    for split in splits:
        for name in ("journeys", "prediction_rows"):
            paths[f"data/{name}_{split}.parquet"] = DATA / f"{name}_{split}.parquet"
        for name in required:
            paths[f"advanced/{name}_{split}.parquet"] = inputs_dir / f"{name}_{split}.parquet"
    return source_fingerprints(HERE.parent) | fingerprint_files(paths)


def final_evaluation(registry, results_dir, inputs_dir, distribution_python=None):
    registration = committed_registry(registry)
    registry_bytes = registry.read_bytes()
    if json.loads(registry_bytes) != registration:
        raise PermissionError("finalist registration changed during validation")
    development = evaluation_fingerprints(registration, inputs_dir, splits=("cv",))
    if development != registration["development_fingerprints"]:
        raise PermissionError("development code or inputs changed after registration")
    dist_models = set(DISTRIBUTIONS) | set(DISTRIBUTION_CLASSIFIERS)
    if any(spec["model"] in dist_models for spec in registration["finalists"]):
        if distribution_python is None or not distribution_python.is_file():
            raise ValueError("set --distribution-python to the pinned distribution-model interpreter")
    output = results_dir / "advanced_holdout"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.txt").exists():
        raise PermissionError("the registered hold-out evaluation is complete")
    lock = output / ".evaluation.lock"
    with lock.open("x"):
        pass
    try:
        if (output / "completed.txt").exists():
            raise PermissionError("the registered hold-out evaluation is complete")
        manifest = output / "registration.json"
        if manifest.exists() and manifest.read_bytes() != registry_bytes:
            raise PermissionError("evaluation directory has a different registration")
        manifest.write_bytes(registry_bytes)
        provenance = output / "provenance.json"
        if not provenance.exists() and (output / "_arms").exists():
            raise PermissionError("existing evaluation arms have no provenance")
        fingerprints = evaluation_fingerprints(registration, inputs_dir)
        if any(fingerprints.get(name) != digest for name, digest in development.items()):
            raise PermissionError("development code or inputs changed during evaluation setup")
        if provenance.exists() and json.loads(provenance.read_text()) != fingerprints:
            raise PermissionError("evaluation code or inputs changed")
        temporary = output / "provenance.tmp"
        temporary.write_text(json.dumps(fingerprints, indent=2) + "\n")
        temporary.replace(provenance)
        run_final_arms(registration, registry, inputs_dir, output, distribution_python)
    finally:
        lock.unlink()
    print(output, flush=True)


def run_final_arms(registration, registry, inputs_dir, output, distribution_python):
    metric_rows, shap_rows, environments = [], [], []
    predictions, metadata = {}, {}
    dist_models = set(DISTRIBUTIONS) | set(DISTRIBUTION_CLASSIFIERS)
    for index, spec in enumerate(registration["finalists"]):
        arm_dir = output / "_arms" / str(index)
        if not (arm_dir / "score.json").exists():
            python = distribution_python if spec["model"] in dist_models else sys.executable
            subprocess.run([str(python), __file__, "evaluate", "--registry", str(registry),
                            "--inputs-dir", str(inputs_dir), "--results-dir", str(output),
                            "--arm-index", str(index)], check=True)
        scores = json.loads((arm_dir / "score.json").read_text())
        environment = json.loads((arm_dir / "environment.json").read_text())
        environments.append({"task": spec["task"], "prediction_time": spec["prediction_time"],
                             "weather": spec["weather"], **environment})
        arm = (spec["task"], spec["weather"])
        predictions.setdefault(arm, []).append(pd.read_parquet(arm_dir / "predictions.parquet"))
        if spec["weather"]:
            metadata.setdefault(spec["task"], []).append(pd.read_parquet(arm_dir / "metadata.parquet"))
        shap_rows.append(pd.read_parquet(arm_dir / "shap.parquet"))
        metric_rows.append({**scores, "step": spec["step"], "model": spec["model"], "variant": spec["variant"],
                            "task": spec["task"], "weather": spec["weather"]})
    pd.DataFrame(metric_rows).drop(columns="fold").to_csv(output / "scores.csv", index=False)
    (output / "environments.json").write_text(json.dumps(environments, indent=2) + "\n")
    if shap_rows:
        pd.concat(shap_rows, ignore_index=True).to_csv(output / "shap.csv", index=False)
    for task in SCORE:
        left, right = (pd.concat(predictions[(task, weather)], ignore_index=True) for weather in (False, True))
        paired_bootstrap_ci(left, right, task).to_csv(output / f"{task}_weather.csv", index=False)
        meta = pd.concat(metadata[task], ignore_index=True)
        errors = [paired_error_table(left[left["prediction_time"].eq(pt)],
                                     right[right["prediction_time"].eq(pt)], task, metadata=meta)
                  .assign(prediction_time=pt) for pt in PREDICTION_TIMES]
        pd.concat(errors, ignore_index=True).to_csv(output / f"{task}_errors.csv", index=False)
        calibration, risk = {}, {}
        for arm, frame in (("railway", left), ("weather", right)):
            frame.to_parquet(output / f"{task}_{arm}.parquet", index=False)
            if task == "severity":
                calibration[arm] = calibration_reliability(frame)
                risk[arm] = example_risk_trajectories(frame)
                calibration[arm].to_csv(output / f"calibration_{arm}.csv", index=False)
                risk[arm].to_csv(output / f"risk_{arm}.csv", index=False)
        if task == "severity":
            from result_plots import calibration_plot, risk_plot
            calibration_plot(calibration, output / "calibration.svg")
            risk_plot(risk, output / "risk.svg")
    (output / "completed.txt").write_text(datetime.now().isoformat(timespec="seconds") + "\n")
    shutil.rmtree(output / "_arms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "evaluate"))
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--inputs-dir", type=Path, default=ADVANCED)
    parser.add_argument("--distribution-python", type=Path)
    parser.add_argument("--jobs", type=int, default=1, help="parallel registration workers")
    parser.add_argument("--arm-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs < 1 or (args.action != "register" and args.jobs != 1):
        parser.error("--jobs must be positive and applies only to registration")
    with threadpool_limits(limits=1):
        if args.action == "register":
            register(args.results_dir, args.inputs_dir, args.registry, jobs=args.jobs)
        elif args.arm_index is not None:
            score_registered_arm(args.registry, args.inputs_dir, args.results_dir, args.arm_index)
        else:
            final_evaluation(args.registry, args.results_dir, args.inputs_dir, args.distribution_python)


if __name__ == "__main__":
    main()
