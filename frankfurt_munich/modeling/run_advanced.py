"""Run the registered cutoff-audited CV reference models and feature screens."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from advanced_data import ADVANCED, load_advanced_rows
from compare import compare_runs
from corridor_transfer import evaluate_corridor_transfer
from cv import RESULTS, TARGETS, cross_validate, git_commit, pooled_cv, save_run
from data import PREDICTION_TIMES, TIME_COLUMN, load_rows
from ensembles import cv_ensemble, ensemble_recipe, saved_average
from experiment_models import LIGHTGBM, LINEAR, MODELS, NEURAL, fixed_factory, flat_choices, flat_search_path, variant_model
from experiment_specs import (AUTO_VARIANTS, DECOMPOSED, LATE_STEPS, LATE_VARIANTS, SEARCH_CONFIGS,
                              VARIANTS, Run, latest_run, nested_feature_variants, registered_runs,
                              tuning_reference)
from features import CURRENT_DELAY_TIMES, PHASE2_GROUPS, feature_columns, feature_groups
from group_selection import ablations, columns_for, cv_importance, nested_groups
from models import L1_LOGISTIC, make_lightgbm_classifier
from network_transfer import NETWORK_FEATURE, inference_context, network_cv
from representations import REPRESENTATIONS, route_representation
from sequence_inputs import SEQUENCE_COLUMN
from tuning import FEATURE_SCREEN_CONFIGS, SCORE, nested_search, sample_configs

def evaluate_run(run: Run, *, inputs_dir: Path, results_dir: Path, prediction_times):
    """Save one run using the same predictions and metrics format as the original CV."""
    revision = git_commit()
    if run.step == "8.2":
        return evaluate_pool(run, inputs_dir, results_dir, prediction_times, revision)
    if run.step == "5.1":
        metrics, predictions, sources = saved_average(run, results_dir, prediction_times)
        config = {"name": run.name, "step": run.step, "task": MODELS[run.model][0], "model": run.model,
                  "weather": run.weather, "evaluation": "exploratory", "variant": run.variant,
                  "members": [str(path.relative_to(results_dir)) for path in sources],
                  "git_commit": revision}
        out, _ = save_run(results_dir, run.name, config, metrics, predictions)
        print(out, flush=True)
        return out
    task, factory, space = MODELS[run.model]
    nested_members = run.variant == "top5_nested" or run.step == "6.2"
    source_model = LIGHTGBM[1] if run.step == "4.2" else run.model
    source = None if run.nested or nested_members else flat_search_path(source_model, run.weather)
    chosen = dict(flat_choices(source_model, run.weather)) if source else {}
    evaluation = "exploratory" if source else "prespecified"
    if run.nested:
        configs = sample_configs(space, n=8 if run.step in NEURAL else run.n_configs)
        evaluation = "nested"
    if run.step.startswith(("1.", "2.")) or run.step == "3.1" or run.diagnostic:
        evaluation = "exploratory"
    if run.diagnostic:
        feature_set = "recorded_route_diagnostic"
    elif run.variant in PHASE2_GROUPS or run.variant == "all_groups":
        feature_set = run.variant
    else:
        feature_set = "base"
    all_metrics, all_predictions, features, tables = [], [], {}, []
    group_choices, permutation, shap, combinations, budgets = [], [], [], {}, {}
    dummy_reference = run.model in ("class_shares", "training_mean")
    with threadpool_limits(limits=1):
        for prediction_time in prediction_times:
            late_only = run.step in LATE_STEPS or run.variant in LATE_VARIANTS
            if late_only and prediction_time not in CURRENT_DELAY_TIMES:
                continue
            numeric, categorical = feature_columns(prediction_time, run.weather,
                                                   feature_set=feature_set)
            if dummy_reference:
                numeric, categorical = [], []
            if run.step == "1.3" or run.variant.startswith("route_"):
                categorical.append("route_id")
            if run.step.startswith("3."):
                group_map = feature_groups(prediction_time, run.weather)
                selected = select_groups(run, group_map)
                if not selected:
                    continue
                numeric, categorical = columns_for(group_map, selected)
            params = chosen[prediction_time]["params"] if source else {}
            if run.step == "4.2":
                cancellation = partial(make_lightgbm_classifier, **params, objective=None)
                make_model = partial(factory, classifier_factory=cancellation)
            else:
                make_model = partial(factory, **params) if params else factory
            if run.variant == "l1":
                make_model = partial(make_model, l1=True)
            if run.step in NEURAL:
                make_model = partial(make_model, calendar=run.calendar)
                if run.step in ("7.3", "7.4"):
                    numeric.append(TIME_COLUMN)
                if run.step == "7.5":
                    numeric.append(SEQUENCE_COLUMN)
            if run.step in DECOMPOSED or run.step == "6.6":
                make_model = partial(variant_model, run.variant,
                                     {} if run.nested else chosen[prediction_time]["params"])
            extra_inputs = tuple(numeric + categorical)
            nested_variants = nested_feature_variants(run, prediction_time, task, make_model)
            if nested_variants:
                extra_inputs = tuple(dict.fromkeys(column for _, num, cat in nested_variants.values()
                                                   for column in num + cat))
            if run.step == "8.4":
                extra_inputs += ("planned_stops",)
            if run.variant in REPRESENTATIONS or run.variant == "route_target":
                rep = route_representation(run.variant, task)
                make_model, numeric, categorical = rep.wrap(make_model, numeric, categorical)
                extra_inputs += rep.extra_inputs
            rows = (load_rows(prediction_time) if dummy_reference else
                    load_advanced_rows(prediction_time, weather=run.weather, inputs_dir=inputs_dir,
                                       extra_inputs=extra_inputs))
            if run.step == "8.4":
                events = pd.read_parquet(inputs_dir / "upstream_events_cv.parquet",
                                         filters=[("prediction_time", "=", prediction_time)])
                context = inference_context(rows, events)
                metrics, predictions = network_cv(
                    rows, context, inputs_dir / "network_training_cv.parquet", prediction_time,
                    task, make_model, numeric, categorical)
                numeric.append(NETWORK_FEATURE)
            elif run.step == "8.3":
                auxiliary = pd.read_parquet(inputs_dir / "corridor_training_cv.parquet",
                                            filters=[("prediction_time", "=", prediction_time)])
                metrics, predictions = evaluate_corridor_transfer(
                    rows.assign(corridor="frankfurt"), auxiliary,
                    prediction_time=prediction_time, task=task, method=run.variant,
                    numeric=numeric, categorical=categorical, params=params)
            elif run.step == "7.3":
                metrics, predictions, budget = run_tabgns(
                    run, rows, prediction_time, task, make_model, numeric, categorical, results_dir)
                budgets[prediction_time] = budget
                evaluation = "nested"
            elif run.step.startswith("5.") or run.step == "6.2":
                method, factories, details = ensemble_recipe(run, prediction_time, results_dir)
                metrics, predictions, fitted = cv_ensemble(
                    rows, prediction_time, task, factories, numeric, categorical, method)
                combinations[prediction_time] = {**details, "fitted": fitted}
                if nested_members:
                    evaluation = "nested"
            elif run.step == "3.2":
                metrics, predictions, perm, contributions = cv_importance(
                    rows, prediction_time, task, make_model, group_map)
                permutation.append(perm)
                shap.append(contributions)
            elif run.step == "3.3" and run.variant in ("forward", "backward"):
                metrics, predictions, selections = nested_groups(
                    rows, prediction_time, task, make_model, group_map, run.variant)
                group_choices.extend(selections)
                evaluation = "nested"
            elif run.nested:
                table, metrics, predictions = nested_search(
                    rows, prediction_time, task, make_model, configs, numeric, categorical,
                    variants=nested_variants)
                tables.append(table)
                chosen[prediction_time] = {}
                for choice in table.loc[table["chosen"]].itertuples():
                    decision = {"config_id": int(choice.config_id), "params": configs[choice.config_id]}
                    if nested_variants:
                        decision["feature_variant"] = choice.feature_variant
                    chosen[prediction_time][str(choice.outer_fold)] = decision
                print(f"{run.name} {prediction_time}: {SCORE[task]} "
                      f"{metrics[SCORE[task]].mean():.4f}, "
                      f"{table['cpu_seconds'].sum() / 60:.1f} CPU-min searching", flush=True)
            else:
                metrics, predictions = cross_validate(
                    rows, prediction_time, task, make_model, numeric, categorical)
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            features[prediction_time] = {"numeric": numeric, "categorical": categorical}
    if not all_metrics:
        return None
    config = {"name": run.name, "step": run.step, "task": task, "model": run.model,
              "target": TARGETS[task], "weather": run.weather, "features": features,
              "feature_set": feature_set, "variant": run.variant, "evaluation": evaluation,
              "git_commit": revision,
              "chosen": {pt: chosen[pt] for pt in features if pt in chosen}}
    if source:
        config["configuration_source"] = str((source / "config.json").relative_to(RESULTS))
    if run.nested:
        config["search"] = {"n_configs": len(configs), "selection": f"inner mean {SCORE[task]}"}
        if run.variant in AUTO_VARIANTS:
            count = min(FEATURE_SCREEN_CONFIGS, len(configs))
            config["search"]["feature_selection"] = f"best inner mean over the first {count} configurations"
        elif run.step == "0.8" and run.variant:
            config["evaluation"] = "nested tuning, exploratory variant"
    if group_choices:
        config["group_choices"] = group_choices
    if combinations:
        config["combination"] = combinations
    if run.variant == "l1":
        config["solver"] = L1_LOGISTIC
        if not all(metrics["converged"].all() for metrics in all_metrics):
            config["evaluation"] = "exploratory"
    if run.step in NEURAL:
        config["calendar"] = run.calendar or "native"
    if budgets:
        config["cpu_budgets"] = budgets
    out, _ = save_run(results_dir, run.name, config, pd.concat(all_metrics, ignore_index=True),
                      pd.concat(all_predictions, ignore_index=True))
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out / "search_results.csv", index=False)
    if permutation:
        pd.concat(permutation, ignore_index=True).to_csv(out / "permutation.csv", index=False)
        pd.concat(shap, ignore_index=True).to_csv(out / "shap.csv", index=False)
    if run.step == "6.2":
        from final_analysis import calibration_reliability
        from result_plots import calibration_plot
        adjusted = calibration_reliability(pd.concat(all_predictions, ignore_index=True))
        reference = latest_run(results_dir, tuning_reference(run))
        original = calibration_reliability(pd.read_parquet(reference / "oof_predictions.parquet"))
        adjusted.to_csv(out / "calibration.csv", index=False)
        calibration_plot({"uncalibrated": original, run.variant: adjusted}, out / "calibration.svg")
    print(out, flush=True)
    return out


def select_groups(run, groups):
    if run.step == "3.1":
        if run.variant == "timetable":
            return {"timetable"}
        if run.variant != "full":
            return dict(ablations(groups)).get(run.variant)
    if run.step == "3.4" and run.variant not in ("full", "weather_extremes"):
        railway = set(feature_groups("1h")) & groups.keys()
        return railway | {run.variant}
    return set(groups)


def run_tabgns(run, rows, prediction_time, task, factory, numeric, categorical, results_dir):
    from cv import fit_fold
    from folds import cv_folds

    reference = latest_run(results_dir, Run("7.1", NEURAL["7.1"][task == "severity"],
                                           run.weather, calendar=run.calendar))
    config = json.loads((reference / "config.json").read_text())
    if config["calendar"] != (run.calendar or "native"):
        raise ValueError("TabGNS and the MLP reference must use the same calendar encoding")
    search = pd.read_csv(reference / "search_results.csv")
    refit = pd.read_csv(reference / "fold_metrics.csv")
    budgets = (search.loc[search["prediction_time"].eq(prediction_time)]
               .groupby("outer_fold")["cpu_seconds"].sum()
               + refit.loc[refit["prediction_time"].eq(prediction_time)].set_index("fold")["fit_cpu_seconds"])
    results = []
    for fold, training, validation in cv_folds(rows):
        budget = float(budgets[fold])
        results.append(fit_fold(rows.loc[training], rows.loc[validation], prediction_time, fold, task,
                                partial(factory, cpu_budget=budget), numeric, categorical))
    metrics, predictions = zip(*results)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), budgets.to_dict()


def evaluate_pool(run, inputs_dir, results_dir, prediction_times, revision):
    if tuple(prediction_times) != tuple(PREDICTION_TIMES):
        raise ValueError("pooling requires all five prediction times")
    rows = pd.concat([load_advanced_rows(pt, weather=run.weather, inputs_dir=inputs_dir)
                      for pt in prediction_times], ignore_index=True)
    task = MODELS[run.model][0]
    numeric, categorical = feature_columns("1h", run.weather)
    categorical.append("prediction_time")
    # One set of parameters for the pooled model. Freeze the 1 h flat-search choice.
    factory = fixed_factory(run.model, run.weather, "1h")
    with threadpool_limits(limits=1):
        metrics, predictions = pooled_cv(rows, task, factory, numeric, categorical)
    config = {"step": run.step, "task": task, "model": run.model, "weather": run.weather,
              "git_commit": revision,
              "evaluation": "exploratory", "features": {"numeric": numeric, "categorical": categorical},
              "chosen": flat_choices(run.model, run.weather)["1h"],
              "configuration_source": str(flat_search_path(run.model, run.weather).relative_to(RESULTS))}
    out, _ = save_run(results_dir, run.name, config, metrics, predictions)
    print(out, flush=True)
    return out


def compare_ensemble_to_members(run, out, results_dir, task):
    """Compare an exploratory ensemble with its strongest single member at each horizon."""
    config = json.loads((out / "config.json").read_text())
    if run.step == "5.1":
        members = [results_dir / name for name in config["members"]]
    else:
        recipe = next(iter(config["combination"].values()))
        members = [latest_run(results_dir, Run("0.7", name, run.weather)) for name in recipe["members"]]
    score = f"{SCORE[task]}_mean"
    means = pd.concat({path: pd.read_csv(path / "summary.csv").set_index("prediction_time")[score]
                       for path in members}, axis=1)
    leaders = means.idxmin(axis=1)
    tables = []
    for reference in leaders.unique():
        differences = compare_runs(reference, out, task, subset_times=True)
        selected = differences["prediction_time"].map(leaders).eq(reference)
        table = differences.loc[selected].copy()
        table.insert(0, "reference", str(reference.relative_to(results_dir)))
        table["evaluation"] = "exploratory"
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(out / "paired_vs_base.csv", index=False)


def compare_to_base(run, out, results_dir, *, require_reference=False):
    standalone = ("4.2", "6.1", "6.3", "6.5", *NEURAL)
    if out is None or (not run.variant and run.step not in standalone):
        return
    if run.variant in ("route_native", "full", "timetable"):
        return
    task = MODELS[run.model][0]
    if run.step in ("5.1", "5.3", "5.4"):
        compare_ensemble_to_members(run, out, results_dir, task)
        return
    if run.step == "1.3":
        reference_run = Run("1.3", run.model, variant="route_native")
    elif run.step.startswith("3."):
        variant = "timetable" if run.variant.startswith("add_") else "full"
        reference_run = Run(run.step, run.model, run.weather, variant)
    elif run.step in ("7.2", "7.3", "7.4", "7.5"):
        reference_run = Run("7.1", NEURAL["7.1"][task == "severity"], run.weather,
                            calendar=run.calendar)
    elif run.step in standalone:
        reference_model = LIGHTGBM[task == "severity"]
        reference_run = (Run("0.7", reference_model, run.weather) if run.step == "4.2"
                         else Run("0.8", reference_model, run.weather, nested=True))
    elif run.variant == "top5_nested" or run.step == "6.2":
        reference_run = tuning_reference(run)
    elif run.nested:
        reference_run = Run(run.step, run.model, run.weather, nested=True)
    else:
        reference_run = Run("0.7", run.model, run.weather)
    try:
        reference = latest_run(results_dir, reference_run)
    except FileNotFoundError:
        if require_reference:
            raise
        print(f"{run.name}: comparison pending {reference_run.name}", flush=True)
        return
    comparison = compare_runs(reference, out, task, subset_times=True)
    comparison.insert(0, "reference", str(reference.relative_to(results_dir)))
    evaluations = {json.loads((path / "config.json").read_text())["evaluation"]
                   for path in (reference, out)}
    if not evaluations <= {"nested", "prespecified"}:
        comparison["evaluation"] = "exploratory"
    else:
        comparison["evaluation"] = "nested" if "nested" in evaluations else "prespecified"
    comparison.to_csv(out / "paired_vs_base.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("0.7", "0.8", *VARIANTS, "3.1", "3.2", "3.3", "3.4",
                                         *DECOMPOSED, "4.2", "5.1", "5.2", "5.3", "5.4",
                                         "6.1", "6.2", "6.3", "6.5", "6.6", *NEURAL, "8.2", "8.3", "8.4"))
    parser.add_argument("--inputs-dir", type=Path, default=ADVANCED)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--prediction-time", nargs="+", choices=PREDICTION_TIMES, default=PREDICTION_TIMES)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--n-configs", type=int,
                        help=f"configuration count for steps 0.8 and 6.1 (default {SEARCH_CONFIGS})")
    parser.add_argument("--model", nargs="+", choices=MODELS)
    parser.add_argument("--weather-arm", choices=("railway", "weather"), help="run one arm of a comparison")
    parser.add_argument("--variant", nargs="+", help="run only these registered variants")
    parser.add_argument("--compare-only", action="store_true", help="compare saved runs without refitting")
    parser.add_argument("--calendar", choices=VARIANTS["1.1"], help="Phase 1 calendar encoding for neural runs")
    parser.add_argument("--nested-variant", choices=(*AUTO_VARIANTS, *PHASE2_GROUPS, "all_groups", *REPRESENTATIONS,
                                                      "route_target"))
    args = parser.parse_args()
    if args.n_configs is not None and (args.n_configs < 1 or args.step not in ("0.8", "6.1")):
        parser.error("--n-configs must be positive and applies only to steps 0.8 and 6.1")
    if args.step in LATE_STEPS and not set(args.prediction_time) & CURRENT_DELAY_TIMES:
        parser.error(f"step {args.step} needs at least one of 1h, 20min or 0min")
    runs = list(registered_runs(args.step))
    if args.n_configs is not None:
        runs = [replace(run, n_configs=args.n_configs) for run in runs]
    if not args.variant:
        runs = [run for run in runs if run.variant != "plus_neural"]
    if args.model:
        runs = [run for run in runs if run.model in args.model]
    if args.weather_arm:
        runs = [run for run in runs if run.weather == (args.weather_arm == "weather")]
    if args.variant:
        runs = [run for run in runs if run.variant in args.variant]
    if args.calendar:
        if args.step not in NEURAL and not (args.step == "5.1" and args.variant == ["plus_neural"]):
            parser.error("--calendar is only for neural runs and their average")
        runs = [replace(run, calendar=args.calendar) for run in runs]
    if args.nested_variant:
        if args.step != "0.8":
            parser.error("--nested-variant is only for the nested booster search")
        runs = [replace(run, variant=args.nested_variant) for run in runs
                if run.weather or not args.nested_variant.startswith("weather_")]
    if not runs or args.jobs < 1:
        parser.error("choose at least one registered model and a positive job count")
    if args.compare_only:
        for run in runs:
            compare_to_base(run, latest_run(args.results_dir, run), args.results_dir,
                            require_reference=True)
        return
    evaluate = partial(evaluate_run, inputs_dir=args.inputs_dir, results_dir=args.results_dir,
                       prediction_times=tuple(dict.fromkeys(args.prediction_time)))
    if args.jobs == 1:
        outputs = list(map(evaluate, runs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            outputs = list(pool.map(evaluate, runs))
    for run, out in zip(runs, outputs):
        compare_to_base(run, out, args.results_dir)


if __name__ == "__main__":
    main()
