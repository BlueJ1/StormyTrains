"""Phase 8.4 training pairs and nested, cross-fitted network delay predictions."""
from __future__ import annotations

import argparse
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cv import fit_fold
from data import DATA, ROW_KEY, TIME_COLUMN
from folds import cv_folds, inner_folds, read_assignment, week_of
from history_features import (CHANGE_DIR, PLAN_DIR, _cancellation_in_force, latest_plan_asof,
                              latest_snapshot_asof, local_clock_to_utc, prepare_plan_snapshots,
                              request_upper_bound_utc)

NETWORK_FEATURE = "network_delay_prediction"
CONTEXT = ["last_stop_delay", "remaining_planned_min", "remaining_planned_stops"]
WEEKS = ["run_week", "source_week", "target_week", "source_report_week",
         "target_report_week", "cutoff_week"]
PAIR_COLUMNS = ["id", "target_id", *CONTEXT, *WEEKS, "delay_target"]
SOURCE = DATA.parents[1] / "data" / "monthly_processed_data"
MONTH_COLUMNS = ["id", "arrival_planned_time", "arrival_change_time",
                 "arrival_is_canceled", "is_additional_stop", "has_plan_record"]
CHANGE_COLUMNS = ["id", "snapshot_timestamp", "arrival_change_time", "arrival_change_status",
                  "arrival_cancellation_time"]


def scheduled_pairs(stops, allowed_weeks):
    """Choose one later stop per eligible source stop across all eligible runs."""
    stops = stops.loc[~stops["is_additional_stop"].fillna(False)
                      & stops["has_plan_record"].fillna(False)
                      & stops["arrival_planned_time"].notna()].copy()
    if stops.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    keys = stops["id"].str.rsplit("-", n=2, expand=True)
    stops["journey_id"] = keys[0] + "-" + keys[1]
    stops["sequence"] = pd.to_numeric(keys[2], errors="coerce")
    run_start = pd.to_datetime(keys[1], format="%y%m%d%H%M", errors="coerce")
    stops["run_week"] = week_of(run_start)
    stops["stop_week"] = week_of(stops["arrival_planned_time"])
    stops = stops.loc[stops["run_week"].isin(allowed_weeks) & stops["stop_week"].isin(allowed_weeks)
                      & stops["sequence"].notna()]
    stops = stops.sort_values(["journey_id", "sequence"]).reset_index(drop=True)
    grouped = stops.groupby("journey_id", sort=False)
    remaining = (grouped["id"].transform("size") - 1 - grouped.cumcount()).to_numpy()
    offsets = 1 + (pd.util.hash_pandas_object(stops["id"], index=False).to_numpy() % 8)
    lead = np.minimum(offsets.astype(int), remaining)
    starts = stops.loc[lead > 0].reset_index(drop=True)
    ends = stops.iloc[np.flatnonzero(lead > 0) + lead[lead > 0]].reset_index(drop=True)
    labels = ((ends["arrival_change_time"] - ends["arrival_planned_time"]).dt.total_seconds() / 60
              ).where(~ends["arrival_is_canceled"].fillna(False))
    report_week = week_of(ends["arrival_change_time"])
    pairs = pd.DataFrame({"id": starts["id"], "target_id": ends["id"],
                          "run_week": starts["run_week"], "source_week": starts["stop_week"],
                          "target_week": ends["stop_week"], "target_report_week": report_week,
                          "target_reported": ends["arrival_change_time"], "delay_target": labels,
                          "remaining_planned_stops": ends["sequence"] - starts["sequence"]})
    return pairs.loc[labels.notna() & report_week.isin(allowed_weeks)]


def archive_rows(directory, month, ids, columns):
    files = [directory / f"data-{period}.parquet" for period in (month - 1, month, month + 1)]
    files = [path for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError(f"no nearby archive files for {month} in {directory}")
    return pq.read_table(files, columns=columns, filters=[("id", "in", set(ids))],
                         use_threads=False).to_pandas()


def usable_arrival(changes, cutoff):
    status = changes["arrival_change_status"]
    cancelled = _cancellation_in_force(changes["arrival_cancellation_time"], status, cutoff)
    arrived = local_clock_to_utc(changes["arrival_change_time"], ambiguous=False).le(cutoff)
    return arrived & ~cancelled & ~status.eq("c").fillna(False)


def paired_month(path, plan_dir, change_dir, allowed_weeks):
    allowed_weeks = tuple(allowed_weeks)
    raw = pq.read_table(path, columns=MONTH_COLUMNS,
                        filters=[("train_type", "in", ["ICE", "IC"])], use_threads=False).to_pandas()
    pairs = scheduled_pairs(raw, allowed_weeks)
    if pairs.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    month = pd.Period(path.stem.removeprefix("data-"), freq="M")
    changes = archive_rows(change_dir, month, pairs["id"], CHANGE_COLUMNS)
    changes["cutoff_utc"] = request_upper_bound_utc(changes["snapshot_timestamp"])
    usable = usable_arrival(changes, changes["cutoff_utc"])
    first_report = changes.loc[usable].sort_values("cutoff_utc").drop_duplicates("id")
    pairs = pairs.merge(first_report[["id", "cutoff_utc"]], on="id", validate="one_to_one")
    pairs = pairs.loc[pairs["cutoff_utc"].lt(local_clock_to_utc(pairs["target_reported"], ambiguous=False))]
    if pairs.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    plans = archive_rows(plan_dir, month, pd.concat([pairs["id"], pairs["target_id"]]),
                         ["id", "snapshot_timestamp", "arrival_planned_time"])
    plans = plans[plans["arrival_planned_time"].notna()].copy()
    plans = prepare_plan_snapshots(plans)
    first_plan = plans.groupby("id")["available_at_utc"].min()
    seen = pairs["target_id"].map(first_plan)
    # Start the training forecast once both the upstream report and target
    # timetable have been observed, while the target arrival is still ahead.
    pairs["cutoff_utc"] = pairs["cutoff_utc"].where(pairs["cutoff_utc"].ge(seen), seen)
    target_utc = local_clock_to_utc(pairs["target_reported"], ambiguous=False)
    pairs = pairs.loc[pairs["cutoff_utc"].lt(target_utc)].copy()
    if pairs.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    cutoff = pairs["cutoff_utc"].reset_index(drop=True)
    scheduled = []
    for key in ("id", "target_id"):
        queries = pd.DataFrame({"id": pairs[key].to_numpy(), "cutoff_utc": cutoff,
                                "_query_id": range(len(pairs))})
        asof = latest_plan_asof(queries, plans)
        if "arrival_planned_time" not in asof:
            return pd.DataFrame(columns=PAIR_COLUMNS)
        scheduled.append(asof["arrival_planned_time"].set_axis(pairs.index))
    prior, later = scheduled
    reports = latest_snapshot_asof(
        pairs[["id", "cutoff_utc"]].assign(_query_id=range(len(pairs))),
        changes[CHANGE_COLUMNS]).set_axis(pairs.index)
    report_time = reports["arrival_change_time"]
    reported = usable_arrival(reports, pairs["cutoff_utc"])
    pairs["source_report_week"] = week_of(report_time)
    pairs["cutoff_week"] = week_of(pairs["cutoff_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None))
    pairs["last_stop_delay"] = (report_time - prior).dt.total_seconds() / 60
    minutes = local_clock_to_utc(later, ambiguous=False) - local_clock_to_utc(prior, ambiguous=False)
    pairs["remaining_planned_min"] = minutes.dt.total_seconds() / 60
    eligible = (reported & pairs[WEEKS].isin(allowed_weeks).all(axis=1)
                & pairs["remaining_planned_min"].gt(0))
    return pairs.loc[eligible, PAIR_COLUMNS].dropna()


def inference_context(rows, events):
    event = events.loc[events["event_report_visible"] & events["event_occurred_by_cutoff"]
                       & events["event_kind"].eq("arrival") & ~events["event_cancelled"]
                       & events["event_delay_min"].notna()].copy()
    event = (event.sort_values(["reported_time_utc", "station_sequence", "event_kind"])
             .drop_duplicates(ROW_KEY, keep="last"))
    frankfurt = events.loc[events["station_name"].eq("Frankfurt (Main) Hbf"),
                           [*ROW_KEY, "station_sequence"]].drop_duplicates(ROW_KEY)
    frankfurt = frankfurt.rename(columns={"station_sequence": "frankfurt_sequence"})
    context = event[[*ROW_KEY, "planned_time_utc", "station_sequence", "event_delay_min"]].merge(
        frankfurt, how="left", on=ROW_KEY, validate="one_to_one")
    joined = rows.merge(context, on=ROW_KEY, how="left", validate="one_to_one")
    return pd.DataFrame({
        "last_stop_delay": joined["event_delay_min"],
        "remaining_planned_min": ((local_clock_to_utc(joined["destination_arrival_planned_local"],
                                                     ambiguous=False) - joined["planned_time_utc"])
                                  .dt.total_seconds() / 60),
        "remaining_planned_stops": (joined["frankfurt_sequence"] - joined["station_sequence"]
                                    + joined["planned_stops"] - 1),
    }, index=rows.index)


@cache
def training_pairs(path):
    pairs = pd.read_parquet(path)
    assignment = read_assignment()
    cv_weeks = assignment.loc[assignment["split"].eq("cv"), "week_start"].tolist()
    if not pairs[WEEKS].isin(cv_weeks).all().all():
        raise ValueError("network pretraining pairs include a non-CV week")
    pairs["journey_id"] = pairs["id"].str.rsplit("-", n=1).str[0]
    return pairs


@lru_cache(maxsize=48)
def fit_network(path, weeks, excluded_runs):
    from lightgbm import LGBMRegressor

    pairs = training_pairs(path)
    valid = pairs[WEEKS].isin(weeks).all(axis=1) & ~pairs["journey_id"].isin(excluded_runs)
    train = pairs.loc[valid]
    model = LGBMRegressor(n_estimators=150, num_leaves=15, n_jobs=1, random_state=0, verbose=-1)
    return model.fit(train[CONTEXT], train["delay_target"])


def predict_network(model, context):
    valid = context["last_stop_delay"].notna() & context["remaining_planned_min"].gt(0)
    result = pd.Series(np.nan, index=context.index)
    if valid.any():
        result.loc[valid] = model.predict(context.loc[valid])
    return result


def network_cv(rows, context, pairs_path, prediction_time, task, factory, numeric, categorical):
    metrics, predictions = [], []
    for fold, training, validation in cv_folds(rows):
        train, valid = rows.loc[training].copy(), rows.loc[validation].copy()
        outer_validation_runs = set(valid.journey_id)
        train[NETWORK_FEATURE] = np.nan
        for _, inner_train, inner_valid in inner_folds(train):
            weeks = week_of(train.loc[inner_train, TIME_COLUMN]).unique()
            excluded = outer_validation_runs | set(train.loc[inner_valid, "journey_id"])
            model = fit_network(pairs_path, tuple(sorted(weeks)), tuple(sorted(excluded)))
            index = train.index[inner_valid]
            train.loc[index, NETWORK_FEATURE] = predict_network(model, context.loc[index])
        model = fit_network(pairs_path, tuple(sorted(week_of(train[TIME_COLUMN]).unique())),
                            tuple(sorted(outer_validation_runs)))
        valid[NETWORK_FEATURE] = predict_network(model, context.loc[valid.index])
        scores, out = fit_fold(train, valid, prediction_time, fold, task, factory,
                               [*numeric, NETWORK_FEATURE], categorical)
        metrics.append(scores)
        predictions.append(out)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=PLAN_DIR)
    parser.add_argument("--change-dir", type=Path, default=CHANGE_DIR)
    parser.add_argument("--month", nargs="+", help="restrict extraction for a local data check")
    parser.add_argument("--output", type=Path, default=DATA / "advanced" / "network_training_cv.parquet")
    args = parser.parse_args()
    assignment = read_assignment()
    weeks = assignment.loc[assignment["split"].eq("cv"), "week_start"]
    output = []
    for path in sorted(SOURCE.glob("data-????-??.parquet")):
        month = path.stem.removeprefix("data-")
        if args.month and month not in args.month:
            continue
        frame = paired_month(path, args.plan_dir, args.change_dir, weeks)
        print(path.name, len(frame), flush=True)
        if not frame.empty:
            output.append(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not output:
        raise ValueError("no cutoff-safe training pairs were found")
    pd.concat(output, ignore_index=True).drop_duplicates(["id", "target_id"]).to_parquet(
        args.output, index=False, compression="zstd")


if __name__ == "__main__":
    main()
