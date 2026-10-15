"""Data-quality audit of the Frankfurt (Main) Hbf -> München Hbf dataset.

Reads the files written by build_dataset.py and writes audit.json. Every number
quoted in README.md comes from that file or from build_audit.json.

The hold-out set is not to be looked at yet, so outcome statistics (classes,
delays, cancellations) are computed on the CV set only. For the hold-out set
the audit reports counts, coverage and missing values. Changes over time compare
July 2024 to September 2025 with October 2025 to September 2026
(LATER_PERIOD_START in folds.py).

Run from the project root: .venv/bin/python frankfurt_munich/audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from folds import LATER_PERIOD_START, cv_folds

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HORIZONS = ["24h", "6h", "1h", "20min", "0min"]
DST_DATES = ["2024-10-27", "2025-03-30", "2025-10-26", "2026-03-29"]


def shares(series: pd.Series) -> dict:
    return {str(k): round(float(v), 4) for k, v in series.items()}


def main() -> None:
    cv = pd.read_parquet(DATA / "journeys_cv.parquet").assign(split="cv")
    holdout = pd.read_parquet(DATA / "journeys_holdout.parquet").assign(split="holdout")
    j = pd.concat([cv, holdout], ignore_index=True)
    stops = pd.read_parquet(DATA / "stops.parquet").sort_values(["journey_id", "stop_index"])
    planned = j.origin_departure_planned_local
    month = planned.dt.to_period("M")
    quarter = planned.dt.to_period("Q")
    is_cv = j.split.eq("cv")
    period = pd.Series(np.where(planned.ge(LATER_PERIOD_START), "from_2025_10", "before_2025_10"), index=j.index)
    delay = j.arrival_delay_min
    canceled = j.journey_canceled
    A: dict = {}

    # Coverage over time
    per_day = j.groupby(planned.dt.date).size().reindex(
        pd.date_range(planned.min().normalize(), planned.max().normalize()).date, fill_value=0)
    stops["delay"] = (stops.event_reported_time - stops.event_planned_time).dt.total_seconds() / 60
    same_as_plan = stops.event_reported_time.eq(stops.event_planned_time).groupby(stops.journey_id).all()
    no_update = j.journey_id.map(same_as_plan)
    monthly = pd.DataFrame({
        "journeys": j.groupby(month).size(),
        "canceled": canceled[is_cv].groupby(month).mean(),
        "severe_or_canceled": j.severe_or_canceled[is_cv].groupby(month).mean(),
        "arrival_exactly_0": delay[is_cv].eq(0).groupby(month).sum() / delay[is_cv].notna().groupby(month).sum(),
        "completed_without_live_updates": (no_update & ~canceled).groupby(month).sum(),
        "with_sequence_gaps": j.unobserved_sequence_positions.gt(0).groupby(month).mean(),
    })
    A["coverage"] = {
        "days": len(per_day), "days_without_journeys": int(per_day.eq(0).sum()),
        "per_day_median": float(per_day.median()),
        "days_below_12": {str(k): int(v) for k, v in per_day[per_day.lt(12)].items()},
        "monthly": {str(k): {c: round(float(v), 4) for c, v in row.items()} for k, row in monthly.iterrows()},
    }

    # Splits, folds and class balance
    A["splits"] = {
        "journeys": j.split.value_counts().to_dict(),
        "cv_class_shares": shares(cv.severity_class.value_counts(normalize=True)),
        "cv_class_shares_by_quarter": {str(q): shares(g.severity_class.value_counts(normalize=True))
                                       for q, g in j[is_cv].groupby(quarter[is_cv])},
        # Only how far the hold-out class shares are from CV, not the shares themselves
        "holdout_max_class_share_difference_from_cv": round(float(
            (holdout.severity_class.value_counts(normalize=True)
             - cv.severity_class.value_counts(normalize=True)).abs().max()), 4),
        "max_fold_class_share_difference_from_cv": round(max(float(
            (cv[cv.fold.eq(f)].severity_class.value_counts(normalize=True)
             - cv.severity_class.value_counts(normalize=True)).abs().max()) for f in range(1, 6)), 4),
        "holdout_days_by_month": {str(k): int(v) for k, v in holdout.origin_departure_planned_local.dt.normalize()
                                  .drop_duplicates().dt.to_period("M").value_counts().sort_index().items()},
        "holdout_journeys_with_train_number_unseen_in_cv":
            int((~holdout.train_number.isin(set(cv.train_number))).sum()),
        "ic_journeys_by_quarter": {str(q): int(g.train_type.eq("IC").sum()) for q, g in j.groupby(quarter)},
        "ic_journeys_by_period": j.train_type.eq("IC").groupby(period).sum().to_dict(),
    }
    A["folds"] = {fold: {"validation_journeys": int(val.sum()),
                         "train": shares(cv[train].severity_class.value_counts(normalize=True)),
                         "validation": shares(cv[val].severity_class.value_counts(normalize=True))}
                  for fold, train, val in cv_folds(cv)}
    family = np.select([j.recorded_route.str.contains("Würzburg"), j.recorded_route.str.contains("Stuttgart")],
                       ["wuerzburg", "stuttgart"], "other")
    A["route_family_by_quarter"] = {str(q): g.value_counts().to_dict()
                                    for q, g in pd.Series(family).groupby(quarter)}

    # Journeys kept from a departure listed under several train numbers
    shared = j[j.shared_departure.gt(1)]
    A["shared_departures_kept"] = {
        "journeys": int(len(shared)), "replacement_trains": int(shared.is_replacement_train.sum()),
        "by_split": shared.split.value_counts().to_dict(),
        "by_period": period[shared.index].value_counts().to_dict(),
    }

    # Delay values (CV only)
    full, j, delay, canceled, no_update = j, j[is_cv], delay[is_cv], canceled[is_cv], no_update[is_cv]
    period = period[is_cv]
    stops = stops[stops.journey_id.isin(cv.journey_id)]
    planned = j.origin_departure_planned_local
    A["arrival_delay"] = {
        "completed": int(delay.notna().sum()), "non_integer": int((delay.dropna() % 1).ne(0).sum()),
        "early_more_than_5": int(delay.lt(-5).sum()), "early_more_than_10": int(delay.lt(-10).sum()),
        "minimum": float(delay.min()), "over_180": int(delay.gt(180).sum()), "maximum": float(delay.max()),
        "exactly_0": int(delay.eq(0).sum()), "exactly_minus_1": int(delay.eq(-1).sum()),
        "exactly_1": int(delay.eq(1).sum()), "exactly_20": int(delay.eq(20).sum()),
        "exactly_60": int(delay.eq(60).sum()),
        "early_more_than_10_by_train": delay[delay.lt(-10)].groupby(j.train_number).size()
                                        .sort_values(ascending=False).head(6).to_dict(),
    }
    # Is a 0-min arrival a missing update? Compare the delay at the stop before München.
    before = stops[~stops.final_destination].groupby("journey_id").tail(1).set_index("journey_id").delay
    prev = j.journey_id.map(before)
    A["zero_delay_check"] = {
        p: {str(v): {"journeys": int((delay.eq(v) & in_p).sum()),
                     "median_delay_at_previous_stop": float(prev[delay.eq(v) & in_p].median()),
                     "share_previous_stop_10_plus": round(float(prev[delay.eq(v) & in_p].ge(10).mean()), 4)}
            for v in (-1, 0, 1)}
        for p, in_p in ((p, period.eq(p)) for p in ("before_2025_10", "from_2025_10"))}
    A["no_live_updates"] = {"journeys": int(no_update.sum()),
                            "canceled": int((no_update & canceled).sum()),
                            "completed": int((no_update & ~canceled).sum())}

    # Cancellations
    oc = j.origin_departure_canceled.astype(bool)
    dc = j.destination_arrival_canceled.astype(bool)
    served = stops[~stops.selected_event_canceled.fillna(False).astype(bool)]
    last_served = served.groupby("journey_id").station_name.last()
    all_canceled = stops.selected_event_canceled.fillna(False).astype(bool).groupby(stops.journey_id).all()
    A["cancellations"] = {
        "canceled": int(canceled.sum()), "origin_and_destination": int((oc & dc).sum()),
        "origin_only": int((oc & ~dc).sum()), "destination_only": int((~oc & dc).sum()),
        "every_stop_canceled": int(j.journey_id.map(all_canceled).sum()),
        "intermediate_only_not_counted": int((j.any_selected_stop_canceled.astype(bool) & ~canceled).sum()),
        "origin_only_median_arrival_delay_at_muenchen": float(
            ((j.destination_arrival_reported_local - j.destination_arrival_planned_local)
             .dt.total_seconds()[oc & ~dc] / 60).median()),
        "destination_only_last_served_station": j[~oc & dc].journey_id.map(last_served)
                                                  .value_counts().head(6).to_dict(),
        "cv_share": round(float(canceled.mean()), 4),
        "replacement_train_journeys": int(j.is_replacement_train.sum()),
    }

    # Time consistency
    planned_min = (j.destination_arrival_planned_local - planned).dt.total_seconds() / 60
    actual_min = (j.destination_arrival_reported_local - j.origin_departure_reported_local).dt.total_seconds() / 60
    step_planned = stops.groupby("journey_id").event_planned_time.diff().dt.total_seconds()
    step_reported = stops.groupby("journey_id").event_reported_time.diff().dt.total_seconds()
    dst = j[planned.dt.date.astype(str).isin(DST_DATES) & planned.dt.hour.le(3)]
    A["time_consistency"] = {
        "planned_travel_min": {"min": float(planned_min.min()), "median": float(planned_min.median()),
                               "max": float(planned_min.max()), "over_300": int(planned_min.gt(300).sum())},
        "actual_travel_below_70pct_of_planned": int((actual_min.lt(0.7 * planned_min) & ~canceled).sum()),
        "journeys_planned_times_go_backwards": int(stops.journey_id[step_planned.lt(0)].nunique()),
        "journeys_reported_times_go_backwards_over_1_min": int(stops.journey_id[step_reported.lt(-60)].nunique()),
        "night_journeys_on_dst_change_dates": int(len(dst)),
    }

    # current_delay: how often it is known in each split; values in CV only.
    # Why it is missing: the train starts at Frankfurt; its run has no usable
    # report at all; or nothing of the run has happened yet.
    runs = pd.read_parquet(DATA / "upstream_runs.parquet")
    usable = pd.concat([
        runs.journey_id[runs[f"{kind}_planned_time"].notna() & runs[f"{kind}_change_time"].notna()
                        & ~runs[f"{kind}_is_canceled"].fillna(False).astype(bool)]
        for kind in ("arrival", "departure")])
    no_report = ~j.journey_id.isin(set(usable)) & ~j.starts_at_origin
    A["current_delay_missing_in_cv"] = {}
    for h in HORIZONS:
        missing = j[f"current_delay_{h}"].isna()
        A["current_delay_missing_in_cv"][h] = {
            "missing": int(missing.sum()), "share": round(float(missing.mean()), 4),
            "starts_at_origin": int((missing & j.starts_at_origin).sum()),
            "no_usable_report": int((missing & no_report).sum()),
            "run_not_started": int((missing & ~j.starts_at_origin & ~no_report).sum()),
            "canceled_share_missing": None if not missing.any() else round(float(canceled[missing].mean()), 4),
            "canceled_share_known": None if missing.all() else round(float(canceled[~missing].mean()), 4)}
    A["current_delay"] = {
        h: {"known_share_by_split": shares(full[f"current_delay_{h}"].notna().groupby(full.split).mean()),
            "missing_in_cv": int(j[f"current_delay_{h}"].isna().sum()),
            "cv_median": None if j[f"current_delay_{h}"].isna().all() else float(j[f"current_delay_{h}"].median()),
            "cv_below_minus_10": int(j[f"current_delay_{h}"].lt(-10).sum()),
            "cv_below_minus_10_by_train": j.train_number[j[f"current_delay_{h}"].lt(-10)]
                                          .value_counts().head(3).to_dict()}
        for h in HORIZONS}
    A["current_delay"]["starts_at_origin_share"] = round(float(full.starts_at_origin.mean()), 4)

    # Prediction rows: journeys left at each prediction time once known cancellations are removed
    rows = pd.read_parquet(DATA / "prediction_rows_cv.parquet")
    holdout_rows = pd.read_parquet(DATA / "prediction_rows_holdout.parquet")
    A["prediction_rows"] = {
        "cv": rows.prediction_time.value_counts(sort=False).to_dict(),
        "holdout": holdout_rows.prediction_time.value_counts(sort=False).to_dict(),
        "cv_class_shares": {h: shares(g.severity_class.value_counts(normalize=True))
                            for h, g in rows.groupby("prediction_time", observed=True)},
        "cv_withdrawn_cancellations": int(j.cancellation_withdrawn.sum()),
    }

    (HERE / "audit.json").write_text(json.dumps(A, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in A.items() if k != "coverage"}, indent=1, ensure_ascii=False, default=str))
    print(pd.DataFrame(A["coverage"]["monthly"]).T.to_string())
    print("days below 12:", A["coverage"]["days_below_12"])


if __name__ == "__main__":
    main()
