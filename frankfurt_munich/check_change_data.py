"""Trace each journey's targets back to the live-update snapshots.

The monthly processed data keep, for every stop, the values of the latest
change snapshot that carries any change (see create_monthly_data_release.py in
github.com/piebro/deutsche-bahn-data). An arrival time without a change falls
back to the planned time, so a 0-min delay can mean "on time" or "no update".
This script reads data/monthly_processed_data_change for the Frankfurt departure
and the München arrival of every journey and records:

- how many snapshots saw the München arrival and how many carried a changed
  arrival time;
- whether the arrival time in our data comes from a snapshot taken after the
  train arrived (an actual time) or before (a forecast), or from no change at
  all (the planned time);
- how long before the planned departure the cancellation was announced
  (`cancellation_announced`, computed in build_dataset.py from the same
  snapshots).

Journeys arriving after the last snapshot day are marked as not covered.
Writes data/change_check.parquet and change_check.json.

Run from the project root: .venv/bin/python frankfurt_munich/check_change_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from snapshots import event_summary, read_snapshots

HERE = Path(__file__).resolve().parent


def main() -> None:
    journeys = pd.concat([pd.read_parquet(HERE / "data" / f"journeys_{s}.parquet").assign(split=s)
                          for s in ("cv", "holdout")], ignore_index=True)
    stops = pd.read_parquet(HERE / "data" / "stops.parquet")
    origin = stops[stops.stop_index.eq(0)].set_index("journey_id").id
    destination = stops[stops.final_destination].set_index("journey_id").id
    j = journeys.set_index("journey_id")
    snaps = read_snapshots(set(origin) | set(destination))
    last_day = snaps.snapshot_timestamp.max().normalize()

    arr = event_summary(snaps, destination.reindex(j.index), "arrival").add_prefix("arr_")
    dep = event_summary(snaps, origin.reindex(j.index), "departure").add_prefix("dep_")
    out = j[["split", "origin_departure_planned_local", "destination_arrival_planned_local",
             "destination_arrival_reported_local", "journey_canceled", "arrival_delay_min",
             "cancellation_announced", "cancellation_withdrawn"]].join(arr).join(dep)
    covered = out.destination_arrival_planned_local.lt(last_day - pd.Timedelta(days=1))
    out["covered"] = covered

    # Where does the arrival time in our data come from?
    # Request time minus arrival time; for block labels the request may be up to
    # 45 minutes later than shown.
    lag = (out.arr_latest_request - out.arr_latest_time).dt.total_seconds() / 60
    slack = out.arr_latest_block_label.astype("boolean").fillna(False).astype(int) * 45
    source = pd.Series("not covered", index=out.index)
    source[covered & out.journey_canceled] = "cancelled"
    done = covered & ~out.journey_canceled
    source[done & out.arr_snapshots.isna()] = "no snapshot"
    source[done & out.arr_snapshots.notna() & out.arr_last_time.isna()] = "never a changed time"
    source[done & out.arr_last_time.notna() & out.arr_latest_time.isna()] = "changed time dropped"
    has = done & out.arr_latest_time.notna()
    source[has & lag.ge(0)] = "after arrival"
    source[has & lag.lt(0) & lag.add(slack).ge(0)] = "around arrival"
    source[has & lag.add(slack).lt(0)] = "before arrival"
    out["arrival_source"] = source
    out["arrival_snapshot_lag_min"] = lag

    # Does our data reproduce from the snapshots?
    rebuilt = out.arr_latest_time.fillna(out.destination_arrival_planned_local)
    out["arrival_reproduced"] = rebuilt.eq(out.destination_arrival_reported_local).where(done)

    # When was a cancellation announced, relative to the planned Frankfurt departure?
    out["cancellation_lead_min"] = ((out.origin_departure_planned_local - out.cancellation_announced)
                                    .dt.total_seconds() / 60)
    out.reset_index().to_parquet(HERE / "data" / "change_check.parquet", index=False)

    c = out[covered]
    month = c.origin_departure_planned_local.dt.to_period("M").astype(str)
    zero = c.arrival_delay_min.eq(0)
    canceled = c[c.journey_canceled & c.split.eq("cv")]  # outcome statistics for CV only
    lead = canceled.cancellation_lead_min
    # The source counts a cancellation whose latest status is "p" (withdrawn) as
    # not cancelled. Check that no cancelled journey has only withdrawn ends.
    still = {side: canceled[f"{side}_latest_status"].eq("c")
             | (canceled[f"{side}_latest_status"].isna() & canceled[f"{side}_latest_cancellation_time"].notna())
             for side in ("arr", "dep")}
    reinstated = canceled[~still["arr"] & ~still["dep"]]
    summary = {
        "snapshots_last_day": str(last_day.date()),
        "journeys_covered": int(len(c)), "by_split": c.split.value_counts().to_dict(),
        "arrival_source": c.arrival_source.value_counts().to_dict(),
        "arrival_source_by_month": pd.crosstab(month, c.arrival_source).to_dict(orient="index"),
        "arrival_source_of_0_min": c[zero].arrival_source.value_counts().to_dict(),
        "arrival_source_of_0_min_by_month": pd.crosstab(month[zero], c[zero].arrival_source).to_dict(orient="index"),
        "arrival_reproduced_share": round(float(c.arrival_reproduced.mean()), 4),
        "request_after_arrival_min": c.arrival_snapshot_lag_min[c.arrival_source.eq("after arrival")]
                                     .describe().round(1).to_dict(),
        "request_before_arrival_min": c.arrival_snapshot_lag_min[c.arrival_source.eq("before arrival")]
                                      .describe().round(1).to_dict(),
        "0_min_arrivals_with_an_earlier_different_time": {
            m: int(((c.arrival_delay_min.eq(0) & c.arr_earlier_time.notna()
                     & c.arr_earlier_time.ne(c.arr_last_time))[month.eq(m)]).sum()) for m in sorted(month.unique())},
        "cancelled_in_cv": {
            "journeys": int(len(canceled)),
            "with_cancellation_time": int(canceled.cancellation_announced.notna().sum()),
            "issued_before_planned_departure_by": {
                f"{h}_or_more": int(lead.ge(m).sum())
                for h, m in [("24h", 1440), ("6h", 360), ("1h", 60), ("20min", 20), ("0min", 0)]},
            "issued_after_planned_departure": int(lead.lt(0).sum()),
        },
        "cancellation_withdrawn": {
            "journeys_by_split": c[c.cancellation_withdrawn].split.value_counts().to_dict(),
            "cancelled_cv_journeys_withdrawn_at_both_ends": int(len(reinstated)),
        },
    }
    (HERE / "change_check.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    print(json.dumps(summary, indent=1, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
