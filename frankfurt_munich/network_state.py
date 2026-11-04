"""Build cutoff-safe network-state features from live change and exact plan snapshots.

Only CV rows are read by default. The feature is the share of unique ICE/IC
runs with an observed arrival/departure event in the trailing hour that was
delayed by at least 20 minutes or cancelled. A cancellation's issue time is its
event clock. Plan-only rows do not establish an on-time outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
DEFAULT_CHANGE_DIR = ROOT / "data" / "monthly_processed_data_change"
DEFAULT_PLAN_DIR = Path("/private/tmp/plan_archive_d27b9c")
PANEL_PATH = ROOT / "data" / "stations_present_in_all_months.csv"
ORIGIN = "Frankfurt (Main) Hbf"
DESTINATION = "München Hbf"
HORIZON = pd.Timedelta(hours=1)

sys.path.insert(0, str(HERE))
from history_features import (  # noqa: E402
    NETWORK_STATE_COLUMNS,
    PLAN_EXACT_UTC_FROM,
    ROW_KEY,
    cutoff_instants,
    latest_snapshot_asof,
    local_clock_to_utc,
    request_upper_bound_utc,
)

CHANGE_READ_COLUMNS = [
    "id", "station_name", "snapshot_timestamp", "train_type", "train_number",
    "arrival_change_time", "arrival_cancellation_time", "arrival_change_status",
    "departure_change_time", "departure_cancellation_time", "departure_change_status",
]
PLAN_READ_COLUMNS = [
    "id", "station_name", "snapshot_timestamp", "train_type", "train_number",
    "arrival_planned_time", "departure_planned_time",
]
SCOPE_FEATURE = {
    "network": NETWORK_STATE_COLUMNS[0],
    "network_frankfurt": NETWORK_STATE_COLUMNS[1],
    "network_munich": NETWORK_STATE_COLUMNS[2],
}


def _month(path: Path) -> pd.Period:
    return pd.Period(path.stem.removeprefix("data-"), freq="M")


def _run_month(ids: pd.Series) -> pd.Series:
    token = ids.astype("string").str.rsplit("-", n=2).str[-2]
    starts = pd.to_datetime(token, format="%y%m%d%H%M", errors="coerce")
    return starts.dt.to_period("M")


def _read_month(path: Path, *, kind: str, stations: list[str]) -> pd.DataFrame:
    columns = CHANGE_READ_COLUMNS if kind == "change" else PLAN_READ_COLUMNS
    table = pq.read_table(
        path,
        columns=columns,
        filters=[("station_name", "in", stations), ("train_type", "in", ["ICE", "IC"])],
        use_threads=False,
    )
    frame = table.to_pandas()
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(["id", "snapshot_timestamp"], keep="last").copy()
    frame["_run_month"] = _run_month(frame["id"])
    return frame


def _timeline_states(changes: pd.DataFrame, plans: pd.DataFrame) -> pd.DataFrame:
    """Join each stop update to the latest change and exact plan state."""
    if changes.empty and plans.empty:
        return pd.DataFrame()
    changes = changes.copy()
    plans = plans.copy()
    if not changes.empty:
        changes["snapshot_timestamp"] = pd.to_datetime(changes["snapshot_timestamp"], utc=True)
        changes["available_at_utc"] = request_upper_bound_utc(changes["snapshot_timestamp"])
    if not plans.empty:
        plans["snapshot_timestamp"] = pd.to_datetime(plans["snapshot_timestamp"], utc=True)
        plans = plans.loc[plans["snapshot_timestamp"].ge(PLAN_EXACT_UTC_FROM)].copy()
        plans["available_at_utc"] = plans["snapshot_timestamp"]
    updates = pd.concat([
        changes[["id", "available_at_utc"]] if not changes.empty else pd.DataFrame(columns=["id", "available_at_utc"]),
        plans[["id", "available_at_utc"]] if not plans.empty else pd.DataFrame(columns=["id", "available_at_utc"]),
    ], ignore_index=True).drop_duplicates().sort_values(["available_at_utc", "id"]).reset_index(drop=True)
    if updates.empty:
        return pd.DataFrame()
    updates["_query_id"] = np.arange(len(updates), dtype=np.int64)
    updates["next_update_utc"] = updates.groupby("id", sort=False)["available_at_utc"].shift(-1)
    queries = updates[["_query_id", "id"]].copy()
    queries["cutoff_utc"] = updates["available_at_utc"]

    change_values = [
        "station_name", "train_type", "train_number", "arrival_change_time",
        "arrival_cancellation_time", "arrival_change_status", "departure_change_time",
        "departure_cancellation_time", "departure_change_status",
    ]
    plan_values = [
        "station_name", "train_type", "train_number", "arrival_planned_time",
        "departure_planned_time",
    ]
    if changes.empty:
        change_state = queries[["_query_id"]].copy()
        for column in change_values:
            change_state[column] = pd.NaT if column.endswith("time") else pd.NA
    else:
        change_state = latest_snapshot_asof(queries, changes, request_bound=True)
        change_state = change_state[["_query_id", *change_values]]
    if plans.empty:
        plan_state = queries[["_query_id"]].copy()
        for column in plan_values:
            plan_state[column] = pd.NaT if column.endswith("time") else pd.NA
    else:
        plan_state = latest_snapshot_asof(queries, plans, request_bound=False)
        plan_state = plan_state[["_query_id", *plan_values]]
    change_state = change_state.rename(columns={column: f"change_{column}" for column in change_values if column != "train_number"})
    plan_state = plan_state.rename(columns={column: f"plan_{column}" for column in plan_values if column != "train_number"})
    states = updates.merge(change_state, on="_query_id", validate="one_to_one")
    states = states.merge(plan_state, on="_query_id", validate="one_to_one")
    states["station_name_asof"] = states["plan_station_name"].combine_first(states["change_station_name"])
    states["train_type_asof"] = states["plan_train_type"].combine_first(states["change_train_type"])
    return states


def _event_intervals(states: pd.DataFrame, stations: set[str]) -> pd.DataFrame:
    """Build half-open active/disrupted intervals for observed stop events."""
    if states.empty:
        return pd.DataFrame(columns=["run_id", "scope", "start_utc", "end_utc", "disrupted"])
    keep_types = states["train_type_asof"].astype("string").isin(["ICE", "IC"])
    keep_stations = states["station_name_asof"].astype("string").isin(stations)
    scoped = states.loc[keep_types & keep_stations].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["run_id", "scope", "start_utc", "end_utc", "disrupted"])
    scoped["run_id"] = scoped["id"].astype("string").str.rsplit("-", n=1).str[0]
    receipt = pd.to_datetime(scoped["available_at_utc"], utc=True)
    next_update = pd.to_datetime(scoped["next_update_utc"], utc=True)
    parts = []
    for event, report_name, plan_name, cancellation_name, status_name in (
        ("arrival", "change_arrival_change_time", "plan_arrival_planned_time",
         "change_arrival_cancellation_time", "change_arrival_change_status"),
        ("departure", "change_departure_change_time", "plan_departure_planned_time",
         "change_departure_cancellation_time", "change_departure_change_status"),
    ):
        report_time = local_clock_to_utc(scoped[report_name], ambiguous=False)
        plan_time = local_clock_to_utc(scoped[plan_name], ambiguous=False)
        issue_time = local_clock_to_utc(scoped[cancellation_name], ambiguous=False)
        withdrawn = scoped[status_name].astype("string").str.lower().eq("p").fillna(False)
        cancelled = issue_time.notna() & ~withdrawn
        delay_min = (report_time - plan_time).dt.total_seconds() / 60
        state_known = cancelled | (report_time.notna() & plan_time.notna())
        event_time = report_time.where(~cancelled, issue_time)
        start = receipt.where(receipt.ge(event_time), event_time)
        event_end = event_time + HORIZON
        end = event_end.where(next_update.isna() | event_end.le(next_update), next_update)
        valid = state_known & event_time.notna() & start.lt(end)
        if not valid.any():
            continue
        base = scoped.loc[valid, ["id", "run_id", "station_name_asof"]].copy()
        base["start_utc"] = start.loc[valid]
        base["end_utc"] = end.loc[valid]
        base["disrupted"] = (cancelled | delay_min.ge(20).fillna(False)).loc[valid].to_numpy()
        base = base.rename(columns={"station_name_asof": "station_name"})
        network = base.loc[base.station_name.isin(stations)].copy()
        network["scope"] = "network"
        parts.append(network)
        frankfurt = base.loc[base.station_name.eq(ORIGIN)].copy()
        frankfurt["scope"] = "network_frankfurt"
        parts.append(frankfurt)
        munich = base.loc[base.station_name.eq(DESTINATION)].copy()
        munich["scope"] = "network_munich"
        parts.append(munich)
    if not parts:
        return pd.DataFrame(columns=["run_id", "scope", "start_utc", "end_utc", "disrupted"])
    return pd.concat(parts, ignore_index=True)[["run_id", "scope", "start_utc", "end_utc", "disrupted"]]


def _run_scope_segments(intervals: pd.DataFrame) -> pd.DataFrame:
    """Union overlapping stop-event intervals so each run counts once per scope."""
    columns = ["scope", "start_utc", "end_utc", "disrupted"]
    if intervals.empty:
        return pd.DataFrame(columns=columns)
    starts = intervals[["run_id", "scope", "start_utc", "disrupted"]].rename(
        columns={"start_utc": "time"}).copy()
    starts["active_delta"] = 1
    starts["disrupted_delta"] = starts["disrupted"].astype(np.int8)
    ends = intervals[["run_id", "scope", "end_utc", "disrupted"]].rename(
        columns={"end_utc": "time"}).copy()
    ends["active_delta"] = -1
    ends["disrupted_delta"] = -ends["disrupted"].astype(np.int8)
    points = pd.concat([
        starts[["run_id", "scope", "time", "active_delta", "disrupted_delta"]],
        ends[["run_id", "scope", "time", "active_delta", "disrupted_delta"]],
    ], ignore_index=True)
    points = points.groupby(["run_id", "scope", "time"], as_index=False, sort=False)[
        ["active_delta", "disrupted_delta"]].sum()
    points = points.sort_values(["scope", "run_id", "time"]).reset_index(drop=True)
    grouped = points.groupby(["scope", "run_id"], sort=False)
    points["active"] = grouped["active_delta"].cumsum()
    points["disrupted_active"] = grouped["disrupted_delta"].cumsum()
    points["end_utc"] = grouped["time"].shift(-1)
    segments = points.loc[points.active.gt(0) & points.end_utc.gt(points.time),
                          ["scope", "time", "end_utc", "disrupted_active"]].copy()
    segments["disrupted"] = segments["disrupted_active"].gt(0)
    return segments.rename(columns={"time": "start_utc"})[
        ["scope", "start_utc", "end_utc", "disrupted"]]


def _scope_deltas(segment_parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not segment_parts:
        return pd.DataFrame(columns=["scope", "time", "active_delta", "disrupted_delta"])
    segments = pd.concat(segment_parts, ignore_index=True)
    if segments.empty:
        return pd.DataFrame(columns=["scope", "time", "active_delta", "disrupted_delta"])
    starts = segments[["scope", "start_utc", "disrupted"]].rename(
        columns={"start_utc": "time"}).copy()
    starts["active_delta"] = 1
    starts["disrupted_delta"] = starts["disrupted"].astype(np.int8)
    ends = segments[["scope", "end_utc", "disrupted"]].rename(
        columns={"end_utc": "time"}).copy()
    ends["active_delta"] = -1
    ends["disrupted_delta"] = -ends["disrupted"].astype(np.int8)
    points = pd.concat([
        starts[["scope", "time", "active_delta", "disrupted_delta"]],
        ends[["scope", "time", "active_delta", "disrupted_delta"]],
    ], ignore_index=True)
    return points.groupby(["scope", "time"], as_index=False, sort=False)[
        ["active_delta", "disrupted_delta"]].sum().sort_values(["scope", "time"])


def _shares_at_cutoffs(rows: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    result = rows[ROW_KEY].copy().reset_index(drop=True)
    cutoffs = cutoff_instants(result["predicted_at"])
    cutoff_ns = cutoffs.dt.as_unit("ns").astype("int64").to_numpy()
    for scope, feature in SCOPE_FEATURE.items():
        events = deltas.loc[deltas.scope.eq(scope)].sort_values("time") if not deltas.empty else deltas
        values = np.full(len(result), np.nan, dtype=float)
        if not events.empty:
            times = pd.to_datetime(events["time"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
            active = events["active_delta"].to_numpy().cumsum()
            disrupted = events["disrupted_delta"].to_numpy().cumsum()
            positions = np.searchsorted(times, cutoff_ns, side="right") - 1
            known = positions >= 0
            denominators = np.zeros(len(result), dtype=np.int64)
            numerators = np.zeros(len(result), dtype=np.int64)
            denominators[known] = active[positions[known]]
            numerators[known] = disrupted[positions[known]]
            usable = known & (denominators > 0)
            values[usable] = numerators[usable] / denominators[usable]
        result[feature] = values
    return result


def _month_files(directory: Path) -> dict[pd.Period, Path]:
    return {_month(path): path for path in directory.glob("data-????-??.parquet")}


def build_network_features(
    rows: pd.DataFrame,
    *,
    change_dir: Path = DEFAULT_CHANGE_DIR,
    plan_dir: Path = DEFAULT_PLAN_DIR,
    stations: list[str] | None = None,
    source_months: list[pd.Period] | None = None,
    split: str = "cv",
) -> tuple[pd.DataFrame, dict]:
    """Build cutoff-safe shares, processing source archives one month at a time."""
    missing = set(ROW_KEY) - set(rows.columns)
    if missing:
        raise ValueError(f"prediction rows are missing keys: {sorted(missing)}")
    if rows.duplicated(ROW_KEY).any():
        raise ValueError("prediction rows must be unique by ROW_KEY")
    if "split" in rows and not rows["split"].eq(split).all():
        raise PermissionError(f"network features are restricted to {split} prediction rows")
    panel = set(stations if stations is not None else pd.read_csv(PANEL_PATH).station_name.astype(str))
    panel.update([ORIGIN, DESTINATION])
    change_files, plan_files = _month_files(change_dir), _month_files(plan_dir)
    if not change_files:
        raise FileNotFoundError(f"no monthly change snapshots in {change_dir}")
    available_months = sorted(set(change_files) | set(plan_files))
    if source_months is not None:
        wanted = set(source_months)
        available_months = [month for month in available_months if month in wanted]
        missing_months = wanted - set(available_months)
        if missing_months:
            raise FileNotFoundError(f"missing source months: {sorted(map(str, missing_months))}")

    pending: dict[pd.Period, dict[str, list[pd.DataFrame]]] = defaultdict(lambda: {"change": [], "plan": []})
    segment_parts: list[pd.DataFrame] = []
    stats = {"change_rows_read": 0, "exact_plan_rows_read": 0,
             "unparseable_run_ids": 0, "run_months_processed": 0,
             "event_intervals": 0, "run_scope_segments": 0}

    def process(run_month: pd.Period) -> None:
        bucket = pending.pop(run_month, {"change": [], "plan": []})
        changes = pd.concat(bucket["change"], ignore_index=True) if bucket["change"] else pd.DataFrame(columns=CHANGE_READ_COLUMNS + ["_run_month"])
        plans = pd.concat(bucket["plan"], ignore_index=True) if bucket["plan"] else pd.DataFrame(columns=PLAN_READ_COLUMNS + ["_run_month"])
        changes = changes.drop(columns="_run_month", errors="ignore")
        plans = plans.drop(columns="_run_month", errors="ignore")
        if not changes.empty:
            changes = changes.drop_duplicates(["id", "snapshot_timestamp"], keep="last")
        if not plans.empty:
            plans = plans.drop_duplicates(["id", "snapshot_timestamp"], keep="last")
        states = _timeline_states(changes, plans)
        intervals = _event_intervals(states, panel)
        segments = _run_scope_segments(intervals)
        stats["run_months_processed"] += 1
        stats["event_intervals"] += int(len(intervals))
        stats["run_scope_segments"] += int(len(segments))
        if not segments.empty:
            segment_parts.append(segments)

    for month in available_months:
        if month in change_files:
            changes = _read_month(change_files[month], kind="change", stations=sorted(panel))
            if not changes.empty:
                stats["change_rows_read"] += len(changes)
                stats["unparseable_run_ids"] += int(changes["_run_month"].isna().sum())
                for run_month, frame in changes.dropna(subset=["_run_month"]).groupby("_run_month", sort=False):
                    pending[run_month]["change"].append(frame)
        if month in plan_files and month >= pd.Period("2025-11", freq="M"):
            plans = _read_month(plan_files[month], kind="plan", stations=sorted(panel))
            if not plans.empty:
                plans["snapshot_timestamp"] = pd.to_datetime(plans["snapshot_timestamp"], utc=True)
                plans = plans.loc[plans["snapshot_timestamp"].ge(PLAN_EXACT_UTC_FROM)].copy()
                stats["exact_plan_rows_read"] += len(plans)
                for run_month, frame in plans.dropna(subset=["_run_month"]).groupby("_run_month", sort=False):
                    pending[run_month]["plan"].append(frame)
        # A service's snapshots can be in the month before or after its run month.
        for ready in sorted(run_month for run_month in pending if run_month <= month - 1):
            process(ready)
    for run_month in sorted(pending):
        process(run_month)

    deltas = _scope_deltas(segment_parts)
    output = _shares_at_cutoffs(rows, deltas)
    if not output[ROW_KEY].reset_index(drop=True).equals(rows[ROW_KEY].reset_index(drop=True)):
        raise ValueError("network feature keys do not exactly match prediction rows")
    stats["prediction_rows"] = int(len(output))
    stats["share_known"] = {feature: int(output[feature].notna().sum()) for feature in NETWORK_STATE_COLUMNS}
    stats["source_policy"] = {
        "stable_panel_path": str(PANEL_PATH),
        "stable_panel_station_count": len(panel),
        "plan_exact_from_utc": str(PLAN_EXACT_UTC_FROM),
        "legacy_plan_snapshots_used": False,
        "change_snapshot_bound": "legacy whole-hour labels receive +45 minutes; exact request timestamps are unchanged",
        "denominator": "unique ICE/IC runs with a visible arrival/departure report and exact plan, or an in-force cancellation, in the trailing hour",
        "cancellation_event_clock": "cancellation issue time",
        "interval_policy": "activate at max(snapshot request time, event time); expire at the next change/plan update or event time + 1 hour",
        "scopes": {
            "network": "stations_present_in_all_months.csv, including both endpoints",
            "network_frankfurt": ORIGIN,
            "network_munich": DESTINATION,
        },
    }
    return output, stats


def _load_prediction_rows(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"prediction_rows_{split}.parquet", columns=ROW_KEY)


def _sample_rows(rows: pd.DataFrame, *, data_dir: Path, split: str, month: str) -> pd.DataFrame:
    period = pd.Period(month, freq="M")
    journeys = pd.read_parquet(data_dir / f"journeys_{split}.parquet",
                               columns=["journey_id", "origin_departure_planned_local"])
    selected = set(journeys.loc[
        pd.to_datetime(journeys.origin_departure_planned_local).dt.to_period("M").eq(period),
        "journey_id",
    ])
    return rows.loc[rows.journey_id.isin(selected)].reset_index(drop=True)


def _write_output(output: Path, frame: pd.DataFrame, audit: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    audit_path = output.with_name(output.stem + "_audit.json")
    audit_tmp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    with audit_tmp.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False, default=str)
    audit_tmp.replace(audit_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("cv", "holdout"), default="cv")
    parser.add_argument("--change-dir", type=Path, default=DEFAULT_CHANGE_DIR)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-month", help="build only rows for one planned-departure month")
    args = parser.parse_args()
    if args.split == "holdout":
        from holdout_gate import REGISTRY, committed_registration
        committed_registration(REGISTRY)

    rows = _load_prediction_rows(args.data_dir, args.split)
    source_months = None
    if args.sample_month:
        rows = _sample_rows(rows, data_dir=args.data_dir, split=args.split, month=args.sample_month)
        target_month = pd.Period(args.sample_month, freq="M")
        source_months = [target_month - 1, target_month, target_month + 1]
    output, audit = build_network_features(
        rows, change_dir=args.change_dir, plan_dir=args.plan_dir,
        source_months=source_months, split=args.split,
    )
    path = args.output or DATA / "advanced" / f"network_{args.split}.parquet"
    audit.update({"split": args.split, "source_months": list(map(str, source_months)) if source_months else "all"})
    _write_output(path, output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
