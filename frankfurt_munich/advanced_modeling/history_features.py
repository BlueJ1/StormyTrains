"""Cutoff-safe feature primitives backed by parsed-plan and live-change snapshots.

The public builders deliberately accept normalized snapshot DataFrames.  This keeps
the time logic testable without scanning the full network archive, and lets the
caller decide when a large source extraction is appropriate.

All timestamps in the source archives are UTC snapshot timestamps and German local
wall-clock train times.  Legacy change snapshots are stamped at the start of a
six-hour fetch block; their request time is therefore bounded above by 45 minutes
after that stamp.  The bound is used so a feature never includes a snapshot that
might have arrived after the prediction cutoff.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]  # frankfurt_munich
ROOT = HERE.parent
CHANGE_DIR = ROOT / "data" / "monthly_processed_data_change"
PLAN_DIR = ROOT / "data" / "monthly_processed_data_plan"
DATA_DIR = HERE / "data"

ROW_KEY = ["journey_id", "prediction_time", "predicted_at"]
PREDICTION_TIMES = ["24h", "6h", "1h", "20min", "0min"]
RECENT_RELIABILITY_COLUMNS = [
    "service_severe_or_cancelled_rate_7d", "service_mean_delay_min_7d",
    "service_severe_or_cancelled_rate_28d", "service_mean_delay_min_28d",
    "corridor_severe_or_cancelled_rate_24h", "corridor_mean_delay_min_24h",
    "corridor_severe_or_cancelled_rate_3h", "corridor_mean_delay_min_3h",
]
DELAY_DYNAMICS_COLUMNS = [
    "current_delay", "current_delay_change_min", "upstream_stops_reported",
    "minutes_since_last_report", "planned_minutes_remaining_before_frankfurt",
]
NETWORK_STATE_COLUMNS = [
    "network_delayed_or_cancelled_share_1h",
    "network_frankfurt_delayed_or_cancelled_share_1h",
    "network_munich_delayed_or_cancelled_share_1h",
]
FEATURE_COLUMNS_BY_GROUP = {
    "recent_reliability": RECENT_RELIABILITY_COLUMNS,
    "delay_dynamics": DELAY_DYNAMICS_COLUMNS,
    "network_state": NETWORK_STATE_COLUMNS,
}
SNAPSHOT_EXACT_UTC_FROM = pd.Timestamp("2025-11-03 03:27:54.653106", tz="UTC")
PLAN_EXACT_UTC_FROM = pd.Timestamp("2025-11-03 03:32:22.619260", tz="UTC")
LEGACY_REQUEST_UPPER_BOUND = pd.Timedelta(minutes=45)
STATIC_PLAN_AVAILABLE_UTC = pd.Timestamp("1900-01-01", tz="UTC")
LEGACY_PLAN_AVAILABLE_UTC = STATIC_PLAN_AVAILABLE_UTC  # earlier name, kept for imports

CHANGE_COLUMNS = [
    "id", "station_name", "snapshot_timestamp", "train_type", "train_number",
    "train_label_type", "arrival_change_time", "arrival_cancellation_time",
    "arrival_change_status", "departure_change_time", "departure_cancellation_time",
    "departure_change_status", "replaced_train_type", "replaced_train_number",
]
PLAN_COLUMNS = [
    "id", "snapshot_timestamp", "station_name", "train_type", "train_number",
    "arrival_planned_time", "departure_planned_time", "arrival_planned_path",
    "departure_planned_path",
]


def request_upper_bound_utc(snapshot_timestamp: pd.Series) -> pd.Series:
    """Latest possible request time for each archive snapshot stamp.

    Before the source switched to exact request timestamps, stamps are six-hour
    block starts.  Those receive a conservative +45 minute bound.  Exact request
    timestamps are kept as-is.  A whole-hour timestamp after the transition is
    treated as exact only if it is at or after the observed first exact timestamp.
    """
    stamps = pd.to_datetime(snapshot_timestamp, utc=True)
    block_start = (stamps.dt.minute.eq(0) & stamps.dt.second.eq(0)
                   & stamps.dt.microsecond.eq(0))
    legacy = stamps.lt(SNAPSHOT_EXACT_UTC_FROM) & block_start
    return stamps + legacy.astype("int64") * LEGACY_REQUEST_UPPER_BOUND


def plan_availability_utc(snapshot_timestamp: pd.Series, *,
                          legacy_hour_label_offset: pd.Timedelta | None = None) -> pd.Series:
    """Treat every plan snapshot as a static schedule.

    A timetable is published long before the day of travel; the snapshot stamp
    says when the archive fetched it, not when it was known. The group decided
    this for the legacy plans on 5 November 2026 and for the plans with exact
    request stamps (from 3 November 2025) on 12 November 2026.

    The optional legacy offset reproduces the earlier sensitivity calculation:
    legacy plans are bounded by their hour label plus the offset, and exact
    request stamps are kept as they are.
    """
    stamps = pd.to_datetime(snapshot_timestamp, utc=True).dt.as_unit("ns")
    if legacy_hour_label_offset is None:
        static = pd.Series(STATIC_PLAN_AVAILABLE_UTC, index=stamps.index).dt.as_unit("ns")
        return static.where(stamps.notna())
    legacy = stamps.lt(PLAN_EXACT_UTC_FROM)
    return stamps.where(~legacy, stamps + legacy_hour_label_offset)


def prepare_plan_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest plan snapshot per stop as its static schedule."""
    plans = snapshots.copy()
    if plans.empty:
        return plans.assign(available_at_utc=pd.Series(dtype="datetime64[ns, UTC]"))
    plans["snapshot_timestamp"] = pd.to_datetime(plans["snapshot_timestamp"], utc=True)
    plans["available_at_utc"] = plan_availability_utc(plans["snapshot_timestamp"])
    return (plans.sort_values("snapshot_timestamp", kind="stable")
            .drop_duplicates(["id", "available_at_utc"], keep="first"))


def latest_plan_asof(queries: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    plans = snapshots if "available_at_utc" in snapshots else prepare_plan_snapshots(snapshots)
    return latest_snapshot_asof(queries, plans, observed_col="available_at_utc", request_bound=False)


def local_clock_to_utc(values: pd.Series, *, ambiguous: bool) -> pd.Series:
    """Convert naive German train-clock values to UTC conservatively.

    For prediction cutoffs, `ambiguous=True` selects the earlier repeated autumn
    hour.  For reported and planned event times, `ambiguous=False` selects the
    later occurrence, so an event with no recorded UTC offset is never counted
    too early.  Spring gaps are shifted backward for cutoffs and forward for
    events, again taking the conservative side of the boundary.
    """
    values = pd.to_datetime(values)
    if values.dt.tz is not None:
        return values.dt.tz_convert("UTC")
    nonexistent = "shift_backward" if ambiguous else "shift_forward"
    return values.dt.tz_localize("Europe/Berlin", ambiguous=ambiguous,
                                 nonexistent=nonexistent).dt.tz_convert("UTC")


def cutoff_instants(predicted_at: pd.Series) -> pd.Series:
    """Return conservative UTC instants for the local `predicted_at` values."""
    return local_clock_to_utc(predicted_at, ambiguous=True)


def read_snapshots_for_ids(ids: Iterable[str], *, kind: str,
                           change_dir: Path = CHANGE_DIR,
                           plan_dir: Path = PLAN_DIR) -> pd.DataFrame:
    """Read only requested stop IDs from a snapshot archive.

    `kind` is `change` or `plan`.  Each monthly file is projected and filtered
    once with Arrow's row-group predicate pushdown and one worker; only matching
    rows are retained in memory.  Callers should pass only IDs needed by the
    current feature batch.
    """
    if kind == "change":
        directory, columns = change_dir, CHANGE_COLUMNS
    elif kind == "plan":
        directory, columns = plan_dir, PLAN_COLUMNS
    else:
        raise ValueError("kind must be 'change' or 'plan'")
    wanted = sorted({str(value) for value in ids if pd.notna(value)})
    if not wanted:
        return pd.DataFrame(columns=columns)
    files = sorted(directory.glob("data-????-??.parquet"))
    if not files:
        raise FileNotFoundError(f"no monthly {kind} snapshots in {directory}")
    parts = []
    for path in files:
        table = pq.read_table(path, columns=columns, use_threads=False,
                              filters=[("id", "in", wanted)])
        if table.num_rows:
            parts.append(table.to_pandas())
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["id", "snapshot_timestamp"], keep="last")
    return out.sort_values(["id", "snapshot_timestamp"]).reset_index(drop=True)


def latest_snapshot_asof(queries: pd.DataFrame, snapshots: pd.DataFrame, *,
                          id_col: str = "id", cutoff_col: str = "cutoff_utc",
                          observed_col: str = "snapshot_timestamp",
                          request_bound: bool = True,
                          query_id_col: str = "_query_id") -> pd.DataFrame:
    """Attach the latest snapshot whose conservative request bound is at cutoff.

    `queries` must have a unique query ID, a stop ID, and a UTC cutoff.  The
    result has one row per query, including queries with no eligible snapshot.
    """
    required_queries = {query_id_col, id_col, cutoff_col}
    if not required_queries.issubset(queries.columns):
        raise ValueError(f"queries must contain {sorted(required_queries)}")
    if not queries[query_id_col].is_unique:
        raise ValueError(f"{query_id_col} must be unique")
    if snapshots.empty:
        return queries.copy()
    required_snapshots = {id_col, observed_col}
    if not required_snapshots.issubset(snapshots.columns):
        raise ValueError(f"snapshots must contain {sorted(required_snapshots)}")
    right = snapshots.copy()
    if request_bound:
        right["_observed_utc"] = request_upper_bound_utc(right[observed_col])
    else:
        right["_observed_utc"] = pd.to_datetime(right[observed_col], utc=True)
    left = queries.copy()
    left[cutoff_col] = pd.to_datetime(left[cutoff_col], utc=True).dt.as_unit("ns")
    right["_observed_utc"] = right["_observed_utc"].dt.as_unit("ns")
    left[id_col] = left[id_col].astype(object)
    right[id_col] = right[id_col].astype(object)
    left = left.sort_values([cutoff_col, id_col])
    right = right.sort_values(["_observed_utc", id_col])
    merged = pd.merge_asof(
        left, right, left_on=cutoff_col, right_on="_observed_utc", by=id_col,
        direction="backward", allow_exact_matches=True,
        suffixes=("", "_snapshot"),
    )
    return merged.sort_values(query_id_col).reset_index(drop=True)


def _cancellation_in_force(issue_time: pd.Series, status: pd.Series,
                           cutoff_utc: pd.Series) -> pd.Series:
    issue_utc = local_clock_to_utc(issue_time, ambiguous=False)
    has_issue = issue_utc.notna() & issue_utc.le(cutoff_utc)
    withdrawn = status.astype("string").str.lower().eq("p").fillna(False)
    return has_issue & ~withdrawn


def build_upstream_events_asof(rows: pd.DataFrame, upstream_stops: pd.DataFrame,
                               change_snapshots: pd.DataFrame,
                               plan_snapshots: pd.DataFrame, *,
                               plan_legacy_policy: str = "static") -> pd.DataFrame:
    """Return per-target upstream stop events visible from snapshots at cutoff.

    The output has one row per prediction key, stop ID, and event kind.  It
    includes only stops with a plan or change snapshot available by the cutoff.
    `event_report_visible` marks an update actually present in the change source;
    `event_occurred_by_cutoff` further requires its reported event time to have
    passed.  This lets sequence models distinguish a visible forecast from an
    event that has already occurred.
    """
    missing = set(ROW_KEY) - set(rows.columns)
    if missing:
        raise ValueError(f"rows are missing keys: {sorted(missing)}")
    if "split" in rows and not rows["split"].eq("cv").all():
        raise PermissionError("history features are restricted to CV prediction rows")
    needed_stops = {"journey_id", "id", "train_line_station_num", "station_name"}
    if not needed_stops.issubset(upstream_stops.columns):
        raise ValueError(f"upstream_stops must contain {sorted(needed_stops)}")
    if rows.duplicated(ROW_KEY).any():
        raise ValueError("prediction rows must be unique by ROW_KEY")
    if rows.empty:
        return pd.DataFrame(columns=ROW_KEY)

    target = rows[ROW_KEY].reset_index(drop=True).copy()
    target["_row_id"] = np.arange(len(target), dtype=np.int64)
    target["cutoff_utc"] = cutoff_instants(target["predicted_at"])
    # Keep IDs only; the monthly file's final event values are never read.
    stop_columns = ["journey_id", "id", "train_line_station_num", "station_name"]
    stops = upstream_stops[stop_columns].drop_duplicates(["journey_id", "id"]).copy()
    expanded = target.merge(stops, on="journey_id", how="left", validate="many_to_many")
    expanded = expanded[expanded["id"].notna()].copy()
    if expanded.empty:
        return pd.DataFrame(columns=ROW_KEY + [
            "event_id", "station_sequence", "station_name", "event_kind",
            "planned_time", "reported_time", "event_delay_min", "event_cancelled",
            "event_report_visible", "event_occurred_by_cutoff",
        ])
    expanded["_query_id"] = np.arange(len(expanded), dtype=np.int64)
    query = expanded[["_query_id", "id", "cutoff_utc"]]
    changes = latest_snapshot_asof(query, change_snapshots, request_bound=True)
    if plan_legacy_policy == "static":
        plans = latest_plan_asof(query, plan_snapshots)
    elif plan_legacy_policy == "exclude":
        plan_source = plan_snapshots.copy()
        if not plan_source.empty:
            stamps = pd.to_datetime(plan_source["snapshot_timestamp"], utc=True)
            plan_source = plan_source.loc[stamps.ge(PLAN_EXACT_UTC_FROM)].copy()
        plan_request_bound = False
    elif plan_legacy_policy == "bounded_sensitivity":
        # This mirrors the former +45m assumption for legacy plan files.  The
        # source provenance does not validate that bound; keep it sensitivity-only.
        plan_source = plan_snapshots
        plan_request_bound = True
    else:
        raise ValueError("unknown legacy plan policy")
    if plan_legacy_policy != "static":
        plans = latest_snapshot_asof(query, plan_source, request_bound=plan_request_bound)
    change_cols = [c for c in (
        "_query_id", "id", "station_name", "train_type", "train_number",
        "snapshot_timestamp", "_observed_utc", "arrival_change_time",
        "arrival_cancellation_time", "arrival_change_status", "departure_change_time",
        "departure_cancellation_time", "departure_change_status",
    ) if c in changes]
    plan_cols = [c for c in (
        "_query_id", "id", "station_name", "train_type", "train_number",
        "snapshot_timestamp", "_observed_utc", "arrival_planned_time",
        "departure_planned_time",
    ) if c in plans]
    changes = changes[change_cols].rename(columns={
        "station_name": "_change_station", "train_type": "_change_train_type",
        "train_number": "_change_train_number", "snapshot_timestamp": "_change_snapshot",
        "_observed_utc": "change_observed_at_utc",
    })
    plans = plans[plan_cols].rename(columns={
        "station_name": "_plan_station", "train_type": "_plan_train_type",
        "train_number": "_plan_train_number", "snapshot_timestamp": "_plan_snapshot",
        "_observed_utc": "plan_observed_at_utc",
        "arrival_planned_time": "_arrival_plan_asof",
        "departure_planned_time": "_departure_plan_asof",
    })
    state = expanded.merge(changes, on=["_query_id", "id"], how="left",
                           validate="one_to_one")
    state = state.merge(plans, on=["_query_id", "id"], how="left",
                        validate="one_to_one")
    for column in ("arrival_change_time", "arrival_cancellation_time",
                   "arrival_change_status", "departure_change_time",
                   "departure_cancellation_time", "departure_change_status",
                   "_arrival_plan_asof", "_departure_plan_asof"):
        if column not in state:
            state[column] = pd.NaT
    for column in ("change_observed_at_utc", "plan_observed_at_utc"):
        if column not in state:
            state[column] = pd.NaT
    for column in ("_plan_station", "_change_station", "_plan_train_type",
                   "_change_train_type", "_plan_train_number", "_change_train_number"):
        if column not in state:
            state[column] = pd.NA
    state["station_name_asof"] = state["_plan_station"].combine_first(state["_change_station"])
    state["train_type_asof"] = state["_plan_train_type"].combine_first(
        state["_change_train_type"])
    state["train_number_asof"] = state["_plan_train_number"].combine_first(
        state["_change_train_number"])
    state["station_sequence"] = pd.to_numeric(
        state["id"].astype("string").str.rsplit("-", n=1).str[-1], errors="coerce")

    parts = []
    for kind, kind_name, report_col, issue_col, status_col, plan_col in (
        (0, "arrival", "arrival_change_time", "arrival_cancellation_time",
         "arrival_change_status", "_arrival_plan_asof"),
        (1, "departure", "departure_change_time", "departure_cancellation_time",
         "departure_change_status", "_departure_plan_asof"),
    ):
        part_columns = list(dict.fromkeys([
            *ROW_KEY, "_row_id", "cutoff_utc", "id", "station_sequence",
            "station_name_asof", "train_type_asof", "train_number_asof",
            "train_line_station_num", "change_observed_at_utc", "plan_observed_at_utc",
            report_col, issue_col, status_col, plan_col, "_departure_plan_asof",
        ]))
        part = state[part_columns].copy()
        part["stop_departure_planned_time"] = part["_departure_plan_asof"]
        part = part.rename(columns={
            "id": "event_id", "station_name_asof": "station_name",
            "train_type_asof": "train_type", "train_number_asof": "train_number",
            "train_line_station_num": "recorded_station_sequence",
            report_col: "reported_time", issue_col: "cancellation_issue_time",
            status_col: "change_status", plan_col: "planned_time",
        })
        part["event_kind"] = kind_name
        part["_event_kind_order"] = kind
        part["reported_time_utc"] = local_clock_to_utc(part["reported_time"], ambiguous=False)
        part["planned_time_utc"] = local_clock_to_utc(part["planned_time"], ambiguous=False)
        part["event_report_visible"] = (
            part["reported_time"].notna() & part["change_observed_at_utc"].notna())
        part["event_occurred_by_cutoff"] = (
            part["event_report_visible"]
            & part["reported_time_utc"].le(part["cutoff_utc"]))
        part["event_cancelled"] = _cancellation_in_force(
            part["cancellation_issue_time"], part["change_status"], part["cutoff_utc"])
        part["event_delay_min"] = (
            part["reported_time"] - part["planned_time"]
        ).dt.total_seconds() / 60
        part["event_visible_asof"] = (
            part["change_observed_at_utc"].notna() | part["plan_observed_at_utc"].notna())
        parts.append(part)
    events = pd.concat(parts, ignore_index=True)
    return events[events["event_visible_asof"]].reset_index(drop=True)


def delay_dynamics_from_events(rows: pd.DataFrame,
                               upstream_events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Phase 2.4 values from `build_upstream_events_asof` output."""
    if rows.duplicated(ROW_KEY).any():
        raise ValueError("prediction rows must be unique by ROW_KEY")
    target = rows[ROW_KEY].reset_index(drop=True).copy()
    target["_row_id"] = np.arange(len(target), dtype=np.int64)
    target["cutoff_utc"] = cutoff_instants(target["predicted_at"])
    if upstream_events.empty:
        return _empty_dynamics(target)
    events = upstream_events.copy()
    if "_event_kind_order" not in events:
        events["_event_kind_order"] = events["event_kind"].map({"arrival": 0, "departure": 1})
    events = events.merge(target[ROW_KEY + ["_row_id", "cutoff_utc"]],
                          on=ROW_KEY, how="inner", suffixes=("", "_target"),
                          validate="many_to_one")
    # The departure from Frankfurt itself is not used (decision of 13 November
    # 2026, as for the final-record current_delay): at 0 min it would say
    # whether the train left on time.
    frankfurt_departure = (events["station_name"].eq("Frankfurt (Main) Hbf")
                           & events["event_kind"].eq("departure"))
    occurred = events[
        events["event_occurred_by_cutoff"] & ~events["event_cancelled"] & ~frankfurt_departure
    ].copy()
    if occurred.empty:
        return _empty_dynamics(target)
    report_agg = occurred.groupby("_row_id", sort=False).agg(
        upstream_stops_reported=("event_id", "nunique"),
        _last_report_receipt_utc=("change_observed_at_utc", "max"),
    )
    report_agg["minutes_since_last_report"] = (
        target.set_index("_row_id")["cutoff_utc"].reindex(report_agg.index)
        - report_agg["_last_report_receipt_utc"]
    ).dt.total_seconds() / 60

    delay_events = occurred[occurred["event_delay_min"].notna()].sort_values(
        ["_row_id", "reported_time_utc", "station_sequence", "_event_kind_order"])
    latest = delay_events.groupby("_row_id", sort=False).tail(1).set_index("_row_id")
    current_delay = latest["event_delay_min"].copy()
    waiting = (
        latest["cutoff_utc"] - local_clock_to_utc(
            latest["stop_departure_planned_time"], ambiguous=False)
    ).dt.total_seconds() / 60
    current_delay = current_delay.where(
        latest["event_kind"].ne("arrival"),
        np.fmax(current_delay, waiting),
    )

    origin = events[
        events["station_name"].eq("Frankfurt (Main) Hbf")
        & events["event_kind"].eq("departure")
        & events["plan_observed_at_utc"].notna()
    ].drop_duplicates("_row_id").set_index("_row_id")
    origin_plan_utc = local_clock_to_utc(origin["planned_time"], ambiguous=False)
    remaining = (origin_plan_utc.reindex(latest.index)
                 - latest["planned_time_utc"]).dt.total_seconds() / 60

    features = target.set_index("_row_id")[[*ROW_KEY]].copy()
    features["current_delay"] = current_delay
    features["planned_minutes_remaining_before_frankfurt"] = remaining
    features = features.join(report_agg[["upstream_stops_reported", "minutes_since_last_report"]])
    time_order = {name: i for i, name in enumerate(PREDICTION_TIMES)}
    ordered = features.reset_index().assign(
        _time_order=lambda frame: frame.prediction_time.map(time_order)
    ).sort_values(["journey_id", "_time_order"])
    ordered["current_delay_change_min"] = (
        ordered.groupby("journey_id", sort=False)["current_delay"].diff())
    features["current_delay_change_min"] = ordered.set_index("_row_id")[
        "current_delay_change_min"]
    return features.reset_index(drop=True)


def _use_scheduled_run(rows: pd.DataFrame, announcements: pd.DataFrame,
                       context: pd.DataFrame) -> pd.Series:
    """Rows whose history comes from the scheduled train rather than the operated one.

    A replacement uses the train it replaced until the replacement is announced.
    When the monthly files record no replaced train, no other scheduled train is
    known, so the row uses the operated run like an ordinary train.
    """
    aligned = rows[ROW_KEY].merge(announcements, on=ROW_KEY, how="left",
                                  validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("replacement announcement table does not cover prediction keys")
    recorded = aligned["journey_id"].map(
        context.set_index("journey_id")["scheduled_train_number"]).notna()
    announced_at = pd.to_datetime(aligned["replacement_announced_at_utc"], utc=True)
    known = (aligned["replacement_announced"].fillna(False)
             & announced_at.le(cutoff_instants(aligned["predicted_at"])))
    return aligned["is_replacement"].fillna(False) & recorded & ~known


def _scheduled_upstream_stops(context: pd.DataFrame) -> pd.DataFrame:
    """Enumerate stop IDs through the original service's Frankfurt origin."""
    records = []
    for journey_id, origin_id in context[["journey_id", "scheduled_origin_id"]].dropna().itertuples(index=False):
        ride, _, sequence = origin_id.rpartition("-")
        if not ride or not sequence.isdecimal():
            continue
        records.extend((journey_id, f"{ride}-{i}", i)
                       for i in range(1, int(sequence) + 1))
    return pd.DataFrame(records, columns=["journey_id", "id", "train_line_station_num"]).assign(
        station_name=pd.NA
    )


def _journey_events(operated_rows, upstream, scheduled_rows, scheduled_stops, changes, plans):
    operated = build_upstream_events_asof(operated_rows, upstream, changes, plans)
    scheduled = build_upstream_events_asof(scheduled_rows, scheduled_stops, changes, plans)
    if not scheduled.empty:
        origins = scheduled_stops.groupby("journey_id", sort=False).tail(1)
        origin_keys = origins[["journey_id", "id"]].rename(columns={"id": "event_id"})
        visible = scheduled.loc[scheduled["plan_observed_at_utc"].notna()].merge(
            origin_keys, on=["journey_id", "event_id"], validate="many_to_one"
        )
        scheduled = scheduled.merge(visible[ROW_KEY].drop_duplicates(), on=ROW_KEY,
                                    validate="many_to_one")
    parts = [events for events in (operated, scheduled) if not events.empty]
    if not parts:
        return operated
    return pd.concat(parts, ignore_index=True).drop(columns="_row_id")


def build_delay_dynamics(rows: pd.DataFrame, upstream_stops: pd.DataFrame,
                         change_snapshots: pd.DataFrame,
                         plan_snapshots: pd.DataFrame, *,
                         plan_legacy_policy: str = "static") -> pd.DataFrame:
    """Build cutoff-safe Phase 2.4 dynamics from as-of plan/change snapshots."""
    events = build_upstream_events_asof(
        rows, upstream_stops, change_snapshots, plan_snapshots,
        plan_legacy_policy=plan_legacy_policy,
    )
    return delay_dynamics_from_events(rows, events)


def _empty_dynamics(target: pd.DataFrame) -> pd.DataFrame:
    out = target[["_row_id", *ROW_KEY]].copy().set_index("_row_id")
    for column in ("current_delay", "current_delay_change_min",
                   "upstream_stops_reported", "minutes_since_last_report",
                   "planned_minutes_remaining_before_frankfurt"):
        out[column] = np.nan
    return out.reset_index(drop=True)


def _read_snapshot_ids_in_months(ids: Iterable[str], *, kind: str,
                                 months: Iterable[pd.Period],
                                 change_dir: Path,
                                 plan_dir: Path) -> pd.DataFrame:
    """Read a bounded set of IDs from the monthly files covering a time window."""
    if kind == "change":
        directory, columns = change_dir, CHANGE_COLUMNS
    elif kind == "plan":
        directory, columns = plan_dir, PLAN_COLUMNS
    else:
        raise ValueError("kind must be 'change' or 'plan'")
    wanted = sorted({str(value) for value in ids if pd.notna(value)})
    if not wanted:
        return pd.DataFrame(columns=columns)
    paths = [directory / f"data-{month}.parquet" for month in sorted(set(months))]
    parts = []
    for path in paths:
        if not path.is_file():
            continue
        table = pq.read_table(path, columns=columns, filters=[("id", "in", wanted)],
                              use_threads=False)
        if table.num_rows:
            parts.append(table.to_pandas())
    if not parts:
        return pd.DataFrame(columns=columns)
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(["id", "snapshot_timestamp"], keep="last")
            .sort_values(["id", "snapshot_timestamp"]).reset_index(drop=True))


def _journey_run_start_utc(journey_ids: pd.Series) -> pd.Series:
    """Parse the source journey ID's YYMMDDHHMM run-start clock."""
    suffix = journey_ids.astype("string").str.rsplit("-", n=1).str[-1]
    local = pd.to_datetime(suffix, format="%y%m%d%H%M", errors="coerce")
    return local_clock_to_utc(local, ambiguous=False)


def build_history_dynamics_split(
    split: str = "cv",
    *, plan_dir: Path = PLAN_DIR,
    change_dir: Path = CHANGE_DIR,
    data_dir: Path = DATA_DIR,
    output_dir: Path | None = None,
    event_prediction_times: Iterable[str] = ("1h", "20min", "0min"),
    require_full_archive: bool = True,
    registry_path: Path | None = None,
) -> tuple[Path, Path]:
    """Build split-specific dynamics after replacement evidence is available.

    The holdout gate runs before any holdout path is read. Inputs for a holdout
    rebuild are projected to prediction keys and stop identifiers only.
    """
    if split not in {"cv", "holdout"}:
        raise ValueError("split must be 'cv' or 'holdout'")
    gate_passed = False
    if split == "holdout":
        from holdout_gate import REGISTRY, committed_registration
        committed_registration(registry_path or REGISTRY)
        gate_passed = True
    output_dir = output_dir or data_dir / "advanced"
    announcements_path = output_dir / f"replacement_announcements_{split}.parquet"
    if not announcements_path.is_file():
        raise FileNotFoundError(
            "run the split-specific timetable builder first; replacement announcement "
            f"cache is required at {announcements_path}"
        )
    announcements = pd.read_parquet(announcements_path)
    if require_full_archive:
        expected = {f"data-{month}.parquet" for month in
                    pd.period_range("2024-07", "2026-09", freq="M").astype(str)}
        for label, directory in (("plan", plan_dir), ("change", change_dir)):
            actual = {p.name for p in directory.glob("data-????-??.parquet")}
            missing = sorted(expected - actual)
            if missing:
                raise FileNotFoundError(
                    f"incomplete {label} archive at {directory}; missing {missing[:4]}"
                )

    rows_path = data_dir / f"prediction_rows_{split}.parquet"
    rows = pd.read_parquet(rows_path, columns=ROW_KEY)
    target_journeys = set(rows["journey_id"].unique())
    # The shared upstream file covers both splits. Project stop IDs and
    # metadata only, then select the requested cohort before reading snapshots.
    upstream = pd.read_parquet(data_dir / "upstream_runs.parquet", columns=[
        "journey_id", "id", "train_line_station_num", "station_name",
    ])
    upstream = upstream[upstream["journey_id"].isin(target_journeys)].copy()
    context = pd.read_parquet(output_dir / f"replacement_context_{split}.parquet",
                              columns=["journey_id", "scheduled_origin_id", "scheduled_train_number"])
    use_scheduled = _use_scheduled_run(rows, announcements, context)
    scheduled_rows = rows.loc[use_scheduled.to_numpy()]
    scheduled_stops = _scheduled_upstream_stops(
        context[context["journey_id"].isin(scheduled_rows["journey_id"])]
    )
    stop_ids = set(upstream["id"].dropna().astype(str)) | set(scheduled_stops["id"])
    changes = read_snapshots_for_ids(
        stop_ids, kind="change", change_dir=change_dir, plan_dir=plan_dir,
    )
    plans = read_snapshots_for_ids(
        stop_ids, kind="plan", change_dir=change_dir, plan_dir=plan_dir,
    )
    events = _journey_events(rows.loc[~use_scheduled.to_numpy()], upstream, scheduled_rows,
                             scheduled_stops, changes, plans)
    dynamics = delay_dynamics_from_events(rows, events)
    expected_keys = rows[ROW_KEY].reset_index(drop=True)
    if not dynamics[ROW_KEY].reset_index(drop=True).equals(expected_keys):
        raise ValueError(f"{split} dynamics keys do not exactly match prediction_rows_{split}.parquet")

    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics_path = output_dir / f"history_{split}.parquet"
    events_path = output_dir / f"upstream_events_{split}.parquet"
    event_times = set(event_prediction_times)
    persisted_events = events[
        events["prediction_time"].astype(str).isin(event_times)
    ].drop(columns=["_row_id", "_event_kind_order"], errors="ignore")
    for frame, path in ((dynamics, dynamics_path), (persisted_events, events_path)):
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    audit = {
        "split": split,
        "cv_only": split == "cv",
        "holdout_gate_passed": gate_passed,
        "holdout_target_columns_read": False,
        "prediction_rows": int(len(dynamics)),
        "upstream_events": int(len(persisted_events)),
        "legacy_plan_policy": "earliest plan snapshot per stop used as a static schedule in both archive periods, by user instruction (5 and 12 November 2026)",
        "replacement_rows_using_scheduled_run": int(use_scheduled.sum()),
    }
    audit_path = output_dir / f"history_{split}_audit.json"
    audit_tmp = audit_path.with_name(audit_path.stem + ".tmp" + audit_path.suffix)
    audit_tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_tmp.replace(audit_path)
    return dynamics_path, events_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("cv", "holdout"), default="cv")
    parser.add_argument("--feature", choices=("dynamics", "reliability", "all"),
                        default="dynamics")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-dir", type=Path, default=PLAN_DIR)
    parser.add_argument("--change-dir", type=Path, default=CHANGE_DIR)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--allow-partial-archive", action="store_true",
                        help="skip the 27-month completeness check and write to a separate --output-dir")
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir / "advanced"
    if args.allow_partial_archive and output_dir.resolve() == (args.data_dir / "advanced").resolve():
        parser.error("--allow-partial-archive requires a separate --output-dir")
    if args.feature in {"dynamics", "all"}:
        paths = build_history_dynamics_split(
            args.split, plan_dir=args.plan_dir, change_dir=args.change_dir,
            data_dir=args.data_dir, output_dir=output_dir,
            require_full_archive=not args.allow_partial_archive,
            registry_path=args.registry,
        )
        print("dynamics outputs:", *(str(path) for path in paths))
    if args.feature in {"reliability", "all"}:
        from recent_reliability import build_reliability_file

        path = build_reliability_file(
            split=args.split, data_dir=args.data_dir, plan_dir=args.plan_dir,
            change_dir=args.change_dir,
            output=output_dir / f"reliability_{args.split}.parquet",
            require_full_archive=not args.allow_partial_archive,
            registry_path=args.registry,
        )
        print("reliability output:", path)


if __name__ == "__main__":
    main()
