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

HERE = Path(__file__).resolve().parent
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
                               plan_legacy_policy: str = "exclude") -> pd.DataFrame:
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
    if plan_legacy_policy == "exclude":
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
        raise ValueError("plan_legacy_policy must be 'exclude' or 'bounded_sensitivity'")
    plans = latest_snapshot_asof(query, plan_source,
                                 request_bound=plan_request_bound)
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
    occurred = events[
        events["event_occurred_by_cutoff"] & ~events["event_cancelled"]
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


def mask_unlinked_replacement_rows(dynamics: pd.DataFrame,
                                  upstream_events: pd.DataFrame,
                                  replacement_announcements: pd.DataFrame
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mask live dynamics/events for replacement rows before link evidence.

    The monthly `upstream_runs` table describes the operated replacement train.
    Until an origin snapshot announces the scheduled-to-replacement link, those
    IDs cannot safely be treated as the passenger's known run.  Ordinary rows
    and replacement rows with a verified announcement at/before cutoff remain.
    """
    required = set(ROW_KEY) | {
        "is_replacement", "replacement_announced", "replacement_announced_at_utc",
    }
    missing = required - set(replacement_announcements.columns)
    if missing:
        raise ValueError(f"replacement announcements missing {sorted(missing)}")
    announcements = replacement_announcements[list(ROW_KEY) + [
        "is_replacement", "replacement_announced", "replacement_announced_at_utc",
    ]].copy()
    if announcements.duplicated(ROW_KEY).any():
        raise ValueError("replacement announcement keys must be unique")
    announcement_index = announcements.set_index(ROW_KEY)

    def attach(frame: pd.DataFrame, *, unique: bool) -> tuple[pd.DataFrame, pd.Series]:
        if unique and frame.duplicated(ROW_KEY).any():
            raise ValueError("feature/event keys must be unique before masking")
        index = pd.MultiIndex.from_frame(frame[ROW_KEY])
        if len(index.difference(announcement_index.index)):
            raise ValueError("replacement announcement table does not cover all CV keys")
        aligned = announcement_index.reindex(index).reset_index(drop=True)
        out = frame.reset_index(drop=True).copy()
        is_replacement = aligned["is_replacement"].astype("boolean").fillna(False)
        announced = aligned["replacement_announced"].astype("boolean").fillna(False)
        stamp = pd.to_datetime(aligned["replacement_announced_at_utc"], utc=True)
        cutoff = cutoff_instants(out["predicted_at"])
        known = announced & stamp.notna() & stamp.le(cutoff)
        return out, is_replacement & ~known

    features, mask = attach(dynamics, unique=True)
    for column in DELAY_DYNAMICS_COLUMNS:
        if column in features:
            features.loc[mask, column] = np.nan

    events, event_mask = attach(upstream_events, unique=False)
    events = events.loc[~event_mask].reset_index(drop=True)
    return features, events


def build_delay_dynamics(rows: pd.DataFrame, upstream_stops: pd.DataFrame,
                         change_snapshots: pd.DataFrame,
                         plan_snapshots: pd.DataFrame, *,
                         plan_legacy_policy: str = "exclude") -> pd.DataFrame:
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


def build_recent_reliability(rows: pd.DataFrame,
                             outcomes_asof: pd.DataFrame) -> pd.DataFrame:
    """Aggregate already-cutoff-filtered historical outcome states.

    `outcomes_asof` has one row per target prediction and historical journey,
    after its live-change and plan snapshots have been joined at that target's
    cutoff.  Required columns are ROW_KEY, `candidate_journey_id`,
    `history_train_type`, `history_train_number`, `origin_departure_utc`,
    `outcome_known`, `cancelled`, and `arrival_delay_min`.  This boundary makes
    the rolling aggregations cheap to test and prevents them from silently
    reading final monthly outcomes.
    """
    required = set(ROW_KEY) | {
        "candidate_journey_id", "history_train_type", "history_train_number",
        "origin_departure_utc", "outcome_time_utc", "outcome_known", "cancelled",
        "arrival_delay_min",
    }
    missing = required - set(outcomes_asof.columns)
    if missing:
        raise ValueError(f"outcomes_asof is missing columns: {sorted(missing)}")
    if not {"history_train_type", "history_train_number"}.issubset(rows.columns):
        raise ValueError("rows need history_train_type and history_train_number")
    target = rows[ROW_KEY + ["history_train_type", "history_train_number"]].copy()
    if target.duplicated(ROW_KEY).any():
        raise ValueError("prediction rows must be unique by ROW_KEY")
    target_cutoff = cutoff_instants(target["predicted_at"])
    states = outcomes_asof.copy()
    states["origin_departure_utc"] = pd.to_datetime(states["origin_departure_utc"], utc=True)
    states["outcome_time_utc"] = pd.to_datetime(states["outcome_time_utc"], utc=True)
    states["outcome_known"] = states["outcome_known"].fillna(False).astype(bool)
    states["cancelled"] = states["cancelled"].fillna(False).astype(bool)
    state_groups = {key: group.copy() for key, group in states.groupby(ROW_KEY, sort=False)}
    rate_columns = RECENT_RELIABILITY_COLUMNS
    records = []
    periods = [
        ("service", "7d", pd.Timedelta(days=7)),
        ("service", "28d", pd.Timedelta(days=28)),
        ("corridor", "24h", pd.Timedelta(hours=24)),
        ("corridor", "3h", pd.Timedelta(hours=3)),
    ]
    for position, row in enumerate(target.itertuples(index=False)):
        key = tuple(getattr(row, column) for column in ROW_KEY)
        cutoff = target_cutoff.iloc[position]
        candidates = state_groups.get(key)
        record = dict(zip(ROW_KEY, key))
        if candidates is None:
            records.append(record)
            continue
        eligible_base = candidates[
            candidates["outcome_known"]
            & candidates["candidate_journey_id"].ne(row.journey_id)
        ]
        for scope, suffix, window in periods:
            if scope == "service":
                selected = eligible_base[
                    eligible_base["origin_departure_utc"].lt(cutoff)
                    & eligible_base["origin_departure_utc"].ge(cutoff - window)
                ]
                selected = selected[
                    selected["history_train_type"].eq(row.history_train_type)
                    & selected["history_train_number"].astype("string").eq(
                        str(row.history_train_number))
                ]
            else:
                selected = eligible_base[
                    eligible_base["outcome_time_utc"].notna()
                    & eligible_base["outcome_time_utc"].lt(cutoff)
                    & eligible_base["outcome_time_utc"].ge(cutoff - window)
                ]
            selected = selected.drop_duplicates("candidate_journey_id")
            if scope == "service":
                rate_col = f"service_severe_or_cancelled_rate_{suffix}"
                mean_col = f"service_mean_delay_min_{suffix}"
            else:
                rate_col = f"corridor_severe_or_cancelled_rate_{suffix}"
                mean_col = f"corridor_mean_delay_min_{suffix}"
            if selected.empty:
                record[rate_col] = np.nan
                record[mean_col] = np.nan
            else:
                severe = (selected["cancelled"]
                          | selected["arrival_delay_min"].ge(60).fillna(False))
                record[rate_col] = float(severe.mean())
                completed = selected.loc[
                    ~selected["cancelled"] & selected["arrival_delay_min"].notna(),
                    "arrival_delay_min",
                ]
                record[mean_col] = float(completed.mean()) if len(completed) else np.nan
        records.append(record)
    out = pd.DataFrame(records)
    for column in rate_columns:
        if column not in out:
            out[column] = np.nan
    return out[ROW_KEY + rate_columns]


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


def _candidate_pairs_for_target_batch(targets: pd.DataFrame,
                                      candidates: pd.DataFrame) -> pd.DataFrame:
    """Create only 28-day same-service and 24-hour corridor candidate pairs."""
    if targets.empty or candidates.empty:
        return pd.DataFrame(columns=ROW_KEY + ["candidate_journey_id"])
    service_index: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in candidates.dropna(subset=["scheduled_train_type",
                                                "scheduled_train_number"]).groupby(
            ["scheduled_train_type", "scheduled_train_number"], sort=False):
        service_index[(str(key[0]), str(key[1]))] = group
    replacement_index: dict[tuple[str, str], pd.DataFrame] = {}
    replacement_candidates = candidates.dropna(subset=["replacement_train_type",
                                                         "replacement_train_number"])
    for key, group in replacement_candidates.groupby(
            ["replacement_train_type", "replacement_train_number"], sort=False):
        replacement_index[(str(key[0]), str(key[1]))] = group

    sorted_candidates = candidates.sort_values("run_start_utc").reset_index(drop=True)
    sorted_times = sorted_candidates["run_start_utc"].dt.as_unit("ns").astype("int64").to_numpy()
    rows: list[dict] = []
    broad_horizon = pd.Timedelta(days=32)
    # Include the corridor window plus a conservative maximum journey time so
    # recent arrivals whose origin departure was more than 24 hours ago remain
    # eligible for the outcome-time aggregation.
    corridor_horizon = pd.Timedelta(hours=48)
    for target in targets.to_dict("records"):
        cutoff = target["_cutoff_utc"]
        lower_broad = cutoff - broad_horizon
        lower_corridor = cutoff - corridor_horizon
        upper = np.searchsorted(sorted_times, cutoff.value, side="left")
        lower = np.searchsorted(sorted_times, lower_broad.value, side="left")
        recent = sorted_candidates.iloc[lower:upper]
        selected = recent.loc[recent["run_start_utc"].ge(lower_corridor)]
        service_type = target["history_train_type"]
        service_number = target["history_train_number"]
        if pd.notna(service_type) and pd.notna(service_number):
            key = (str(service_type), str(service_number))
            same = []
            for index in (service_index, replacement_index):
                group = index.get(key)
                if group is not None:
                    same.append(group.loc[
                        group["run_start_utc"].ge(lower_broad)
                        & group["run_start_utc"].lt(cutoff)
                    ])
            if same:
                selected = pd.concat([selected, *same], ignore_index=False)
        selected = selected.drop_duplicates("journey_id")
        selected = selected.loc[selected["journey_id"].ne(target["journey_id"])]
        for candidate in selected.itertuples(index=False):
            record = {column: target[column] for column in ROW_KEY}
            record["candidate_journey_id"] = candidate.journey_id
            rows.append(record)
    if not rows:
        return pd.DataFrame(columns=ROW_KEY + ["candidate_journey_id"])
    return pd.DataFrame(rows).drop_duplicates(ROW_KEY + ["candidate_journey_id"])


def _candidate_outcomes_asof(pairs: pd.DataFrame, candidates: pd.DataFrame,
                             change_snapshots: pd.DataFrame,
                             plan_snapshots: pd.DataFrame) -> pd.DataFrame:
    """Join candidate origin/arrival state at each target cutoff."""
    if pairs.empty:
        return pd.DataFrame(columns=ROW_KEY + [
            "candidate_journey_id", "origin_departure_utc", "outcome_time_utc", "outcome_known",
            "cancelled", "arrival_delay_min",
        ])
    candidate_meta = candidates[["journey_id", "origin_id", "destination_id"]]
    paired = pairs.merge(candidate_meta, left_on="candidate_journey_id",
                         right_on="journey_id", how="left", validate="many_to_one",
                         suffixes=("", "_candidate"))
    paired["cutoff_utc"] = cutoff_instants(paired["predicted_at"])
    paired["_pair_id"] = np.arange(len(paired), dtype=np.int64)
    endpoint_frames = []
    for endpoint, id_column in (("origin", "origin_id"), ("destination", "destination_id")):
        frame = paired[["_pair_id", id_column, "cutoff_utc"]].rename(
            columns={id_column: "id"}
        ).copy()
        frame["endpoint"] = endpoint
        endpoint_frames.append(frame)
    queries = pd.concat(endpoint_frames, ignore_index=True)
    queries["_query_id"] = np.arange(len(queries), dtype=np.int64)
    plans = plan_snapshots.copy()
    if not plans.empty:
        plans = plans.loc[pd.to_datetime(plans["snapshot_timestamp"], utc=True)
                          .ge(PLAN_EXACT_UTC_FROM)].copy()
    asof_plan = latest_snapshot_asof(queries, plans, request_bound=False)
    asof_change = latest_snapshot_asof(queries, change_snapshots, request_bound=True)
    plan_columns = [column for column in (
        "_query_id", "arrival_planned_time", "departure_planned_time",
        "train_type", "train_number", "snapshot_timestamp",
    ) if column in asof_plan]
    change_columns = [column for column in (
        "_query_id", "arrival_change_time", "arrival_cancellation_time",
        "arrival_change_status", "departure_change_time",
        "departure_cancellation_time", "departure_change_status",
    ) if column in asof_change]
    state = queries[["_pair_id", "endpoint", "cutoff_utc", "_query_id"]].copy()
    if len(plan_columns) > 1:
        plan = asof_plan[plan_columns].rename(columns={
            "arrival_planned_time": "_arrival_plan",
            "departure_planned_time": "_departure_plan",
            "train_type": "_plan_train_type", "train_number": "_plan_train_number",
            "snapshot_timestamp": "plan_snapshot_timestamp",
        })
        state = state.merge(plan, on="_query_id", how="left", validate="one_to_one")
    if len(change_columns) > 1:
        changes = asof_change[change_columns]
        state = state.merge(changes, on="_query_id", how="left", validate="one_to_one")
    for column in ("_arrival_plan", "_departure_plan", "arrival_change_time",
                   "arrival_cancellation_time", "arrival_change_status",
                   "departure_change_time", "departure_cancellation_time",
                   "departure_change_status"):
        if column not in state:
            state[column] = pd.NaT
    origin = state.loc[state["endpoint"].eq("origin")].set_index("_pair_id")
    destination = state.loc[state["endpoint"].eq("destination")].set_index("_pair_id")
    origin_departure_utc = local_clock_to_utc(origin["_departure_plan"], ambiguous=False)
    planned_arrival_utc = local_clock_to_utc(destination["_arrival_plan"], ambiguous=False)
    reported_arrival_utc = local_clock_to_utc(destination["arrival_change_time"], ambiguous=False)
    cutoff = paired.set_index("_pair_id")["cutoff_utc"]
    origin_cancelled = _cancellation_in_force(
        origin["departure_cancellation_time"], origin["departure_change_status"],
        origin["cutoff_utc"],
    )
    destination_cancelled = _cancellation_in_force(
        destination["arrival_cancellation_time"], destination["arrival_change_status"],
        destination["cutoff_utc"],
    )
    cancelled = (origin_cancelled | destination_cancelled).reindex(paired["_pair_id"])
    arrival_known = (
        destination["arrival_change_time"].notna()
        & reported_arrival_utc.le(destination["cutoff_utc"])
        & planned_arrival_utc.notna()
    )
    outcome_known = (cancelled.to_numpy() | arrival_known.reindex(paired["_pair_id"]).fillna(False).to_numpy())
    delay = (reported_arrival_utc - planned_arrival_utc).dt.total_seconds() / 60
    paired["origin_departure_utc"] = origin_departure_utc.reindex(paired["_pair_id"]).to_numpy()
    outcome_time = reported_arrival_utc.where(
        ~cancelled & arrival_known.reindex(paired["_pair_id"]).fillna(False)
    )
    cancellation_arrival_time = planned_arrival_utc.where(
        cancelled & planned_arrival_utc.le(paired.set_index("_pair_id")["cutoff_utc"])
    )
    outcome_time = outcome_time.fillna(cancellation_arrival_time)
    paired["outcome_time_utc"] = outcome_time.reindex(paired["_pair_id"]).to_numpy()
    paired["origin_train_type_asof"] = (
        origin.get("_plan_train_type", pd.Series(index=origin.index, dtype="object"))
        .reindex(paired["_pair_id"]).to_numpy()
    )
    paired["origin_train_number_asof"] = (
        origin.get("_plan_train_number", pd.Series(index=origin.index, dtype="object"))
        .reindex(paired["_pair_id"]).to_numpy()
    )
    paired["outcome_known"] = outcome_known
    paired["cancelled"] = cancelled.to_numpy()
    paired["arrival_delay_min"] = delay.reindex(paired["_pair_id"]).to_numpy()
    return paired[ROW_KEY + ["candidate_journey_id", "origin_departure_utc", "outcome_time_utc",
                             "origin_train_type_asof", "origin_train_number_asof",
                             "outcome_known", "cancelled", "arrival_delay_min"]]


def build_recent_reliability_cv_file(
    *,
    split: str = "cv",
    data_dir: Path = DATA_DIR,
    plan_dir: Path = PLAN_DIR,
    change_dir: Path = CHANGE_DIR,
    output: Path | None = None,
    batch_target_rows: int = 200,
    max_target_rows: int | None = None,
    require_full_archive: bool = True,
    registry_path: Path | None = None,
) -> Path:
    """Build split-specific recent-reliability features from visible snapshots.

    The source candidates come from the selected split roster, but only stop IDs
    and scheduled-service metadata are used. Candidate pairs are bounded to 32 days,
    and only same-service candidates plus the corridor's prior 24-hour window
    are materialized. Legacy plan snapshots are excluded because their request
    availability is not evidenced.
    """
    if split not in {"cv", "holdout"}:
        raise ValueError("split must be 'cv' or 'holdout'")
    gate_passed = False
    if split == "holdout":
        from holdout_gate import REGISTRY, committed_registration
        committed_registration(registry_path or REGISTRY)
        gate_passed = True
    if batch_target_rows < 1:
        raise ValueError("batch_target_rows must be positive")
    if require_full_archive:
        expected = {f"data-{month}.parquet" for month in
                    pd.period_range("2024-07", "2026-09", freq="M").astype(str)}
        for label, directory in (("plan", plan_dir), ("change", change_dir)):
            actual = {path.name for path in directory.glob("data-????-??.parquet")}
            missing = sorted(expected - actual)
            if missing:
                raise FileNotFoundError(
                    f"incomplete {label} archive at {directory}; missing {missing[:4]}"
                )
    rows_path = data_dir / f"prediction_rows_{split}.parquet"
    rows = pd.read_parquet(rows_path, columns=ROW_KEY)
    rows = rows.copy()
    if max_target_rows is not None:
        if output is None:
            raise ValueError("bounded sample runs must specify a separate output path")
        rows = rows.tail(max_target_rows).copy()
    if rows.duplicated(ROW_KEY).any():
        raise ValueError(f"{split} prediction rows must be unique by ROW_KEY")
    context_path = data_dir / "advanced" / f"replacement_context_{split}.parquet"
    announcements_path = data_dir / "advanced" / f"replacement_announcements_{split}.parquet"
    if not context_path.is_file() or not announcements_path.is_file():
        raise FileNotFoundError(
            f"run the {split} timetable builder before reliability extraction"
        )
    context = pd.read_parquet(context_path)
    candidate_context = context
    if split == "holdout":
        cv_context = pd.read_parquet(
            data_dir / "advanced" / "replacement_context_cv.parquet"
        )
        candidate_context = pd.concat([cv_context, context], ignore_index=True)
        candidate_context = candidate_context.drop_duplicates("journey_id", keep="last")
    announcements = pd.read_parquet(announcements_path)
    from scheduled_timetable import replacement_announcements_asof, service_keys_asof
    target_keys = service_keys_asof(rows, pd.DataFrame(), context, announcements)
    rows = rows.merge(target_keys[[*ROW_KEY, "history_train_type", "history_train_number"]],
                      on=ROW_KEY, how="left", validate="one_to_one")
    rows["_cutoff_utc"] = cutoff_instants(rows["predicted_at"])

    stops = pd.read_parquet(data_dir / "stops.parquet", columns=[
        "id", "journey_id", "station_name", "train_line_station_num",
    ])
    stops = stops.loc[stops["station_name"].isin(
        ["Frankfurt (Main) Hbf", "München Hbf"]
    )].copy()
    stops = stops.sort_values(["journey_id", "train_line_station_num"])
    origin = (stops.loc[stops["station_name"].eq("Frankfurt (Main) Hbf")]
              .drop_duplicates("journey_id").rename(columns={"id": "origin_id"}))
    destination = (stops.loc[stops["station_name"].eq("München Hbf")]
                   .drop_duplicates("journey_id").rename(columns={"id": "destination_id"}))
    candidates = origin[["journey_id", "origin_id"]].merge(
        destination[["journey_id", "destination_id"]], on="journey_id", how="inner",
        validate="one_to_one",
    ).merge(candidate_context, on="journey_id", how="inner", validate="one_to_one")
    candidates["run_start_utc"] = _journey_run_start_utc(candidates["journey_id"])
    candidates = candidates.dropna(subset=["run_start_utc"]).sort_values("run_start_utc")
    candidates = candidates.drop_duplicates("journey_id").reset_index(drop=True)

    outputs = []
    exact = rows.loc[rows["_cutoff_utc"].ge(PLAN_EXACT_UTC_FROM)].copy()
    unavailable = rows.loc[~rows["_cutoff_utc"].ge(PLAN_EXACT_UTC_FROM), ROW_KEY].copy()
    if len(unavailable):
        empty = unavailable.copy()
        for column in RECENT_RELIABILITY_COLUMNS:
            empty[column] = np.nan
        outputs.append(empty)

    processed_months: list[str] = []
    exact["_cutoff_month"] = exact["_cutoff_utc"].dt.tz_localize(None).dt.to_period("M")
    for month, month_targets in exact.groupby("_cutoff_month", sort=True):
        print(f"reliability CV {month}: {len(month_targets)} target rows", flush=True)
        month_targets = month_targets.reset_index(drop=True)
        lower_bound = month_targets["_cutoff_utc"].min() - pd.Timedelta(days=32)
        upper_bound = month_targets["_cutoff_utc"].max()
        month_candidates = candidates.loc[
            candidates["run_start_utc"].ge(lower_bound)
            & candidates["run_start_utc"].le(upper_bound)
        ].copy()
        if month_candidates.empty:
            empty = month_targets[ROW_KEY].copy()
            for column in RECENT_RELIABILITY_COLUMNS:
                empty[column] = np.nan
            outputs.append(empty)
            continue
        source_month_start = (lower_bound.tz_convert("Europe/Berlin").tz_localize(None)
                              - pd.Timedelta(days=3)).to_period("M")
        source_month_end = upper_bound.tz_convert("Europe/Berlin").tz_localize(None).to_period("M")
        source_months = list(pd.period_range(source_month_start, source_month_end, freq="M"))
        processed_months.extend(str(value) for value in source_months)
        event_ids = set(month_candidates["origin_id"].astype(str)) | set(
            month_candidates["destination_id"].astype(str)
        )
        plans = _read_snapshot_ids_in_months(
            event_ids, kind="plan", months=source_months,
            change_dir=change_dir, plan_dir=plan_dir,
        )
        changes = _read_snapshot_ids_in_months(
            event_ids, kind="change", months=source_months,
            change_dir=change_dir, plan_dir=plan_dir,
        )
        print(f"  loaded {len(plans):,} plan and {len(changes):,} change snapshots "
              f"for {len(month_candidates):,} candidate journeys", flush=True)
        batches = []
        for start in range(0, len(month_targets), batch_target_rows):
            target_batch = month_targets.iloc[start:start + batch_target_rows].copy()
            pairs = _candidate_pairs_for_target_batch(target_batch, month_candidates)
            if pairs.empty:
                features = target_batch[ROW_KEY].copy()
                for column in RECENT_RELIABILITY_COLUMNS:
                    features[column] = np.nan
                batches.append(features)
                continue
            pair_keys = ROW_KEY + ["candidate_journey_id"]
            candidate_announcements = replacement_announcements_asof(
                pairs, candidate_context, changes, entity_col="candidate_journey_id",
                context_id_col="journey_id", row_keys=pair_keys,
            )
            candidate_keys = service_keys_asof(
                pairs, pd.DataFrame(), candidate_context, candidate_announcements,
                entity_col="candidate_journey_id", context_id_col="journey_id", row_keys=pair_keys,
            )
            states = _candidate_outcomes_asof(pairs, month_candidates, changes, plans)
            states = states.merge(
                candidate_keys[ROW_KEY + ["candidate_journey_id", "history_train_type",
                                          "history_train_number"]],
                on=pair_keys, how="left", validate="one_to_one",
            )
            states = states.merge(
                candidate_announcements[ROW_KEY + ["candidate_journey_id", "is_replacement"]],
                on=pair_keys, how="left", validate="one_to_one",
            )
            ordinary = ~states["is_replacement"].fillna(False).astype(bool)
            states.loc[ordinary, "history_train_type"] = states.loc[
                ordinary, "origin_train_type_asof"
            ].astype("string")
            states.loc[ordinary, "history_train_number"] = states.loc[
                ordinary, "origin_train_number_asof"
            ].astype("string")
            feature_rows = target_batch[ROW_KEY + [
                "history_train_type", "history_train_number",
            ]].copy()
            batches.append(build_recent_reliability(feature_rows, states))
        outputs.append(pd.concat(batches, ignore_index=True))

    result = pd.concat(outputs, ignore_index=True).sort_values(ROW_KEY).reset_index(drop=True)
    result = result[ROW_KEY + RECENT_RELIABILITY_COLUMNS]
    if len(result) != len(rows) or result.duplicated(ROW_KEY).any():
        raise ValueError("reliability output does not preserve unique CV prediction keys")
    output = output or data_dir / "advanced" / f"reliability_{split}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    coverage = {}
    for prediction_time, group in result.groupby("prediction_time", observed=True):
        coverage[str(prediction_time)] = {
            "rows": int(len(group)),
            "rows_with_any_reliability": int(group[RECENT_RELIABILITY_COLUMNS].notna().any(axis=1).sum()),
            "known_by_feature": {column: int(group[column].notna().sum())
                                 for column in RECENT_RELIABILITY_COLUMNS},
        }
    audit = {
        "split": split,
        "cv_only": split == "cv",
        "holdout_gate_passed": gate_passed,
        "holdout_target_columns_read": False,
        "strict_plan_availability_from_utc": str(PLAN_EXACT_UTC_FROM),
        "legacy_plan_policy": "exclude; legacy plan labels have no verified request-time bound",
        "candidate_policy": "same-service candidates within 28d, corridor outcomes within 24h/3h; 32d service and 48h corridor run-ID prefilters",
        "candidate_source": "split journey roster; only stop IDs and replacement context used",
        "prediction_rows": int(len(result)),
        "rows_with_cutoff_before_plan_provenance": int(len(unavailable)),
        "source_months_read": sorted(set(processed_months)),
        "coverage_by_prediction_time": coverage,
    }
    audit_path = output.with_name(output.stem + "_audit.json")
    audit_tmp = audit_path.with_name(audit_path.stem + ".tmp" + audit_path.suffix)
    audit_tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_tmp.replace(audit_path)
    return output


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
    stop_ids = set(upstream["id"].dropna().astype(str))
    changes = read_snapshots_for_ids(
        stop_ids, kind="change", change_dir=change_dir, plan_dir=plan_dir,
    )
    plans = read_snapshots_for_ids(
        stop_ids, kind="plan", change_dir=change_dir, plan_dir=plan_dir,
    )
    events = build_upstream_events_asof(rows, upstream, changes, plans,
                                        plan_legacy_policy="exclude")
    dynamics = delay_dynamics_from_events(rows, events)
    dynamics, events = mask_unlinked_replacement_rows(dynamics, events, announcements)
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
        "replacement_rows_masked_until_link": int(
            (announcements["is_replacement"].fillna(False)
             & ~announcements["replacement_announced"].fillna(False)).sum()
        ),
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
                        help="skip the 27-month completeness check for a bounded source sample")
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir / "advanced"
    if args.feature in {"dynamics", "all"}:
        paths = build_history_dynamics_split(
            args.split, plan_dir=args.plan_dir, change_dir=args.change_dir,
            data_dir=args.data_dir, output_dir=output_dir,
            require_full_archive=not args.allow_partial_archive,
            registry_path=args.registry,
        )
        print("dynamics outputs:", *(str(path) for path in paths))
    if args.feature in {"reliability", "all"}:
        path = build_recent_reliability_cv_file(
            split=args.split, data_dir=args.data_dir, plan_dir=args.plan_dir,
            change_dir=args.change_dir,
            output=output_dir / f"reliability_{args.split}.parquet",
            require_full_archive=not args.allow_partial_archive,
            registry_path=args.registry,
        )
        print("reliability output:", path)


if __name__ == "__main__":
    main()
