"""Planned timetable features from the timestamped plan archive.

The parsed plan archive is published at
https://huggingface.co/datasets/piebro/deutsche-bahn-data/tree/d27b9c413536bb5fe393954371f572050e55176b/monthly_processed_data_plan
and documents ``snapshot_timestamp`` as the API request time. The older
fetcher encoded the requested timetable hour in ``{eva}_plan_{hour}.xml``.
Following the user decision on 5 November 2026, the earliest legacy plan
per stop is accepted as its static schedule regardless of fetch time.
Exact-request updates remain cutoff-bound.

The default builder reads only CV journeys and their stop IDs.  The optional
holdout builder calls the committed-registration gate before its first holdout
read and projects only identifiers and timetable metadata.  Plan event times
are German local wall-clock times; plan snapshot timestamps are interpreted
as UTC.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__:
    from .history_features import (
        CHANGE_DIR, PLAN_DIR, PLAN_COLUMNS, PLAN_EXACT_UTC_FROM, ROW_KEY, cutoff_instants,
        request_upper_bound_utc, plan_availability_utc, prepare_plan_snapshots, latest_snapshot_asof,
    )
else:
    from history_features import (
        CHANGE_DIR, PLAN_DIR, PLAN_COLUMNS, PLAN_EXACT_UTC_FROM, ROW_KEY, cutoff_instants,
        request_upper_bound_utc, plan_availability_utc, prepare_plan_snapshots, latest_snapshot_asof,
    )

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ADVANCED = DATA / "advanced"
DEFAULT_REGISTRY = HERE / "modeling" / "finalists.json"
DEFAULT_PLAN_DIR = PLAN_DIR
DEFAULT_CHANGE_DIR = CHANGE_DIR

ORIGIN = "Frankfurt (Main) Hbf"
DESTINATION = "München Hbf"
ARCHIVE_REVISION = "d27b9c413536bb5fe393954371f572050e55176b"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/piebro/deutsche-bahn-data/tree/"
    f"{ARCHIVE_REVISION}/monthly_processed_data_plan"
)
UPSTREAM_REPO_URL = "https://github.com/piebro/deutsche-bahn-data"
LEGACY_FETCH_URL = (
    "https://github.com/piebro/deutsche-bahn-data/blob/"
    "f19888b770a68d5f9010adac33af70e400e2395d/fetch_data.py"
)
# First exact request timestamp observed in the pinned plan archive.  Earlier
# whole-hour stamps are target plan-hour labels according to the old fetcher.
EXACT_PLAN_REQUESTS_FROM_UTC = PLAN_EXACT_UTC_FROM

STOP_COLUMNS = [
    "id", "journey_id", "station_name", "train_type", "train_number",
    "is_replacement_train", "replaced_train_type", "replaced_train_number",
    "train_line_ride_id", "train_line_station_num", "stop_index",
    "arrival_planned_time", "departure_planned_time",
]
UPSTREAM_COLUMNS = ["id", "journey_id", "train_line_station_num", "station_name"]
TIMETABLE_COLUMNS = [
    "scheduled_train_type", "scheduled_starts_at_origin", "weather_branch",
    "planned_stops", "route_id", "upstream_planned_min",
]


def _asof_by_id(queries: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest plan state under the static-legacy timetable policy."""
    if queries.empty or snapshots.empty:
        return queries.copy()
    available = snapshots.loc[snapshots["available_at_utc"].notna()]
    return latest_snapshot_asof(queries, available, observed_col="available_at_utc", request_bound=False)


def _path_to_destination(path: object) -> list[str] | None:
    if not isinstance(path, (list, tuple, np.ndarray)) or len(path) == 0:
        return None
    values = [str(value).strip() for value in path if pd.notna(value) and str(value).strip()]
    if DESTINATION not in values:
        return None
    return values[: values.index(DESTINATION) + 1]


def _branch_from_route(route: list[str] | None) -> str | pd.NA:
    if not route:
        return pd.NA
    values = set(route)
    via_nuernberg = bool(values.intersection({"Nürnberg Hbf", "Würzburg Hbf"}))
    via_stuttgart = bool(values.intersection({"Stuttgart Hbf", "Ulm Hbf", "Mannheim Hbf"}))
    if via_nuernberg == via_stuttgart:
        return pd.NA
    return "nuernberg" if via_nuernberg else "stuttgart"


def _plan_features(origin_plan: pd.Series, upstream_plan: pd.Series | None) -> dict:
    route_tail = _path_to_destination(origin_plan.get("departure_planned_path"))
    route = [ORIGIN, *route_tail] if route_tail else None
    starts = pd.NA
    source_id = origin_plan.get("id")
    if isinstance(source_id, str):
        try:
            starts = int(source_id.rsplit("-", 1)[-1]) == 1
        except ValueError:
            starts = pd.NA

    upstream_min: float = np.nan
    arrival_at_origin = origin_plan.get("arrival_planned_time")
    if starts is True or starts is np.bool_(True):
        upstream_min = 0.0
    elif upstream_plan is not None and pd.notna(arrival_at_origin):
        first_departure = upstream_plan.get("departure_planned_time")
        if pd.notna(first_departure):
            delta = pd.Timestamp(arrival_at_origin) - pd.Timestamp(first_departure)
            upstream_min = delta.total_seconds() / 60.0

    return {
        "scheduled_train_type": origin_plan.get("train_type", pd.NA),
        "scheduled_starts_at_origin": starts,
        "weather_branch": _branch_from_route(route),
        "planned_stops": len(route) if route else pd.NA,
        "route_id": " | ".join(route) if route else pd.NA,
        "upstream_planned_min": upstream_min,
    }


def build_timetable_features(
    prediction_rows: pd.DataFrame,
    replacement_context: pd.DataFrame,
    plan_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Build one timetable row per prediction key under the registered plan policy.

    ``replacement_context`` has one row per journey and carries
    ``scheduled_origin_id``.  For ordinary trains this is the current origin
    stop ID; for a retained replacement it is the matched original scheduled
    train's origin stop ID.  Every timetable feature is from that scheduled
    service at every prediction cutoff.  All plan rows must already carry
    ``available_at_utc`` produced by :func:`plan_availability_utc`.
    """
    missing = set(ROW_KEY) - set(prediction_rows.columns)
    if missing:
        raise ValueError(f"prediction_rows are missing keys: {sorted(missing)}")
    if "split" in prediction_rows and not prediction_rows["split"].eq("cv").all():
        raise PermissionError("timetable features are restricted to CV prediction rows")
    if prediction_rows.duplicated(ROW_KEY).any():
        raise ValueError("prediction rows must be unique by ROW_KEY")
    required_context = {"journey_id", "scheduled_origin_id"}
    if not required_context.issubset(replacement_context.columns):
        raise ValueError(f"replacement_context must contain {sorted(required_context)}")
    if replacement_context["journey_id"].duplicated().any():
        raise ValueError("replacement_context must have one row per journey")
    target = prediction_rows[ROW_KEY].copy().reset_index(drop=True)
    target["_row_id"] = np.arange(len(target), dtype=np.int64)
    target["cutoff_utc"] = cutoff_instants(target["predicted_at"])
    target = target.merge(replacement_context, on="journey_id", how="left", validate="many_to_one")
    # D8 defines every planned-timetable feature from the scheduled service.
    # For ordinary services current_origin_id == scheduled_origin_id; retained
    # replacements use the matched original service at every prediction time.
    target["plan_query_id"] = target["scheduled_origin_id"]

    target["_query_id"] = np.arange(len(target), dtype=np.int64)
    queries = target[[
        "_query_id", "journey_id", "prediction_time", "predicted_at", "cutoff_utc",
        "plan_query_id",
    ]].rename(columns={"plan_query_id": "id"})
    origin = _asof_by_id(queries, plan_snapshots)

    # Upstream minutes need the first stop's planned departure at the same
    # cutoff.  The source stop ID sequence is stable within a ride ID.
    origin_ids = origin.get("id")
    upstream_ids: list[object] = []
    for value in origin_ids if origin_ids is not None else []:
        if pd.isna(value):
            upstream_ids.append(pd.NA)
            continue
        try:
            ride_id, _ = str(value).rsplit("-", 1)
            upstream_ids.append(f"{ride_id}-1")
        except ValueError:
            upstream_ids.append(pd.NA)
    upstream_queries = pd.DataFrame({
        "_query_id": origin["_query_id"], "id": upstream_ids,
        "cutoff_utc": origin["cutoff_utc"],
    })
    upstream = _asof_by_id(upstream_queries, plan_snapshots)
    upstream = upstream.set_index("_query_id")

    result_rows = []
    for _, row in origin.iterrows():
        if pd.isna(row.get("available_at_utc")):
            feature = {column: pd.NA for column in TIMETABLE_COLUMNS}
            feature["upstream_planned_min"] = np.nan
        else:
            prior = upstream.loc[row["_query_id"]]
            prior = None if pd.isna(prior.get("available_at_utc")) else prior
            feature = _plan_features(row, prior)
        result_rows.append(feature)
    result = target[ROW_KEY].copy().reset_index(drop=True)
    for column in TIMETABLE_COLUMNS:
        result[column] = [value.get(column, pd.NA) for value in result_rows]
    result["scheduled_train_type"] = result["scheduled_train_type"].astype("string")
    result["scheduled_starts_at_origin"] = result["scheduled_starts_at_origin"].astype("boolean")
    result["weather_branch"] = result["weather_branch"].astype("string")
    result["planned_stops"] = pd.to_numeric(result["planned_stops"], errors="coerce").astype("Int64")
    result["route_id"] = result["route_id"].astype("string")
    result["upstream_planned_min"] = pd.to_numeric(
        result["upstream_planned_min"], errors="coerce"
    ).astype("float64")
    return result


def replacement_announcements_asof(
    prediction_rows: pd.DataFrame,
    replacement_context: pd.DataFrame,
    change_snapshots: pd.DataFrame,
    *,
    entity_col: str = "journey_id",
    context_id_col: str = "journey_id",
    row_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Return replacement map evidence visible at each CV cutoff.

    ``replacement_context`` is one row per journey and supplies
    ``current_origin_id``.  A map is announced only when a cutoff-eligible
    change snapshot for that stop carries a replaced train type/number.
    Change snapshots use the existing change-source conservative request-time
    bound (+45 minutes for old six-hour labels); plan snapshots do not.
    """
    row_keys = list(row_keys or ROW_KEY)
    required_rows = set(row_keys) | {entity_col, "predicted_at"}
    if not required_rows.issubset(prediction_rows.columns):
        raise ValueError(f"prediction_rows must contain {sorted(required_rows)}")
    if prediction_rows.duplicated(row_keys).any():
        raise ValueError(f"prediction rows must be unique by {row_keys}")
    context_columns = {
        context_id_col, "current_origin_id", "scheduled_train_type",
        "scheduled_train_number", "is_replacement_train",
    }
    if not context_columns.issubset(replacement_context.columns):
        raise ValueError(f"replacement_context must contain {sorted(context_columns)}")
    target = prediction_rows[row_keys].copy().reset_index(drop=True)
    target["_query_id"] = np.arange(len(target), dtype=np.int64)
    target["cutoff_utc"] = cutoff_instants(target["predicted_at"])
    context = replacement_context[[
        context_id_col, "current_origin_id", "scheduled_train_type",
        "scheduled_train_number", "is_replacement_train",
    ]].rename(columns={context_id_col: entity_col})
    if context[entity_col].duplicated().any():
        raise ValueError("replacement_context must have one row per entity")
    target = target.merge(context, on=entity_col, how="left", validate="many_to_one")
    target = target.rename(columns={"current_origin_id": "id"})
    if change_snapshots.empty:
        target["is_replacement"] = target["is_replacement_train"].fillna(False)
        target["replacement_announced"] = False
        target["replacement_announced_at_utc"] = pd.NaT
        target["announced_scheduled_train_type"] = pd.NA
        target["announced_scheduled_train_number"] = pd.NA
        return target[[*row_keys, "is_replacement", "replacement_announced",
                       "replacement_announced_at_utc", "announced_scheduled_train_type",
                       "announced_scheduled_train_number"]]
    required = {"id", "snapshot_timestamp", "replaced_train_type", "replaced_train_number"}
    if not required.issubset(change_snapshots.columns):
        raise ValueError(f"change_snapshots must contain {sorted(required)}")
    changes = change_snapshots.copy()
    changes["available_at_utc"] = request_upper_bound_utc(changes["snapshot_timestamp"])
    has_map = changes["replaced_train_type"].notna() & changes["replaced_train_number"].notna()
    changes = changes[has_map]
    queries = target[["_query_id", "id", "cutoff_utc"]].dropna(subset=["id"])
    if changes.empty or queries.empty:
        target["is_replacement"] = target["is_replacement_train"].fillna(False)
        target["replacement_announced"] = False
        target["replacement_announced_at_utc"] = pd.NaT
        target["announced_scheduled_train_type"] = pd.NA
        target["announced_scheduled_train_number"] = pd.NA
    else:
        left = queries.sort_values(["cutoff_utc", "id"])
        right = changes[["id", "available_at_utc", "replaced_train_type", "replaced_train_number"]].drop_duplicates().sort_values(
            ["available_at_utc", "id"]
        )
        matched = pd.merge_asof(
            left, right, left_on="cutoff_utc", right_on="available_at_utc", by="id",
            direction="backward", allow_exact_matches=True,
        )
        matched = matched.set_index("_query_id")
        target["replacement_announced_at_utc"] = target["_query_id"].map(matched["available_at_utc"])
        target["announced_scheduled_train_type"] = target["_query_id"].map(matched["replaced_train_type"])
        target["announced_scheduled_train_number"] = target["_query_id"].map(matched["replaced_train_number"])
        is_replacement = target["is_replacement_train"].fillna(False)
        target["replacement_announced"] = (
            is_replacement
            & target["announced_scheduled_train_type"].astype("string").eq(
                target["scheduled_train_type"].astype("string")
            ).fillna(False)
            & target["announced_scheduled_train_number"].astype("string").eq(
                target["scheduled_train_number"].astype("string")
            ).fillna(False)
        )
        target.loc[~target["replacement_announced"], "replacement_announced_at_utc"] = pd.NaT
        target.loc[~target["replacement_announced"], [
            "announced_scheduled_train_type", "announced_scheduled_train_number"
        ]] = pd.NA
    target["is_replacement"] = target["is_replacement_train"].fillna(False)
    target["replacement_announced"] = target["replacement_announced"].astype("boolean")
    target["is_replacement"] = target["is_replacement"].astype("boolean")
    return target[[*row_keys, "is_replacement", "replacement_announced",
                   "replacement_announced_at_utc", "announced_scheduled_train_type",
                   "announced_scheduled_train_number"]]


def service_keys_asof(
    rows: pd.DataFrame,
    journeys: pd.DataFrame,
    replacement_context: pd.DataFrame,
    announcements: pd.DataFrame,
    *,
    entity_col: str = "journey_id",
    context_id_col: str = "journey_id",
    row_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Attach scheduled-service lookup keys, switching on visible announcement.

    ``rows`` may be the CV prediction table or a candidate
    prediction-row-by-source-journey table.  It must have ROW_KEY plus
    ``journey_id``.  ``journeys`` and ``replacement_context`` provide the
    journey-level scheduled/replacement type and number.  Output key columns
    are nullable strings so unknown services remain missing instead of becoming
    a string such as ``"<NA>"``.
    """
    del journeys  # Reserved in the signature for caller compatibility/context.
    row_keys = list(row_keys or ROW_KEY)
    needed = set(row_keys) | {entity_col, "predicted_at"}
    if not needed.issubset(rows.columns):
        raise ValueError(f"rows must contain {sorted(needed)}")
    context_columns = [
        context_id_col, "scheduled_train_type", "scheduled_train_number",
        "replacement_train_type", "replacement_train_number", "is_replacement_train",
    ]
    missing = set(context_columns) - set(replacement_context.columns)
    if missing:
        raise ValueError(f"replacement_context is missing {sorted(missing)}")
    out = rows[row_keys].copy().reset_index(drop=True)
    out["_key_row"] = np.arange(len(out), dtype=np.int64)
    context = replacement_context[context_columns].rename(columns={context_id_col: entity_col})
    if context[entity_col].duplicated().any():
        raise ValueError("replacement_context must have one row per entity")
    out = out.merge(
        context, on=entity_col, how="left",
        validate="many_to_one",
    )
    if announcements is not None and not announcements.empty:
        out = out.merge(
            announcements[[*row_keys, "replacement_announced"]], on=row_keys,
            how="left", validate="many_to_one",
        )
    else:
        out["replacement_announced"] = False
    switched = out["replacement_announced"].fillna(False).astype(bool)
    out["history_train_type"] = out["scheduled_train_type"].where(
        ~switched, out["replacement_train_type"]
    ).astype("string")
    out["history_train_number"] = out["scheduled_train_number"].where(
        ~switched, out["replacement_train_number"]
    ).astype("string")
    out["history_train_key"] = (
        out["history_train_type"].fillna("") + ":" + out["history_train_number"].fillna("")
    ).astype("string")
    missing_key = out["history_train_type"].isna() | out["history_train_number"].isna()
    out.loc[missing_key, "history_train_key"] = pd.NA
    key_is_replacement = pd.Series(pd.NA, index=out.index, dtype="boolean")
    known_context = out["is_replacement_train"].notna()
    key_is_replacement.loc[known_context] = False
    key_is_replacement.loc[switched & ~missing_key] = True
    out["service_key_is_replacement"] = key_is_replacement
    return out.sort_values("_key_row")[row_keys + [
        "history_train_type", "history_train_number", "history_train_key",
        "service_key_is_replacement",
    ]].reset_index(drop=True)


def _cv_stops(journey_ids: Iterable[str], *, path: Path = DATA / "stops.parquet") -> pd.DataFrame:
    """Read only non-target stop metadata for the requested journey IDs."""
    wanted = sorted({str(value) for value in journey_ids})
    if not wanted:
        return pd.DataFrame(columns=STOP_COLUMNS)
    table = pq.read_table(path, columns=STOP_COLUMNS, filters=[("journey_id", "in", wanted)])
    frame = table.to_pandas()
    return frame[frame["journey_id"].isin(wanted)].reset_index(drop=True)


def _cv_upstream_ids(journey_ids: Iterable[str], *, path: Path = DATA / "upstream_runs.parquet") -> pd.DataFrame:
    wanted = sorted({str(value) for value in journey_ids})
    if not wanted:
        return pd.DataFrame(columns=UPSTREAM_COLUMNS)
    table = pq.read_table(path, columns=UPSTREAM_COLUMNS, filters=[("journey_id", "in", wanted)])
    frame = table.to_pandas()
    return frame[frame["journey_id"].isin(wanted)].reset_index(drop=True)


def _read_change_references(ids: Iterable[str], *, change_dir: Path,
                            batch_size: int = 65_536) -> pd.DataFrame:
    """Read only replacement identity references, never event outcomes."""
    wanted = sorted({str(value) for value in ids if pd.notna(value)})
    columns = ["id", "snapshot_timestamp", "replaced_train_type", "replaced_train_number"]
    if not wanted:
        return pd.DataFrame(columns=columns)
    files = sorted(change_dir.glob("data-????-??.parquet"))
    if not files:
        raise FileNotFoundError(f"no monthly change snapshots in {change_dir}")
    wanted_array = pa.array(wanted, type=pa.string())
    parts = []
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size, use_threads=False):
            selected = batch.filter(pc.is_in(batch.column("id"), value_set=wanted_array))
            if selected.num_rows:
                parts.append(selected.to_pandas())
    if not parts:
        return pd.DataFrame(columns=columns)
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(["id", "snapshot_timestamp"], keep="last")
            .sort_values(["id", "snapshot_timestamp"]).reset_index(drop=True))


def _read_plan_ids(
    ids: Iterable[str],
    *,
    plan_dir: Path,
    files: Iterable[Path] | None = None,
    include_legacy: bool = True,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    wanted = sorted({str(value) for value in ids if pd.notna(value)})
    if not wanted:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in PLAN_COLUMNS + ["available_at_utc"]})
    source_files = sorted(files) if files is not None else sorted(plan_dir.glob("data-????-??.parquet"))
    if not source_files:
        raise FileNotFoundError(f"no parsed plan archive files in {plan_dir}")
    wanted_array = pa.array(wanted, type=pa.string())
    rows = []
    for path in source_files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=PLAN_COLUMNS, batch_size=batch_size, use_threads=False):
            filtered = batch.filter(pc.is_in(batch.column("id"), value_set=wanted_array))
            if filtered.num_rows:
                rows.append(filtered.to_pandas())
    if not rows:
        out = pd.DataFrame(columns=PLAN_COLUMNS)
    else:
        out = pd.concat(rows, ignore_index=True).drop_duplicates(["id", "snapshot_timestamp"], keep="last")
    out = prepare_plan_snapshots(out)
    if not include_legacy:
        out = out[pd.to_datetime(out["snapshot_timestamp"], utc=True).ge(EXACT_PLAN_REQUESTS_FROM_UTC)].copy()
    return out.sort_values(["id", "snapshot_timestamp"]).reset_index(drop=True)


def _find_replacement_origin_ids(
    replacement_keys: pd.DataFrame,
    *,
    plan_dir: Path,
) -> pd.DataFrame:
    """Find unambiguous scheduled-origin IDs by type, number and exact time.

    Only candidate origins needed by CV journeys are considered.  This lookup
    identifies an archive stop ID; all feature values still come from an
    eligible per-cutoff plan snapshot.
    """
    keys = replacement_keys.dropna(subset=["scheduled_train_type", "scheduled_train_number", "departure_planned_time"])
    if keys.empty:
        return pd.DataFrame(columns=["journey_id", "scheduled_origin_id", "candidate_id_count"])
    wanted_numbers = sorted(keys["scheduled_train_number"].astype(str).unique())
    wanted_types = sorted(keys["scheduled_train_type"].astype(str).unique())
    source_columns = ["id", "snapshot_timestamp", "station_name", "train_type", "train_number", "departure_planned_time"]
    matches: list[pd.DataFrame] = []
    for path in _month_files_for_rows(plan_dir, keys["departure_planned_time"]):
        # Station/type/number filters reduce the projected scan; exact planned
        # time matching below resolves train numbers that recur on other days.
        table = pq.read_table(path, columns=source_columns, filters=[
            ("station_name", "==", ORIGIN),
            ("train_type", "in", wanted_types),
            ("train_number", "in", wanted_numbers),
        ])
        if table.num_rows:
            matches.append(table.to_pandas())
    if not matches:
        return pd.DataFrame(columns=["journey_id", "scheduled_origin_id", "candidate_id_count"])
    found = pd.concat(matches, ignore_index=True).drop_duplicates(["id", "snapshot_timestamp"])
    found = found.merge(
        keys[["journey_id", "scheduled_train_type", "scheduled_train_number", "departure_planned_time"]].drop_duplicates(),
        left_on=["train_type", "train_number", "departure_planned_time"],
        right_on=["scheduled_train_type", "scheduled_train_number", "departure_planned_time"],
        how="inner",
    )
    count = found.groupby("journey_id")["id"].nunique().rename("candidate_id_count")
    single = found.groupby("journey_id")["id"].agg(lambda values: values.iloc[0] if values.nunique() == 1 else pd.NA).rename("scheduled_origin_id")
    out = pd.concat([single, count], axis=1).reset_index()
    out["candidate_id_count"] = out["candidate_id_count"].astype("Int64")
    return out


def build_replacement_context(stops_cv: pd.DataFrame, journeys_cv: pd.DataFrame,
                              scheduled_ids: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one context row per CV journey from its origin stop only."""
    origins = stops_cv[stops_cv["stop_index"].eq(0)].drop_duplicates("journey_id").copy()
    journeys = journeys_cv[["journey_id", "train_type", "train_number"]].drop_duplicates("journey_id")
    origins = origins.merge(journeys, on="journey_id", how="left", validate="one_to_one", suffixes=("_stop", "_journey"))
    origins["current_train_type"] = origins["train_type_journey"].combine_first(origins["train_type_stop"])
    origins["current_train_number"] = origins["train_number_journey"].combine_first(origins["train_number_stop"])
    is_replacement = origins["is_replacement_train"].fillna(False).astype(bool)
    origins["scheduled_train_type"] = origins["replaced_train_type"].where(is_replacement, origins["current_train_type"])
    origins["scheduled_train_number"] = origins["replaced_train_number"].where(is_replacement, origins["current_train_number"])
    origins["replacement_train_type"] = origins["current_train_type"].where(is_replacement, pd.NA)
    origins["replacement_train_number"] = origins["current_train_number"].where(is_replacement, pd.NA)
    origins["current_origin_id"] = origins["id"]
    origins["scheduled_origin_id"] = origins["id"].where(~is_replacement)
    if scheduled_ids is not None and not scheduled_ids.empty:
        origins = origins.merge(
            scheduled_ids[["journey_id", "scheduled_origin_id"]].rename(
                columns={"scheduled_origin_id": "candidate_scheduled_origin_id"}
            ),
            on="journey_id", how="left", validate="one_to_one",
        )
        origins["scheduled_origin_id"] = origins["scheduled_origin_id"].combine_first(
            origins["candidate_scheduled_origin_id"]
        )
    columns = [
        "journey_id", "current_origin_id", "scheduled_origin_id",
        "scheduled_train_type", "scheduled_train_number", "replacement_train_type",
        "replacement_train_number", "is_replacement_train", "departure_planned_time",
    ]
    out = origins.copy()
    if "is_replacement_train" not in out:
        out["is_replacement_train"] = False
    return out[columns].drop_duplicates("journey_id").reset_index(drop=True)


def _month_files_for_rows(plan_dir: Path, timestamps: pd.Series) -> list[Path]:
    months = pd.to_datetime(timestamps).dt.to_period("M")
    wanted = set()
    for month in months.dropna().unique():
        period = pd.Period(month, freq="M")
        wanted.add(str(period))
        wanted.add(str(period - 1))
        wanted.add(str(period + 1))
    return [path for path in sorted(plan_dir.glob("data-????-??.parquet"))
            if path.stem.removeprefix("data-") in wanted]


def build_timetable_split(
    split: str = "cv",
    *,
    plan_dir: Path = DEFAULT_PLAN_DIR,
    change_dir: Path = DEFAULT_CHANGE_DIR,
    output_dir: Path = ADVANCED,
    sample_journeys: int | None = None,
    registry_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build split-specific timetable inputs with a gate before holdout reads.

    Holdout paths are not touched until the committed registration gate passes.
    Both splits project only identifiers, timetable metadata and replacement
    references; no delay, cancellation or severity columns are read here.
    """
    if split not in {"cv", "holdout"}:
        raise ValueError("split must be 'cv' or 'holdout'")
    gate_passed = False
    if split == "holdout":
        from holdout_gate import committed_registration

        committed_registration(registry_path or DEFAULT_REGISTRY)
        gate_passed = True

    # For a gated holdout build, this is the first point that any holdout input
    # path is opened.  Explicit column projection omits all target fields.
    journey_columns = ["journey_id", "train_type", "train_number", "origin_departure_planned_local"]
    journeys = pd.read_parquet(DATA / f"journeys_{split}.parquet", columns=journey_columns)
    rows = pd.read_parquet(DATA / f"prediction_rows_{split}.parquet", columns=ROW_KEY)
    if sample_journeys is not None:
        if sample_journeys < 1:
            raise ValueError("sample_journeys must be positive")
        ordered = journeys.sort_values("origin_departure_planned_local")
        selected = set(ordered.tail(sample_journeys).journey_id)
        journeys = journeys[journeys.journey_id.isin(selected)].copy()
        rows = rows[rows.journey_id.isin(selected)].copy()
    entity_ids = set(journeys.journey_id.astype(str))
    stops = _cv_stops(entity_ids)
    upstream_runs = _cv_upstream_ids(entity_ids)
    origins = stops[stops.stop_index.eq(0)].copy()
    replacements = origins[origins.is_replacement_train.fillna(False)].copy()
    keys = replacements[["journey_id", "replaced_train_type", "replaced_train_number", "departure_planned_time"]].rename(columns={
        "replaced_train_type": "scheduled_train_type",
        "replaced_train_number": "scheduled_train_number",
    })
    candidate = _find_replacement_origin_ids(keys, plan_dir=plan_dir)
    context = build_replacement_context(stops, journeys, candidate)

    plan_ids = set(stops.id.dropna().astype(str)) | set(upstream_runs.id.dropna().astype(str))
    plan_ids |= set(context.scheduled_origin_id.dropna().astype(str))
    first_ids = []
    for value in context.scheduled_origin_id.dropna().astype(str):
        try:
            ride_id, _ = value.rsplit("-", 1)
            first_ids.append(f"{ride_id}-1")
        except ValueError:
            pass
    plan_ids |= set(first_ids)
    source_files = _month_files_for_rows(plan_dir, rows.predicted_at)
    snapshots = (_read_plan_ids(plan_ids, plan_dir=plan_dir, files=source_files,
                                include_legacy=True)
                 if source_files else
                 pd.DataFrame(columns=PLAN_COLUMNS + ["available_at_utc"]))

    change_snapshots = _read_change_references(replacements.id.astype(str), change_dir=change_dir)
    announcements = replacement_announcements_asof(rows, context, change_snapshots)
    timetable = build_timetable_features(rows, context, snapshots)

    output_dir.mkdir(parents=True, exist_ok=True)
    timetable.to_parquet(output_dir / f"timetable_{split}.parquet", index=False)
    snapshots.to_parquet(output_dir / f"plan_snapshots_{split}.parquet", index=False)
    announcements.to_parquet(output_dir / f"replacement_announcements_{split}.parquet", index=False)
    context.to_parquet(output_dir / f"replacement_context_{split}.parquet", index=False)
    counts = {}
    for name, group in timetable.groupby("prediction_time", dropna=False):
        counts[str(name)] = {
            "rows": int(len(group)),
            **{column: {"known": int(group[column].notna().sum()),
                        "coverage": float(group[column].notna().mean())}
               for column in TIMETABLE_COLUMNS},
        }
    archive_files = sorted(plan_dir.glob("data-????-??.parquet"))
    audit = {
        "source": {
            "url": ARCHIVE_URL,
            "upstream_repo": UPSTREAM_REPO_URL,
            "revision": ARCHIVE_REVISION,
            "source_description": "One row per train stop and parsed plan API snapshot; ordered planned paths.",
            "timestamp_semantics": (
                "The dataset README calls snapshot_timestamp the API request time. The upstream pre-refactor fetcher (`fetch_data.py` at the parent of the refactor) names plan XML `{eva}_plan_{hour}.xml`, where hour is the requested timetable hour (current hour through next five), while fchg filenames use current fetch hour. Legacy whole-hour plan stamps therefore are target-hour labels, not request-time evidence. The post-refactor client records datetime.now() after the response and the parser preserves that timestamp."
            ),
            "exact_request_from_utc": str(EXACT_PLAN_REQUESTS_FROM_UTC),
            "timestamp_evidence": {
                "dataset_readme": "https://github.com/piebro/deutsche-bahn-data#monthly-parsed-data",
                "legacy_fetcher": LEGACY_FETCH_URL,
                "post_refactor_fetcher": "https://github.com/piebro/deutsche-bahn-data/blob/main/scripts/db_data_fetcher.py",
                "post_refactor_parser": "https://github.com/piebro/deutsche-bahn-data/blob/main/scripts/parse_monthly_xml.py",
                "first_non_hour_archive_stamp_utc": str(EXACT_PLAN_REQUESTS_FROM_UTC),
                "legacy_plan_maximum_fetch_lag": None,
            },
            "legacy_plan_policy": "earliest legacy snapshot per stop is a static schedule, regardless of fetch time, by user instruction",
            "change_source_bound": "+45 minutes remains specific to legacy change snapshots only",
            "feature_policy": "Use the earliest legacy plan per stop as a static schedule; exact-request updates remain bounded by predicted_at in UTC.",
        },
        "split": split,
        "cv_only": split == "cv",
        "holdout_gate_passed": gate_passed,
        "holdout_inputs_projected_without_targets": split == "holdout",
        "journeys": int(journeys.journey_id.nunique()),
        "prediction_rows": int(len(timetable)),
        "selected_sample_journeys": sample_journeys,
        "pinned_archive_file_count": len(archive_files),
        "pinned_archive_total_bytes": sum(path.stat().st_size for path in archive_files),
        "plan_snapshot_rows_saved": int(len(snapshots)),
        "plan_snapshot_ids_saved": int(snapshots.id.nunique()) if len(snapshots) else 0,
        "plan_snapshot_exact_rows_saved": int(pd.to_datetime(snapshots.snapshot_timestamp, utc=True).ge(EXACT_PLAN_REQUESTS_FROM_UTC).sum()) if len(snapshots) else 0,
        "replacement_journeys": int(len(replacements)),
        "replacement_candidate_resolution": {
            "eligible_replacement_journeys": int(keys.journey_id.nunique()),
            "unique_candidate_id": int(candidate.candidate_id_count.eq(1).sum()) if len(candidate) else 0,
            "ambiguous_candidate_id": int(candidate.candidate_id_count.gt(1).sum()) if len(candidate) else 0,
            "unmatched_candidate_id": int(keys.journey_id.nunique() - candidate.journey_id.nunique()),
            "candidate_id_counts": (
                {str(key): int(value) for key, value in
                 candidate.candidate_id_count.value_counts(dropna=False).items()}
                if len(candidate) else {}
            ),
        },
        "replacement_announcements_visible": int(announcements.replacement_announced.fillna(False).sum()),
        "replacement_journeys_with_announcement": int(
            announcements.loc[announcements.replacement_announced.fillna(False), "journey_id"].nunique()
        ),
        "coverage_by_prediction_time": counts,
        "source_files_read": [path.name for path in source_files],
        "source_files_read_bytes": sum(path.stat().st_size for path in source_files),
    }
    audit_path = output_dir / f"timetable_{split}_audit.json"
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, audit_path)
    return timetable, audit


def build_cv_timetable(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Compatibility wrapper for the default CV build."""
    return build_timetable_split("cv", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--change-dir", type=Path, default=DEFAULT_CHANGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=ADVANCED)
    parser.add_argument("--sample-journeys", type=int)
    parser.add_argument("--split", choices=("cv", "holdout"), default="cv")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    if args.sample_journeys is not None and args.output_dir.resolve() == ADVANCED.resolve():
        parser.error("--sample-journeys requires a separate --output-dir")
    _, audit = build_timetable_split(
        split=args.split,
        plan_dir=args.plan_dir, change_dir=args.change_dir,
        output_dir=args.output_dir, sample_journeys=args.sample_journeys,
        registry_path=args.registry,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
