"""Reconstruct past Frankfurt–München outcomes as known at each cutoff."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  shared modelling code and dataset builders
from history_features import (CHANGE_DIR, DATA_DIR, PLAN_DIR, PLAN_EXACT_UTC_FROM,
                              RECENT_RELIABILITY_COLUMNS, ROW_KEY, _cancellation_in_force,
                              _journey_run_start_utc, _read_snapshot_ids_in_months, cutoff_instants,
                              latest_plan_asof, latest_snapshot_asof, local_clock_to_utc,
                              prepare_plan_snapshots)


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
    Replacement outcomes enter service history only after a visible link
    announcement. Their independently observed corridor outcomes stay eligible.
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
    default_flags = pd.Series(False, index=states.index)
    replacement = states.get("is_replacement", default_flags).fillna(False).astype(bool)
    announced = states.get("replacement_announced", default_flags).fillna(False).astype(bool)
    states["service_outcome_known"] = states["outcome_known"] & (~replacement | announced)
    state_groups = {key: group.copy() for key, group in states.groupby(ROW_KEY, sort=False)}
    rate_columns = RECENT_RELIABILITY_COLUMNS
    records = []
    periods = [
        ("service", "7d", pd.Timedelta(days=7)),
        ("service", "28d", pd.Timedelta(days=28)),
        ("corridor", "24h", pd.Timedelta(hours=24)),
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
                    eligible_base["service_outcome_known"]
                    & eligible_base["origin_departure_utc"].lt(cutoff)
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
    plans = (plan_snapshots if "available_at_utc" in plan_snapshots
             else prepare_plan_snapshots(plan_snapshots))
    asof_plan = latest_plan_asof(queries, plans)
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


def build_reliability_file(
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

    Both target splits use the same target-free stop roster. Candidate pairs are bounded to 32 days,
    and only same-service candidates plus the corridor's prior 24-hour window
    are materialized. The shared plan policy treats the earliest plan snapshot
    per stop as a static schedule in both archive periods.
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
    announcements = pd.read_parquet(announcements_path)
    from scheduled_timetable import (build_replacement_context, replacement_announcements_asof,
                                     service_keys_asof)
    target_keys = service_keys_asof(rows, pd.DataFrame(), context, announcements)
    rows = rows.merge(target_keys[[*ROW_KEY, "history_train_type", "history_train_number"]],
                      on=ROW_KEY, how="left", validate="one_to_one")
    rows["_cutoff_utc"] = cutoff_instants(rows["predicted_at"])

    stops = pd.read_parquet(data_dir / "stops.parquet", columns=[
        "id", "journey_id", "station_name", "train_line_station_num",
        "stop_index", "train_type", "train_number", "is_replacement_train",
        "replaced_train_type", "replaced_train_number", "departure_planned_time",
    ])
    stops = stops.loc[stops["station_name"].isin(
        ["Frankfurt (Main) Hbf", "München Hbf"]
    )].copy()
    stops = stops.sort_values(["journey_id", "train_line_station_num"])
    origin = (stops.loc[stops["station_name"].eq("Frankfurt (Main) Hbf")]
              .drop_duplicates("journey_id").rename(columns={"id": "origin_id"}))
    candidate_context = build_replacement_context(stops, origin)
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
    processed_months: list[str] = []
    target_rows = rows.copy()
    target_rows["_cutoff_month"] = target_rows["_cutoff_utc"].dt.tz_localize(None).dt.to_period("M")
    for month, month_targets in target_rows.groupby("_cutoff_month", sort=True):
        print(f"reliability {split} {month}: {len(month_targets)} target rows", flush=True)
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
        source_month_end = upper_bound.tz_convert("Europe/Berlin").tz_localize(None).to_period("M") + 1
        source_months = list(pd.period_range(source_month_start, source_month_end, freq="M"))
        processed_months.extend(str(value) for value in source_months)
        event_ids = set(month_candidates["origin_id"].astype(str)) | set(
            month_candidates["destination_id"].astype(str)
        )
        plans = prepare_plan_snapshots(_read_snapshot_ids_in_months(
            event_ids, kind="plan", months=source_months,
            change_dir=change_dir, plan_dir=plan_dir,
        ))
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
                candidate_announcements[ROW_KEY + ["candidate_journey_id", "is_replacement",
                                                   "replacement_announced"]],
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
        "plan_snapshot_policy": (
            "earliest snapshot per stop treated as a static schedule available at "
            "1900-01-01 UTC, in both archive periods"
        ),
        "legacy_plan_policy": "earliest snapshot per stop is static from 1900-01-01 UTC",
        "candidate_policy": "same-service candidates within 28d, corridor outcomes within 24h; 32d service and 48h corridor run-ID prefilters",
        "candidate_source": "shared target-free stop roster for both splits",
        "candidate_journeys": int(len(candidates)),
        "prediction_rows": int(len(result)),
        "rows_with_legacy_static_plan_policy": int(
            rows["_cutoff_utc"].lt(PLAN_EXACT_UTC_FROM).sum()
        ),
        "source_months_read": sorted(set(processed_months)),
        "coverage_by_prediction_time": coverage,
    }
    audit_path = output.with_name(output.stem + "_audit.json")
    audit_tmp = audit_path.with_name(audit_path.stem + ".tmp" + audit_path.suffix)
    audit_tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_tmp.replace(audit_path)
    return output
