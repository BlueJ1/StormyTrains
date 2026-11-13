"""Read the live-update snapshots (data/monthly_processed_data_change).

The source keeps one row per stop and snapshot. The monthly processing uses, for
every stop, the latest snapshot that carries any change (see
create_monthly_data_release.py in github.com/piebro/deutsche-bahn-data);
`qualifies` mirrors that filter. Used by build_dataset.py and
check_change_data.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CHANGE_DIR = ROOT / "data" / "monthly_processed_data_change"
STATIONS = ["Frankfurt (Main) Hbf", "München Hbf"]
# First exact request stamp in the change archive (UTC). Earlier stamps are
# block labels, and the request can be up to 45 minutes later. This is the one
# definition; advanced_modeling/history_features.py imports it.
EXACT_REQUESTS_FROM_UTC = pd.Timestamp("2025-11-03 03:27:54.653106")
LEGACY_REQUEST_DELAY = pd.Timedelta(minutes=45)


def request_upper_bound(stamps: pd.Series) -> pd.Series:
    """The latest time at which each snapshot can have been requested, in UTC,
    naive or time-zone aware like `stamps`: the stamp itself, or 45 minutes
    later for a block label before the exact request stamps began."""
    exact_from = EXACT_REQUESTS_FROM_UTC if stamps.dt.tz is None else EXACT_REQUESTS_FROM_UTC.tz_localize("UTC")
    block_label = stamps.dt.minute.eq(0) & stamps.dt.second.eq(0) & stamps.dt.microsecond.eq(0)
    legacy = block_label & stamps.lt(exact_from)
    return stamps.where(~legacy, stamps + LEGACY_REQUEST_DELAY)
COLUMNS = ["id", "snapshot_timestamp", "train_label_type", "replaced_train_type", "replaced_train_number",
           "replaced_train_label_type", "replaced_train_owner", "replaced_train_filter_flags"] + [
    f"{side}_{field}" for side in ("arrival", "departure")
    for field in ("planned_time", "change_time", "cancellation_time", "planned_status", "change_status")]


def read_snapshots(ids: set[str]) -> pd.DataFrame:
    parts = []
    for path in sorted(CHANGE_DIR.glob("data-????-??.parquet")):
        frame = pq.read_table(path, columns=COLUMNS, filters=[("station_name", "in", STATIONS)]).to_pandas()
        parts.append(frame[frame.id.isin(ids)])
    return with_request_times(pd.concat(parts, ignore_index=True).drop_duplicates(["id", "snapshot_timestamp"]))


def with_request_times(snaps: pd.DataFrame) -> pd.DataFrame:
    """Add `request_local`, `block_label` and `known_local`; sort by stop and stamp."""
    snaps = snaps.copy()
    # snapshot_timestamp is UTC (the request time on the fetching machine), while
    # all train times are German local time. Until early November 2025 it is the
    # start of the 6-hour fetch block, and the request itself can be up to about
    # 45 minutes later; afterwards it is the exact request time.
    stamp = snaps.snapshot_timestamp
    snaps["request_local"] = (stamp.dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
                              .dt.tz_localize(None))
    snaps["block_label"] = stamp.dt.minute.eq(0) & stamp.dt.second.eq(0) & stamp.dt.microsecond.eq(0)
    # The latest local time at which the snapshot can have been requested.
    snaps["known_local"] = (request_upper_bound(stamp).dt.tz_localize("UTC")
                            .dt.tz_convert("Europe/Berlin").dt.tz_localize(None))
    return snaps.sort_values(["id", "snapshot_timestamp"])


def qualifies(s: pd.DataFrame) -> pd.Series:
    """The rows the monthly processing considers (any change field set)."""
    return (s.arrival_change_time.notna() | s.departure_change_time.notna()
            | s.arrival_cancellation_time.notna() | s.departure_cancellation_time.notna()
            | s.arrival_planned_status.eq("a") | s.departure_planned_status.eq("a")
            | s.arrival_change_status.notna() | s.departure_change_status.notna()
            | s.train_label_type.eq("e") | s.replaced_train_type.notna() | s.replaced_train_number.notna()
            | s.replaced_train_label_type.notna() | s.replaced_train_owner.notna()
            | s.replaced_train_filter_flags.notna())


def event_summary(snaps: pd.DataFrame, ids: pd.Series, side: str) -> pd.DataFrame:
    """Per stop id in `ids`, what the snapshots say about its `side` ("arrival" or
    "departure"); indexed like `ids`. The `latest_*` columns come from the latest
    qualifying snapshot, the one the monthly processing uses."""
    s = snaps[snaps.id.isin(ids)]
    ct, clt, cs = f"{side}_change_time", f"{side}_cancellation_time", f"{side}_change_status"
    g = s.groupby("id")
    latest = s[qualifies(s)].groupby("id").tail(1).set_index("id")
    with_ct_rows = s[s[ct].notna()]
    with_ct = with_ct_rows.groupby("id").tail(1).set_index("id")
    before_last = with_ct_rows.groupby("id").nth(-2).set_index("id")
    canceled_rows = s[s[clt].notna() | s[cs].eq("c")]
    out = pd.DataFrame({
        "snapshots": g.size(),
        "snapshots_with_time": s[ct].notna().groupby(s.id).sum(),
        "latest_request": latest.request_local,
        "latest_block_label": latest.block_label,
        "latest_time": latest[ct],
        "last_time": with_ct[ct],
        "earlier_time": before_last[ct],
        "latest_cancellation_time": latest[clt],
        "latest_status": latest[cs],
        "first_cancellation_time": canceled_rows.groupby("id")[clt].min(),
    })
    return out.reindex(ids.values).set_axis(ids.index)


def cancellation_events(snaps: pd.DataFrame, ids, side: str) -> pd.DataFrame:
    """Every change of the cancellation state of one `side` ("arrival" or
    "departure") of the stops in `ids`: one row per change with the stop `id`,
    the local `time` from which it holds and `in_force`.

    A row is cancelled if it has an issue time or status "c" (as in
    `event_summary`), withdrawn if its status is "p".

    - A cancellation holds from its issue time (`cancellation_time`), or from
      the time its snapshot can have been requested (`known_local`) if it has
      no issue time.
    - A withdrawal holds from the time the first snapshot that shows it can
      have been requested. The snapshots carry no time for the withdrawal
      itself. A later snapshot of the stop that shows no cancellation at all
      ends it in the same way.
    - A cancellation that is shown again after a withdrawal, with an issue time
      from before that withdrawal, holds from the time its snapshot can have
      been requested.
    - A withdrawn row starts a cancellation only if it is the first sign of one
      at the stop (cancelled from its issue time until the withdrawal shows); a
      withdrawn row with a new issue time later on does not cancel the train
      again.
    """
    issue_col, status_col = f"{side}_cancellation_time", f"{side}_change_status"
    s = snaps[snaps.id.isin(set(ids))].sort_values(["id", "snapshot_timestamp"])
    rows = []
    for stop_id, group in s.groupby("id", sort=False):
        seen, withdrawn_at, in_force, started = set(), None, False, False
        for issue, status, known in zip(group[issue_col], group[status_col], group["known_local"]):
            withdrawn = status == "p"
            cancelled = not withdrawn and (pd.notna(issue) or status == "c")
            if pd.notna(issue) and issue not in seen:
                seen.add(issue)
                if not withdrawn or not started:
                    rows.append((stop_id, issue if withdrawn_at is None or issue > withdrawn_at else known, True))
                    in_force = started = True
            elif cancelled and not in_force:
                rows.append((stop_id, known, True))
                in_force = started = True
            if in_force and not cancelled:
                rows.append((stop_id, known, False))
                in_force, withdrawn_at = False, known
    events = pd.DataFrame(rows, columns=["id", "time", "in_force"])
    events["time"] = pd.to_datetime(events["time"]).astype("datetime64[ns]")
    events["in_force"] = events.in_force.astype(bool)
    events["order"] = range(len(events))
    return events


def cancellation_in_force(events: pd.DataFrame, ids: pd.Series, when: pd.Series) -> pd.Series:
    """Whether a cancellation of the stops `ids` is in force at the local times
    `when`; both are indexed alike. The state is that of the latest change at or
    before `when`; a stop without any change is not cancelled."""
    query = pd.DataFrame({"id": ids, "when": when.astype("datetime64[ns]")}).reset_index(names="_key")
    usable = query.dropna(subset=["id", "when"]).sort_values("when")
    if events.empty or usable.empty:
        return pd.Series(False, index=ids.index)
    state = pd.merge_asof(
        usable, events.sort_values(["time", "order"]), left_on="when", right_on="time", by="id",
        direction="backward", allow_exact_matches=True)
    in_force = state.set_index("_key").in_force.eq(True)
    return in_force.reindex(ids.index, fill_value=False)
