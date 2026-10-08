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
COLUMNS = ["id", "snapshot_timestamp", "train_label_type", "replaced_train_type", "replaced_train_number",
           "replaced_train_label_type", "replaced_train_owner", "replaced_train_filter_flags"] + [
    f"{side}_{field}" for side in ("arrival", "departure")
    for field in ("planned_time", "change_time", "cancellation_time", "planned_status", "change_status")]


def read_snapshots(ids: set[str]) -> pd.DataFrame:
    parts = []
    for path in sorted(CHANGE_DIR.glob("data-????-??.parquet")):
        frame = pq.read_table(path, columns=COLUMNS, filters=[("station_name", "in", STATIONS)]).to_pandas()
        parts.append(frame[frame.id.isin(ids)])
    snaps = pd.concat(parts, ignore_index=True).drop_duplicates(["id", "snapshot_timestamp"])
    # snapshot_timestamp is UTC (the request time on the fetching machine), while
    # all train times are German local time. Until early November 2025 it is the
    # start of the 6-hour fetch block, and the request itself can be up to about
    # 45 minutes later; afterwards it is the exact request time.
    stamp = snaps.snapshot_timestamp
    snaps["request_local"] = (stamp.dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
                              .dt.tz_localize(None))
    snaps["block_label"] = stamp.dt.minute.eq(0) & stamp.dt.second.eq(0) & stamp.dt.microsecond.eq(0)
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
