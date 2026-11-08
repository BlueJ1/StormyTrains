"""Build compact, cutoff-specific sequences from the audited upstream event table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  shared modelling code and dataset builders
from data import DATA, ROW_KEY
from holdout_gate import REGISTRY, committed_registration

SEQUENCE_COLUMN = "upstream_sequence"
SEQUENCE_LENGTH = 24
SEQUENCE_CHANNELS = 2
FRANKFURT = "Frankfurt (Main) Hbf"


def sequence_inputs(rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Retain the latest occurred event per upstream stop and the last 24 stops."""
    visible = (events["event_report_visible"] & events["event_occurred_by_cutoff"]
               & ~events["event_cancelled"] & events["event_delay_min"].notna()
               & events["station_sequence"].notna()
               & events["station_name"].ne(FRANKFURT))
    events = events.loc[visible].copy()
    sequence_key = [*ROW_KEY, "station_sequence"]
    events = (events.sort_values([*sequence_key, "reported_time", "event_kind"])
              .drop_duplicates(sequence_key, keep="last"))
    report_age = (events["cutoff_utc"] - events["reported_time_utc"]).dt.total_seconds() / 60
    events["age_hours"] = report_age.clip(lower=0) / 60
    events["delay_half_hours"] = events["event_delay_min"] / 30
    sequences = {}
    for key, frame in events.groupby(ROW_KEY, observed=True, sort=False):
        channels = frame[["delay_half_hours", "age_hours"]].tail(SEQUENCE_LENGTH)
        sequences[key] = channels.to_numpy(dtype=np.float32).reshape(-1).tolist()
    result = rows[ROW_KEY].copy()
    result[SEQUENCE_COLUMN] = [sequences.get(key, []) for key in result.itertuples(index=False, name=None)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=DATA / "advanced")
    parser.add_argument("--split", choices=("cv", "holdout"), default="cv")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    if args.split == "holdout":
        committed_registration(args.registry)
    rows = pd.read_parquet(DATA / f"prediction_rows_{args.split}.parquet", columns=ROW_KEY)
    events = pd.read_parquet(args.inputs_dir / f"upstream_events_{args.split}.parquet")
    result = sequence_inputs(rows, events)
    result.to_parquet(args.inputs_dir / f"sequence_{args.split}.parquet", index=False, compression="zstd")
    print(result.groupby("prediction_time", observed=True)[SEQUENCE_COLUMN].agg(
        lambda column: column.map(bool).sum()).to_string())


if __name__ == "__main__":
    main()
