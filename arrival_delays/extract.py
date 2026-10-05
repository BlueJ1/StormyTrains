"""Collect ICE/IC arrival delays for density estimation.

Writes two delay-only datasets (`delay_min`, `train_type`):

- `stop_arrivals.parquet`: every recorded arrival of an ICE/IC run.
- `terminus_arrivals.parquet`: one arrival per dated run, at its last recorded
  stop (highest `train_line_station_num` in the source). The source does not
  record every station, so for runs ending abroad or at an unrecorded station
  this is the last recorded stop, not the true terminus.

The delay is `arrival_change_time - arrival_planned_time` in minutes; early
arrivals are negative. The source `delay_in_min` is not used because it is the
departure delay where the train continues. Canceled arrivals are excluded, and
only stations in `data/stations_present_in_all_months.csv` are kept. The
terminus is chosen before these filters, so a run whose last recorded arrival
is canceled or outside the panel has no terminus row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "monthly_processed_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
PANEL_FILE = ROOT / "data" / "stations_present_in_all_months.csv"
TRAIN_TYPES = ["ICE", "IC"]
COLUMNS = [
    "id", "station_name", "train_type", "train_line_station_num",
    "arrival_planned_time", "arrival_change_time", "arrival_is_canceled",
]


def read_stops(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for path in paths:
        table = pq.read_table(path, columns=COLUMNS,
                              filters=[("train_type", "in", TRAIN_TYPES)])
        parts.append(table.to_pandas())
    return pd.concat(parts, ignore_index=True)


def delay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    delay = (frame.arrival_change_time - frame.arrival_planned_time).dt.total_seconds() / 60
    return pd.DataFrame({
        "delay_min": delay.astype("int16"),
        "train_type": pd.Categorical(frame.train_type, categories=TRAIN_TYPES),
    })


def main() -> None:
    paths = sorted(INPUT_DIR.glob("data-????-??.parquet"))
    if not paths:
        raise FileNotFoundError(f"No monthly Parquet files in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = set(pd.read_csv(PANEL_FILE).station_name)

    stops = read_stops(paths)
    audit = {"source_files": len(paths), "source_rows": len(stops)}
    # Runs crossing a month boundary can appear in two files.
    audit["duplicate_ids_dropped"] = int(stops.duplicated(subset=["id"]).sum())
    stops = stops.drop_duplicates(subset=["id"], keep="first")
    stops["run_id"] = stops.id.str.rsplit("-", n=1).str[0]

    # The terminus is chosen from all recorded stops, before any filter.
    terminus = (stops.sort_values(["run_id", "train_line_station_num"])
                .drop_duplicates(subset=["run_id"], keep="last"))
    audit["runs"] = len(terminus)

    def keep(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        no_arrival = frame.arrival_planned_time.isna()
        no_actual = ~no_arrival & frame.arrival_change_time.isna()
        canceled = ~no_arrival & ~no_actual & frame.arrival_is_canceled
        outside = ~no_arrival & ~no_actual & ~canceled & ~frame.station_name.isin(panel)
        audit.update({
            f"{prefix}_rows": len(frame),
            f"{prefix}_dropped_no_planned_arrival": int(no_arrival.sum()),
            f"{prefix}_dropped_no_actual_arrival": int(no_actual.sum()),
            f"{prefix}_dropped_canceled": int(canceled.sum()),
            f"{prefix}_dropped_outside_panel": int(outside.sum()),
        })
        out = delay_frame(frame[~(no_arrival | no_actual | canceled | outside)])
        audit[f"{prefix}_kept"] = len(out)
        audit[f"{prefix}_kept_by_type"] = out.train_type.value_counts().to_dict()
        return out

    stop_out = keep(stops, "stop")
    terminus_out = keep(terminus, "terminus")
    stop_out.to_parquet(OUTPUT_DIR / "stop_arrivals.parquet", index=False)
    terminus_out.to_parquet(OUTPUT_DIR / "terminus_arrivals.parquet", index=False)
    (Path(__file__).resolve().parent / "extraction_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
