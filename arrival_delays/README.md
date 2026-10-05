# ICE/IC arrival delays

Two delay-only datasets for density estimation, covering every monthly file (July 2024 to August 2026). Each has two columns: `delay_min` (int16; early arrivals are negative) and `train_type` (`ICE` or `IC`).

| File | Rows | What one row is |
|---|---|---|
| `data/stop_arrivals.parquet` | 4,061,115 (ICE 3,407,523; IC 653,592) | every recorded arrival of an ICE/IC run |
| `data/terminus_arrivals.parquet` | 558,424 (ICE 430,324; IC 128,100) | the arrival at each dated run's last recorded stop |

## Definitions

- **Delay:** `arrival_change_time − arrival_planned_time`, in minutes. The source `delay_in_min` is not used, because it is the departure delay where the train continues.
- **Excluded:**
  - canceled arrivals;
  - stops with no planned arrival (first stops);
  - stations not in `data/stations_present_in_all_months.csv`.
- **Terminus:** the stop with the highest `train_line_station_num` recorded for the dated run, chosen before the filters above.
  - The source does not record every station. For runs ending abroad or at an unrecorded station, the terminus is therefore the last recorded stop, not the real one. In March 2025 this was about 30% of runs.
  - If the terminus arrival is canceled or outside the panel, the run has no terminus row. There is no fallback to an earlier stop.

Counts for every exclusion are in `extraction_audit.json`.

## Rebuild

```bash
.venv/bin/python arrival_delays/extract.py   # about 15 s
```
