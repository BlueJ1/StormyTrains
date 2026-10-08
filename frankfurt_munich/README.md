# Frankfurt (Main) Hbf → München Hbf dataset

The modelling dataset for the chosen origin–destination pair, railway data only.

- **Journeys and targets** follow `corridor_analysis/extract.py`. `build_dataset.py` reuses its functions for this pair alone.
- **On top of that:**
  - each departure is kept once;
  - `current_delay` is added at each prediction time;
  - the time a cancellation was announced is added, so that no prediction is made once a cancellation is known;
  - the data are split into a cross-validation period and a one-year hold-out set.
- **Source:** the 27 monthly files from July 2024 to September 2026 and the matching live-update snapshots, as rebuilt by the source on 6–8 October 2026 and downloaded on 8 October 2026 (checked against the published SHA-256 sums).

**The hold-out year is not to be looked at yet.** Outcome statistics below (classes, delays, cancellations) are for the CV period only. For the hold-out year we only check data quality: counts, coverage, missing values and where the labels come from.

## Files

| File | Rows | Contents |
|---|---|---|
| `data/journeys_cv.parquet` | 12,084 | Planned Frankfurt departure from 1 July 2024 to 30 September 2025, for cross-validation |
| `data/journeys_holdout.parquet` | 9,940 | 1 October 2025 to 30 September 2026, the hold-out year |
| `data/prediction_rows_cv.parquet` | 59,562 | One row per CV journey and prediction time, without known cancellations (see below). A model for one prediction time uses the rows with that `prediction_time` |
| `data/prediction_rows_holdout.parquet` | 48,544 | The same for the hold-out year |
| `data/stops.parquet` | 136,606 | Recorded stops from Frankfurt to München of the kept journeys |
| `data/upstream_runs.parquet` | 116,782 | Every recorded stop of each run up to and including Frankfurt, used for `current_delay` |
| `data/change_check.parquet` | 22,024 | Per journey: where the München arrival time comes from, and when a cancellation was issued (see below) |
| `cv_folds.json` | 5 | Fold boundaries and journey counts |
| `build_audit.json`, `audit.json`, `change_check.json` | | Counts behind every number in this README |

## Rebuild

```bash
.venv/bin/python frankfurt_munich/build_dataset.py      # about 1 min; needs both data folders
.venv/bin/python frankfurt_munich/audit.py
.venv/bin/python frankfurt_munich/check_change_data.py
```

Both `data/monthly_processed_data` and `data/monthly_processed_data_change` from the Hugging Face dataset are needed.

| Script | Purpose |
|---|---|
| `build_dataset.py` | Builds every file in `data/`, `cv_folds.json` and `build_audit.json` |
| `folds.py` | Hold-out period and cross-validation folds |
| `snapshots.py` | Reads the live-update snapshots; used by the two scripts above and below |
| `audit.py` | Data-quality audit, `audit.json` |
| `check_change_data.py` | Where each label comes from and when cancellations were announced, `change_check.json` and `data/change_check.parquet` |

## Hold-out set and cross-validation folds

The hold-out set is one full year, October 2025 to September 2026. The split is by planned Frankfurt departure; the definition is in `folds.py`.

The CV period, July 2024 to September 2025 (15 months), has 5 folds:
- The training window always starts in July 2024 and grows by 1/12 of the period per fold.
- The validation window is the following 2/12 of the period (about 76 days), so consecutive validation windows overlap by 1/12.
- Fractions are of calendar time, cut at midnight.

| Fold | Training | Validation | Training journeys | Validation journeys |
|---|---|---|---|---|
| 1 | 1 Jul 2024 – 13 Feb 2025 | 14 Feb – 30 Apr 2025 | 6,345 | 1,594 |
| 2 | to 23 Mar 2025 | 24 Mar – 7 Jun 2025 | 7,068 | 1,766 |
| 3 | to 30 Apr 2025 | 1 May – 15 Jul 2025 | 7,939 | 2,044 |
| 4 | to 7 Jun 2025 | 8 Jun – 22 Aug 2025 | 8,834 | 2,135 |
| 5 | to 15 Jul 2025 | 16 Jul – 30 Sep 2025 | 9,983 | 2,101 |

## Evaluation

- **Choosing and tuning models on the CV folds:** multi-class log loss.
- **Reported at the end:** macro-F1 as well.

## Targets

- `arrival_delay_min`: arrival delay at München Hbf in minutes, computed from arrival times. Missing for cancelled journeys.
- `journey_canceled`: the Frankfurt departure or the München arrival is cancelled. A cancellation only at an intermediate stop does not count.
- `severity_class`: `normal_minor` (under 20 min), `moderate` (20–59 min; exactly 20 belongs here), `severe` (60 min or more), `canceled`.
- `severe_or_canceled`: the binary version.

CV class shares: 64.4% under 20 min, 25.0% 20–59 min, 5.9% 60+ min, 4.7% cancelled.

## One journey per departure

Some departures appear as two or more journeys under different train numbers, with the same planned Frankfurt departure and München arrival. There were 1,341 such groups, covering 2,765 journeys; 1,066 of these departures are in the hold-out year and 275 in CV.

- **The usual case:** the scheduled train is marked cancelled, and a replacement train with another number ran at the same times.
- **Rule (passenger's view):** one journey is kept per departure: one that completed if there is one, then one not flagged as a replacement, then the lowest train number. 1,424 journeys were removed.
  - In 265 of the 298 groups where one train was cancelled and another ran, the kept journey is the replacement.
  - One cancelled train was replaced at different times (2849 for 699, 43 min earlier); it stays cancelled.
- **`shared_departure`** is the number of journeys that were listed for the departure.

## current_delay

`current_delay_24h`, `_6h`, `_1h`, `_20min` and `_0min` are taken at 24 h, 6 h, 1 h, 20 min and 0 min before the planned Frankfurt departure.

The value is the latest delay known at that moment, from the train's own run up to Frankfurt:

- The latest non-cancelled arrival or departure that has happened by then.
- If that event is an arrival (the train is standing at a stop and has not left yet), the delay accumulated by then. That is the larger of the arrival delay and the minutes since the planned departure from that stop.
- Missing if nothing has happened yet.

Only stations in `data/stations_present_in_all_months.csv` are used, so the coverage expansion on 3 November 2025 does not add upstream stops in the hold-out year.

Missing values in the CV period (12,084 journeys):

| Prediction time | Missing | Train starts in Frankfurt | Run not started yet | No usable upstream report | Cancelled, missing vs known |
|---|---|---|---|---|---|
| 24 h | 12,084 (100%) | 1,463 | 10,560 | 61 | 4.7% vs – |
| 6 h | 11,249 (93%) | 1,463 | 9,725 | 61 | 4.9% vs 3.0% |
| 1 h | 1,723 (14%) | 1,463 | 199 | 61 | 10.4% vs 3.8% |
| 20 min | 1,675 (14%) | 1,463 | 151 | 61 | 10.1% vs 3.9% |
| 0 min | 1,217 (10%) | 1,059 | 97 | 61 | 12.4% vs 3.9% |

At 0 min, the 1,059 trains that start in Frankfurt and have no value had not left by their scheduled time.

**Decided treatment.** Missing values are not set to plain 0: a 0 means "on time", while a missing value goes with about three times as many cancellations.
- 24 h: not used (always missing). 6 h: probably not used (93% missing).
- From 1 h on: 0 plus a missing indicator for linear models; left missing for tree models.
- The files keep the missing values; the treatment is applied when a model is fitted.

## Where the labels come from

The monthly files keep, for each stop, the values from the latest live-update snapshot that carries any change. An arrival time without a change would fall back to the planned time, and so to a 0-min delay. `check_change_data.py` traces each München arrival back to the snapshots in `data/monthly_processed_data_change`.

**Coverage.** The snapshots cover every journey arriving before 30 September 2026: 21,993 of 22,024 journeys, 20,774 of them completed. The last 31 are left out of the check only because the last snapshot day is too close.

| Source of the München arrival time | CV | Hold-out |
|---|---|---|
| Snapshot taken after the train arrived (an actual time) | 11,502 | 9,133 |
| Snapshot taken before arrival (a forecast) | 6 | 126 |
| Snapshot within 45 min of arrival (can't tell which) | 4 | 2 |
| No snapshot at all, so the planned time was used | 0 | 1 |

Our files reproduce the snapshot values exactly for every completed journey.

- **Forecast labels** are rare in CV (10, all June 2025) and more common from May 2026: 17 in May, 7 in June, 39 in July, 25 in August, 14 in September. The median forecast was taken about 45 min before arrival in the hold-out year (about 100 min in CV).
- **One journey without a snapshot** (7 January 2026) got the planned arrival time, so a 0-min delay.

**Exactly 0 min arrivals.**
- 1,555 of the 1,571 covered 0-min arrivals are DB's own time, read after arrival; 15 are forecasts and 1 is the journey without a snapshot.
- In the CV period they are no more common than −1 or +1 min (665, 664, 576). The delay at the stop before is normal (median 1 min, 0.6% at 10+ min).
- In the hold-out year they are more than twice as common as −1 or +1 min (909, 389, 392). They are DB's own reported times as well, so they are not a gap in our data. Whether DB changed how it reports on-time arrivals can't be told from the snapshots.

**Snapshot times are UTC.** `snapshot_timestamp` is UTC, while every train time is German local time. Until early November 2025, it is the start of the 6-hour fetch block, and the request itself can be up to about 45 min later. The script converts it before comparing.

## When cancellations were issued

The snapshots carry the time each cancellation was issued (`cancellation_time`). All 572 cancelled CV journeys have one (`cancellation_announced`): the issue time of the cancellation in force at the Frankfurt departure or the München arrival, whichever was earlier.

| Issued at least … before planned departure | Cancelled CV journeys |
|---|---|
| 24 h | 16 |
| 6 h | 57 |
| 1 h | 206 |
| 20 min | 276 |
| 0 min | 303 |
| after planned departure | 269 |

So at 1 h ahead, 36% of the eventual cancellations were already announced, and at 20 min ahead, 48%.

**Decision: no prediction when a cancellation is already known.** A journey whose cancellation was announced at or before the prediction time has no prediction row for that prediction time. The journey files keep every journey.

| Prediction time | CV rows | Left out | Cancelled share of the rows | Hold-out rows |
|---|---|---|---|---|
| 24 h | 12,068 | 16 | 4.6% | 9,905 |
| 6 h | 12,027 | 57 | 4.3% | 9,841 |
| 1 h | 11,878 | 206 | 3.1% | 9,667 |
| 20 min | 11,808 | 276 | 2.5% | 9,584 |
| 0 min | 11,781 | 303 | 2.3% | 9,547 |

**Flag: withdrawn cancellations stay in.** A journey whose cancellation was announced and later withdrawn, so the train ran, stays in at every prediction time and keeps its real outcome, even if the cancellation was announced before the prediction time. They are marked `cancellation_withdrawn`: 34 in CV and 44 in the hold-out year. With the rebuilt source files they all count as completed; the 32 CV journeys that the old files still marked cancelled are now completed.

## Data-quality audit

Numbers come from `audit.json` and `change_check.json`.

### Open issues

1. **Negative `current_delay` (kept; revisit).** At 20 min ahead, 48 CV values are below −10 min, 42 of them the night train 699 arriving at Frankfurt well before its long planned stop. A train cannot leave early.
2. **Forecast labels in summer 2026.** 128 hold-out arrival times come from a forecast or an unclear snapshot, 102 of them in May–September 2026 (see above). The decision to keep forecast labels without a flag was made when there were 22.
3. **0-min arrivals in the hold-out year** are more than twice as common as −1 or +1 min. They are DB's reported times; whether DB changed how it reports them is unknown.
4. **Early arrivals at München.** In CV, 25 arrivals are more than 10 min early, down to −57 min, most often the evening and night trains 821, 693 and 729. This is plausibly timetable slack.

### Checked, no action needed

- **Forecast labels in CV.** 10 München arrival times come from a forecast or an unclear snapshot. Kept as they are, without a flag.
- **Rebuilt source files.** Compared with the files from before the rebuild, only the 32 withdrawn CV cancellations (now completed) and a few `current_delay` values changed in CV (26 at 1 h, 33 at 0 min). In the hold-out year, 35 cancellations became completed journeys, and in 4 departures a different train of the same departure is kept.
- **Coverage.** Every day has journeys.
  - The lowest days in CV are 4, 6, 7, 8 and 10 May and 19 July 2025, with 10 or 11.
  - In the hold-out year, weekends in late August and September 2026 have 2 to 6. On those days Frankfurt Hbf has under half its usual ICE/IC events while Frankfurt Airport and Mannheim are normal, which points to construction work.
- **Cancellations (CV).** Both ends cancelled: 164. Frankfurt only: 137 (the train still reached München, median 30 min late). München only: 271 (mostly ending at Würzburg or Aschaffenburg). A cancellation only at an intermediate stop does not count: 369 journeys.
- **No live updates at any stop (CV).** 56 journeys, 53 of them cancelled.
- **Time consistency (CV).**
  - Planned times go backwards in 5 journeys; reported times go backwards by more than a minute in 52.
  - Delays are whole minutes.
  - Two night journeys run across a clock change; only their planned travel time is off by an hour.
- **Class boundaries (CV).** 155 arrivals are exactly 20 min late (`moderate`) and 24 exactly 60 min late (`severe`).
- **IC trains.** There are 196 in CV, 30 in late 2025, and none in 2026.
- **Unseen trains.** 721 hold-out journeys (7.3%) have a train number never seen in CV.
- **Edges of the data.**
  - Trains departing late on 30 September 2026 that arrive after midnight are missing, because there is no October file.
  - Runs that started on 30 June 2024 lack their upstream stops.

## Not built yet

The other railway features in the Initial Data Exploration document:

- calendar features;
- `route_id`, `planned_travel_min`, `planned_stops`, `upstream_planned_min`;
- the recent-reliability features.

Weather has not been joined for this pair yet.
