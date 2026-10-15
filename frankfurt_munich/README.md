# Frankfurt (Main) Hbf → München Hbf dataset

The modelling dataset for the chosen origin–destination pair, railway data only.

- **Journeys and targets** follow `corridor_analysis/extract.py`. `build_dataset.py` reuses its functions for this pair alone.
- **On top of that:**
  - each departure is kept once;
  - journeys over 7 h and journeys with an additional stop at Frankfurt or München are dropped;
  - `current_delay` is added at each prediction time;
  - the time a cancellation was announced is added, so that no prediction is made once a cancellation is known;
  - the data are split at random by day into a cross-validation set (80% of the days, in 5 folds) and a hold-out set (20%).
- **Source:** the 27 monthly files from July 2024 to September 2026 and the matching live-update snapshots, as rebuilt by the source on 6–8 October 2026 and downloaded on 8 October 2026 (checked against the published SHA-256 sums).

**The hold-out set is not to be looked at yet.** Outcome statistics below (classes, delays, cancellations) are for the CV set only. For the hold-out set we only check data quality: counts, coverage, missing values and where the labels come from. Where something changes over time, the README compares July 2024–September 2025 with October 2025–September 2026; these are periods, not splits.

## Files

| File | Rows | Contents |
|---|---|---|
| `data/journeys_cv.parquet` | 17,505 | The cross-validation set: journeys on 662 of the 822 days (80%), planned Frankfurt departure from 1 July 2024 to 30 September 2026, with their `fold` |
| `data/journeys_holdout.parquet` | 4,326 | The hold-out set: journeys on the other 160 days (20%) |
| `data/prediction_rows_cv.parquet` | 85,870 | One row per CV journey and prediction time, without known cancellations (see below). A model for one prediction time uses the rows with that `prediction_time` |
| `data/prediction_rows_holdout.parquet` | 21,291 | The same for the hold-out set |
| `data/stops.parquet` | 133,794 | Recorded stops from Frankfurt to München of the kept journeys |
| `data/upstream_runs.parquet` | 116,528 | Every recorded stop of each run up to and including Frankfurt, used for `current_delay` |
| `data/change_check.parquet` | 21,831 | Per journey: where the München arrival time comes from, and when a cancellation was issued (see below) |
| `build_audit.json`, `audit.json`, `change_check.json` | | Counts behind every number in this README |

## Dataset files

All times are German local time without a time zone, except `snapshot_timestamp` in the source (see below). Files link through `journey_id`: the source's `train_line_ride_id`, a dash and the planned start of the run (`yymmddHHMM`). A stop `id` is `journey_id`, a dash and `train_line_station_num`.

### `journeys_cv.parquet` and `journeys_holdout.parquet`

One row per kept journey (one per departure), sorted by planned Frankfurt departure. Both files have the same columns, except that only `journeys_cv.parquet` has `fold`.

| Column | Meaning |
|---|---|
| `journey_id`, `train_type`, `train_number` | The journey; `train_type` is `ICE` or `IC` |
| `origin_departure_planned_local`, `origin_departure_reported_local` | Planned and reported departure from Frankfurt (Main) Hbf |
| `destination_arrival_planned_local`, `destination_arrival_reported_local` | Planned and reported arrival at München Hbf. A reported time is the planned time when no change was ever reported (see "Where the labels come from") |
| `origin_departure_canceled`, `destination_arrival_canceled` | The Frankfurt departure or the München arrival is cancelled |
| `any_selected_stop_canceled` | Any recorded stop's event (departure, or arrival at München) is cancelled, including intermediate stops |
| `observed_stops` | Number of recorded stops from Frankfurt to München |
| `recorded_route` | Their station names in stop order, joined with ` \| ` |
| `unobserved_sequence_positions` | Positions in the run's stop sequence between Frankfurt and München with no record at all |
| `journey_canceled`, `arrival_delay_min`, `severity_class`, `severe_or_canceled` | The targets (see "Targets") |
| `is_replacement_train` | Any stop of the journey is flagged as a replacement train |
| `shared_departure` | Number of journeys listed for this departure before one was kept (see "One journey per departure") |
| `starts_at_origin` | The run begins at Frankfurt, so it has no upstream stops |
| `current_delay_24h` … `current_delay_0min` | Latest known delay at each prediction time, in minutes (see "current_delay") |
| `cancellation_announced` | When the cancellation of the Frankfurt departure or München arrival was issued, the earlier of the two; missing for journeys that ran |
| `cancellation_withdrawn` | A cancellation was announced and later withdrawn, and the train ran |
| `fold` | CV file only: the cross-validation fold, 1 to 5 (see "Hold-out set and cross-validation folds") |

### `prediction_rows_cv.parquet` and `prediction_rows_holdout.parquet`

One row per journey and prediction time, sorted by planned departure, journey and prediction time. Journeys whose cancellation was already announced at that time have no row (see "When cancellations were issued").

| Column | Meaning |
|---|---|
| `journey_id`, `origin_departure_planned_local`, `train_number`, `cancellation_withdrawn` | Copied from the journey file |
| `arrival_delay_min`, `journey_canceled`, `severity_class`, `severe_or_canceled` | The targets, copied from the journey file |
| `prediction_time` | `24h`, `6h`, `1h`, `20min` or `0min` (an ordered category) |
| `predicted_at` | Planned Frankfurt departure minus the prediction time |
| `current_delay` | The journey's `current_delay_<prediction_time>` |
| `fold` | CV file only: the journey's fold |

### `stops.parquet`

One row per recorded stop from Frankfurt to München of every kept journey, both splits, sorted by journey and stop. Only stations in `data/stations_present_in_all_months.csv` are kept.

- **Columns from the monthly files, unchanged:** `id`, `station_name`, `xml_station_name`, `eva`, `train_type`, `train_number`, `line_number` (always empty here), `final_destination_station`, `train_line_ride_id`, `train_line_station_num`, `arrival_planned_time`, `arrival_change_time`, `departure_planned_time`, `departure_change_time`, `arrival_is_canceled`, `departure_is_canceled`, `time`, `delay_in_min`, `is_additional_stop`, `is_replacement_train`, `replaced_train_type`, `replaced_train_number`, `has_plan_record`.
- **Added:**

| Column | Meaning |
|---|---|
| `journey_id` | The journey the stop belongs to |
| `stop_index` | Position after Frankfurt in the run's stop sequence; 0 at Frankfurt. Skips numbers where a stop is not recorded |
| `final_destination` | The stop is München Hbf |
| `event_kind` | `arrival` at München, `departure` everywhere else |
| `event_planned_time`, `event_reported_time`, `selected_event_canceled` | Planned time, reported time and cancellation of that event |

### `upstream_runs.parquet`

Every recorded stop of each journey's run up to and including Frankfurt, both splits, sorted by journey and `train_line_station_num`. Only panel stations are kept. The columns are a subset of the monthly files' (`id`, `station_name`, `train_type`, `train_line_ride_id`, `train_line_station_num`, the four planned and change times, the two cancellation flags, `is_replacement_train`) plus `journey_id`. `current_delay` is computed from this file.

### `change_check.parquet`

One row per journey, both splits, with what the live-update snapshots say about its München arrival and Frankfurt departure.

| Column | Meaning |
|---|---|
| `journey_id`, `split` (`cv` or `holdout`) | The journey |
| `origin_departure_planned_local`, `destination_arrival_planned_local`, `destination_arrival_reported_local`, `journey_canceled`, `arrival_delay_min`, `cancellation_announced`, `cancellation_withdrawn` | Copied from the journey files |
| `arr_*` / `dep_*` | The same ten columns for the München arrival (`arr_`) and the Frankfurt departure (`dep_`), listed below |
| `…snapshots`, `…snapshots_with_time` | Number of snapshots that saw the stop, and how many of them carry a changed time |
| `…latest_request` | Request time (local) of the latest snapshot with any change, the one the monthly files use |
| `…latest_block_label` | That snapshot's timestamp is the start of a 6-hour fetch block, so the request can be up to 45 min later |
| `…latest_time` | The changed time in that snapshot |
| `…last_time`, `…earlier_time` | The last changed time in any snapshot, and the one before it |
| `…latest_cancellation_time`, `…latest_status` | Cancellation issue time and change status in the latest snapshot with any change (`c` cancelled, `p` withdrawn) |
| `…first_cancellation_time` | The first cancellation issue time in any snapshot |
| `covered` | The arrival is early enough to be checked against the snapshots |
| `arrival_source` | Where the München arrival time comes from: `after arrival`, `before arrival` (a forecast), `around arrival`, `no snapshot`, `cancelled` or `not covered` |
| `arrival_snapshot_lag_min` | Request time minus arrival time of the latest snapshot; negative for a forecast |
| `arrival_reproduced` | Our arrival time equals the snapshot value; only for covered, completed journeys |
| `cancellation_lead_min` | Minutes from `cancellation_announced` to the planned Frankfurt departure; negative if issued after it |

## Rebuild

```bash
.venv/bin/python frankfurt_munich/build_dataset.py      # about 1 min; needs both data folders
.venv/bin/python frankfurt_munich/audit.py
.venv/bin/python frankfurt_munich/check_change_data.py
```

Both `data/monthly_processed_data` and `data/monthly_processed_data_change` from the Hugging Face dataset are needed.

| Script | Purpose |
|---|---|
| `build_dataset.py` | Builds every file in `data/` and `build_audit.json` |
| `folds.py` | Hold-out set and cross-validation folds |
| `snapshots.py` | Reads the live-update snapshots; used by the two scripts above and below |
| `audit.py` | Data-quality audit, `audit.json` |
| `check_change_data.py` | Where each label comes from and when cancellations were announced, `change_check.json` and `data/change_check.parquet` |

## Hold-out set and cross-validation folds

We have only about two years of data, so the split is random rather than chronological. The definition is in `folds.py`.

- **By day.** All journeys whose planned Frankfurt departure falls on the same calendar day go to the same part. Journeys on one day share weather, strikes and network disruptions, and delays carry over from train to train, so splitting single journeys would make the scores too optimistic.
- **Hold-out set:** 20% of the days, 160 days with 4,326 journeys (19.8% of the journeys). Every month has hold-out days, 2 to 11 of them.
- **CV set:** the other 662 days, 17,505 journeys, divided into 5 folds of days in the same way. Fold *k* validates on the journeys with `fold == k` and trains on the other four folds.
- **Stratified** by `severity_class` (scikit-learn's `StratifiedGroupKFold`), so the class shares of the hold-out set and of every fold are within 0.3 percentage points of the CV set's.
- **Fixed seed** (42), so a rebuild gives the same split.

| Fold | Days | Journeys |
|---|---|---|
| 1 | 131 | 3,490 |
| 2 | 132 | 3,491 |
| 3 | 132 | 3,510 |
| 4 | 134 | 3,516 |
| 5 | 133 | 3,498 |

## Evaluation

- **Choosing and tuning models on the CV folds:** multi-class log loss.
- **Reported at the end:** macro-F1 as well.

## Targets

- `arrival_delay_min`: arrival delay at München Hbf in minutes, computed from arrival times. Missing for cancelled journeys.
- `journey_canceled`: the Frankfurt departure or the München arrival is cancelled. A cancellation only at an intermediate stop does not count.
- `severity_class`: `normal_minor` (under 20 min), `moderate` (20–59 min; exactly 20 belongs here), `severe` (60 min or more), `canceled`.
- `severe_or_canceled`: the binary version.

CV class shares: 62.9% under 20 min, 25.2% 20–59 min, 6.4% 60+ min, 5.5% cancelled.

## One journey per departure

Some departures appear as two or more journeys under different train numbers, with the same planned Frankfurt departure and München arrival. There were 1,341 such groups, covering 2,765 journeys. After unusual journeys are dropped (see below), 1,337 of these departures remain: 271 from July 2024 to September 2025 and 1,066 from October 2025 on (1,107 in the CV set, 230 in the hold-out set).

- **The usual case:** the scheduled train is marked cancelled, and a replacement train with another number ran at the same times.
- **Rule (passenger's view):** one journey is kept per departure: one that completed if there is one, then one not flagged as a replacement, then the lowest train number. 1,424 journeys were removed.
  - In 265 of the 298 groups where one train was cancelled and another ran, the kept journey is the replacement.
  - One cancelled train was replaced at different times (2849 for 699, 43 min earlier); it stays cancelled. The replacement itself is dropped as unusual, because München was an additional stop.
- **`shared_departure`** is the number of journeys that were listed for the departure.

## Unusual journeys

After one journey is kept per departure, two kinds of journeys are dropped, so the whole departure leaves the dataset. 193 journeys in all: 148 from July 2024 to September 2025 and 45 from October 2025 on. They are dropped before the split, so they are in neither set.

- **Planned travel time over 7 h: 174 journeys.** ICE 1223 (172) and ICE 2923 (2) leave Frankfurt northwards and go via Köln, the Ruhr and Kassel, arriving in München about 9 h later. Nobody would take them from Frankfurt to München, and their outcomes differ (16.1% of these 174 journeys are cancelled, against 5.5% in the CV set). Every other journey is planned at 185 to 369 min.
- **Frankfurt or München is an additional stop (`is_additional_stop`): 19 journeys.** The stop was not in the train's original timetable. In 18, München was added (in one of them Frankfurt as well), for example a train extended from Würzburg to München or one diverted through München Hbf. In 1, only Frankfurt was added. The source numbers added München stops from 100, so the recorded stop order of those journeys was wrong. Two of them were replacement trains kept for a departure whose scheduled train was cancelled (827 on 9 January 2025, 693 on 19 May 2025). The departure is dropped rather than counted as cancelled.

Journeys where Frankfurt was added and numbered from 100 while München was not were never in the dataset: the extraction treats them as running in the wrong direction. There are 25 of them.

## current_delay

`current_delay_24h`, `_6h`, `_1h`, `_20min` and `_0min` are taken at 24 h, 6 h, 1 h, 20 min and 0 min before the planned Frankfurt departure.

The value is the latest delay known at that moment, from the train's own run up to Frankfurt:

- The latest non-cancelled arrival or departure that has happened by then.
- If that event is an arrival (the train is standing at a stop and has not left yet), the delay accumulated by then. That is the larger of the arrival delay and the minutes since the planned departure from that stop.
- Missing if nothing has happened yet.

Only stations in `data/stations_present_in_all_months.csv` are used, so the coverage expansion on 3 November 2025 does not add upstream stops from then on.

Missing values in the CV set (17,505 journeys):

| Prediction time | Missing | Train starts in Frankfurt | Run not started yet | No usable upstream report | Cancelled, missing vs known |
|---|---|---|---|---|---|
| 24 h | 17,505 (100%) | 2,084 | 15,335 | 86 | 5.5% vs – |
| 6 h | 16,568 (95%) | 2,084 | 14,398 | 86 | 5.6% vs 3.0% |
| 1 h | 2,485 (14%) | 2,084 | 315 | 86 | 10.4% vs 4.6% |
| 20 min | 2,409 (14%) | 2,084 | 239 | 86 | 10.0% vs 4.7% |
| 0 min | 1,755 (10%) | 1,510 | 159 | 86 | 13.1% vs 4.6% |

At 0 min, the 1,510 trains that start in Frankfurt and have no value had not left by their scheduled time.

**Decided treatment.** Missing values are not set to plain 0: a 0 means "on time", while a missing value goes with two to three times as many cancellations.
- 24 h: not used (always missing). 6 h: probably not used (95% missing).
- From 1 h on: 0 plus a missing indicator for linear models; left missing for tree models.
- The files keep the missing values; the treatment is applied when a model is fitted.

## Where the labels come from

The monthly files keep, for each stop, the values from the latest live-update snapshot that carries any change. An arrival time without a change would fall back to the planned time, and so to a 0-min delay. `check_change_data.py` traces each München arrival back to the snapshots in `data/monthly_processed_data_change`.

**Coverage.** The snapshots cover every journey arriving before 30 September 2026: 21,800 of 21,831 journeys, 20,610 of them completed. The last 31 are left out of the check only because the last snapshot day is too close.

| Source of the München arrival time | Jul 2024–Sep 2025 | Oct 2025–Sep 2026 | CV set | Hold-out set |
|---|---|---|---|---|
| Snapshot taken after the train arrived (an actual time) | 11,378 | 9,093 | 16,425 | 4,046 |
| Snapshot taken before arrival (a forecast) | 6 | 126 | 90 | 42 |
| Snapshot within 45 min of arrival (can't tell which) | 4 | 2 | 4 | 2 |
| No snapshot at all, so the planned time was used | 0 | 1 | 1 | 0 |

Our files reproduce the snapshot values exactly for every completed journey.

- **Forecast labels** are rare until September 2025 (10, all June 2025) and more common from May 2026: 17 in May, 7 in June, 39 in July, 25 in August, 14 in September. The median forecast was taken about 47 min before arrival.
- **One journey without a snapshot** (7 January 2026) got the planned arrival time, so a 0-min delay.

**Exactly 0 min arrivals.**
- 1,547 of the 1,563 covered 0-min arrivals are DB's own time, read after arrival; 15 are forecasts and 1 is the journey without a snapshot.
- In the CV set until September 2025 they are no more common than −1 or +1 min (523, 502, 454). The delay at the stop before is normal (median 1 min, 0.4% at 10+ min).
- In the CV set from October 2025 they are more than twice as common as −1 or +1 min (759, 328, 326). They are DB's own reported times as well, so they are not a gap in our data. Whether DB changed how it reports on-time arrivals can't be told from the snapshots.

**Snapshot times are UTC.** `snapshot_timestamp` is UTC, while every train time is German local time. Until early November 2025, it is the start of the 6-hour fetch block, and the request itself can be up to about 45 min later. The script converts it before comparing.

## When cancellations were issued

The snapshots carry the time each cancellation was issued (`cancellation_time`). All 954 cancelled CV journeys have one (`cancellation_announced`): the issue time of the cancellation in force at the Frankfurt departure or the München arrival, whichever was earlier.

| Issued at least … before planned departure | Cancelled CV journeys |
|---|---|
| 24 h | 42 |
| 6 h | 133 |
| 1 h | 395 |
| 20 min | 520 |
| 0 min | 565 |
| after planned departure | 389 |

So at 1 h ahead, 41% of the eventual cancellations were already announced, and at 20 min ahead, 55%.

**Decision: no prediction when a cancellation is already known.** A journey whose cancellation was announced at or before the prediction time has no prediction row for that prediction time. The journey files keep every journey.

| Prediction time | CV rows | Left out | Cancelled share of the rows | Hold-out rows |
|---|---|---|---|---|
| 24 h | 17,463 | 42 | 5.2% | 4,318 |
| 6 h | 17,372 | 133 | 4.7% | 4,306 |
| 1 h | 17,110 | 395 | 3.3% | 4,247 |
| 20 min | 16,985 | 520 | 2.6% | 4,219 |
| 0 min | 16,940 | 565 | 2.3% | 4,201 |

**Flag: withdrawn cancellations stay in.** A journey whose cancellation was announced and later withdrawn, so the train ran, stays in at every prediction time and keeps its real outcome, even if the cancellation was announced before the prediction time. They are marked `cancellation_withdrawn`: 67 in the CV set and 9 in the hold-out set (32 until September 2025, 44 from October 2025). With the rebuilt source files they all count as completed; 30 of them, all until September 2025, were still marked cancelled in the old files.

## Data-quality audit

Numbers come from `audit.json` and `change_check.json`.

### Open issues

1. **Negative `current_delay` (kept; revisit).** At 20 min ahead, 61 CV values are below −10 min, 53 of them the night train 699 arriving at Frankfurt well before its long planned stop. A train cannot leave early.
2. **Forecast labels in summer 2026.** 128 arrival times from October 2025 on come from a forecast or an unclear snapshot, 102 of them in May–September 2026 (see above). The decision to keep forecast labels without a flag was made when there were 22.
3. **0-min arrivals from October 2025** are more than twice as common as −1 or +1 min. They are DB's reported times; whether DB changed how it reports them is unknown.
4. **Early arrivals at München.** In the CV set, 49 arrivals are more than 10 min early, down to −30 min, most often the evening and night trains 821, 729 and 693. This is plausibly timetable slack.

### Checked, no action needed

- **Forecast labels until September 2025.** 10 München arrival times come from a forecast or an unclear snapshot. Kept as they are, without a flag.
- **Rebuilt source files.** Compared with the files from before the rebuild, only the 30 withdrawn cancellations (now completed) and a few `current_delay` values changed until September 2025 (26 at 1 h, 32 at 0 min). From October 2025, 35 cancellations became completed journeys, and in 4 departures a different train of the same departure is kept.
- **Coverage.** Every day has journeys.
  - The lowest days until September 2025 are 4, 6, 7, 8 and 10 May and 19 July 2025, with 10 or 11.
  - From October 2025, weekends in late August and September 2026 have 2 to 6. On those days Frankfurt Hbf has under half its usual ICE/IC events while Frankfurt Airport and Mannheim are normal, which points to construction work.
- **Cancellations (CV).** Both ends cancelled: 262. Frankfurt only: 288 (the train still reached München, median 24 min late). München only: 404 (mostly ending at Würzburg or Aschaffenburg). A cancellation only at an intermediate stop does not count: 428 journeys.
- **No live updates at any stop (CV).** 87 journeys, 78 of them cancelled.
- **Time consistency (CV).**
  - Planned times never go backwards (the 5 journeys where they did had an additional stop and were dropped); reported times go backwards by more than a minute in 59.
  - Delays are whole minutes.
  - 6 night journeys leave in the early hours of a clock-change date. Times are local clock time, so a travel time across the change can be off by up to an hour.
- **Class boundaries (CV).** 203 arrivals are exactly 20 min late (`moderate`) and 32 exactly 60 min late (`severe`).
- **IC trains.** There are 196 until September 2025, 30 in late 2025, and none in 2026.
- **Unseen trains.** 4 hold-out journeys (0.1%) have a train number never seen in the CV set.
- **Edges of the data.**
  - Trains departing late on 30 September 2026 that arrive after midnight are missing, because there is no October file.
  - Runs that started on 30 June 2024 lack their upstream stops.

## Not built yet

The other railway features in the Initial Data Exploration document:

- calendar features;
- `route_id`, `planned_travel_min`, `planned_stops`, `upstream_planned_min`;
- the recent-reliability features. Decided: they may use every earlier journey, including hold-out journeys, as long as it is known at the prediction time.

Weather has not been joined for this pair yet.
