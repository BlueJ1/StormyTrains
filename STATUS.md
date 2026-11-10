# Status

Updated 10 November 2026. This file records decisions and where things are. What is left to do is in [FINISHING_PLAN.md](FINISHING_PLAN.md); the report draft is in [report/](report/README.md).

`FINISHING_PLAN.md` and `finishing_plan.html` hold the same plan; the HTML is published at https://claude.ai/artifact/HKS8W5KAvZb3zty1MbaqBv. Keep the two in sync and republish when a decision is taken or a step is completed.

## Where things are

| Folder | Contents | State |
|---|---|---|
| `frankfurt_munich/` | The modelling dataset for Frankfurt (Main) Hbf → München Hbf: journeys, prediction rows, week split, weather features. See its README | current |
| `frankfurt_munich/weather/` | DWD station mapping, quality review and hourly observations for the route stations | current |
| `frankfurt_munich/modeling/` | Shared cross-validation, tuning, features and models; the first models (15–31 October). Their results use recorded-route inputs and `current_delay` from the final records, so they are optimistic | code current, results historical |
| `frankfurt_munich/advanced_modeling/` | Inputs known at the cutoff, 366 CV arms, finalist registry and the hold-out evaluation (PR #12, merged 8 November). [Plan](frankfurt_munich/advanced_modeling/PLAN.md), [results](frankfurt_munich/advanced_modeling/README.md) | current |
| `report/` | First draft of the final report (LaTeX) and its two figures | draft |
| `corridor_analysis/` | Comparison of candidate origin–destination pairs. `extract.py` is imported by `frankfurt_munich/build_dataset.py`, so the folder must stay | code in use, numbers historical |
| `berlin_munich_weather/`, `arrival_delays/`, `eda.ipynb` | Early work on Berlin → München and on delay distributions. `weather_features.py` imports `berlin_munich_weather/fetch_weather.py` | historical |
| `literature_review/` | Eight papers for the first two weeks, and the surveys behind the model choice | current |

The numbers in the three historical folders come from the monthly files as they were before the source rebuilt them on 6–8 October 2026 (26 months, to August 2026). Those files are no longer on disk, so a rerun gives other counts.

## Decisions

- **Railway data:** the monthly files, live-update snapshots and plan files of `piebro/deutsche-bahn-data` on Hugging Face (from July 2024), not the Bahn-Vorhersage dataset named in the brief. The reason is not recorded yet; see the finishing plan.
- **Origin–destination pair:** Frankfurt (Main) Hbf → München Hbf.
- **Prediction times:** 24 h, 6 h, 1 h, 20 min and 0 min before the scheduled Frankfurt departure.
- **Cancelled** means origin departure or destination arrival cancelled. A cancellation only at an intermediate stop does not count.
- **Four classes:** under 20 min, 20–59 min, 60+ min, cancelled. Exactly 20 min is in the 20–59 class.
- **Split** (12 October 2026, agreed with Qing He): by ISO week of the planned Frankfurt departure, not chronological, because we have only about two years of data. Assigned by a greedy stratified split on `severity_class` (`StratifiedGroupKFold`, seed 42) and committed in `frankfurt_munich/split_weeks.csv`. 23 of 118 weeks are the hold-out set (4,324 journeys); the other 95 are the CV set (17,507), in 5 folds. Changes over time are reported by period (before / from October 2025), not by split.
- **Hold-out set:** opened once, on 6 November 2026, for the registered finalists (Phase 9 of the modelling plan). No model, feature or configuration may be chosen on it.
- **Shared departures** (same planned Frankfurt departure and München arrival under two train numbers): keep one, from the passenger's view. If any of them completed, the departure counts as completed (often a replacement train).
- **Unusual journeys** are dropped before the split: planned travel time over 7 h (ICE 1223 and 2923), and journeys where Frankfurt or München was an additional stop. 193 journeys.
- **IC trains are kept. Negative `current_delay` values are kept.**
- **Metrics:** multi-class log loss chooses severity models, RMSE on completed journeys chooses delay models; macro-F1 is reported as well. One model per prediction time.
- **No prediction** is made for a journey whose cancellation is already announced at the prediction time. Journeys whose cancellation was announced and later withdrawn stay in at every prediction time, flagged `cancellation_withdrawn` (59 in CV, 17 in the hold-out set).
- **`current_delay`** in `frankfurt_munich/data/` is the latest delay that had happened by the prediction time according to the final records. The advanced models use a second version, rebuilt from the snapshots, that requires the report to have been visible by then (`frankfurt_munich/data/advanced/history_*.parquet`). Missing values: 0 plus an indicator for linear models, left missing for tree models. Not used at 24 h.
- **Weather** (17–19 October 2026): DWD observations for the latest full hour at or before the prediction time, aggregated over a fixed station list per branch (via Nürnberg or via Stuttgart). 14 features; nothing is filled. Observed weather, not forecasts.
- **Recent-reliability features** may use hold-out journeys, as long as the outcome was known at the prediction time.
- **Advanced modelling** (4–8 November 2026): decisions D1–D13 and the rules are in `frankfurt_munich/advanced_modeling/PLAN.md`. `PLAN.md` and `plan.html` hold the same plan; the HTML is published at https://claude.ai/artifact/KAso5U9ubq5rFxMy7juUna. Keep the two in sync and republish when the plan changes.

## Main results so far

- With inputs known at the cutoff, nested LightGBM with selected feature groups reaches a CV log loss of 0.920 at 24 h and 0.819 at 0 min (class shares 0.968 and 0.897) and an RMSE of 24.39 and 23.58 min (training mean 25.43).
- Recent reliability is the feature group that helps most. No other model family beat LightGBM by the minimum gain (D12).
- Weather did not help in cross-validation or on the hold-out set: all ten paired 95% intervals include zero.
- The hold-out evaluation ran before the review fixes to two input groups; the CV results include the fixes.
- The first models looked much better (0.759 and 21.55 min at 0 min) because of the recorded-route inputs and, mostly, `current_delay` from the final records.

## Local data, not in the repository

- `data/monthly_processed_data`, `data/monthly_processed_data_change` and `data/monthly_processed_data_plan` (about 19 GB), July 2024 to September 2026, as rebuilt by the source on 6–8 October 2026; the first two were downloaded on 8 October 2026 and checked against the SHA-256 sums. The plan files are pinned to a revision in `frankfurt_munich/advanced_modeling/README.md`.
- `frankfurt_munich/data/advanced/` and `frankfurt_munich/data/advanced_original_065b077/`: generated inputs of the advanced models, and the inputs the hold-out evaluation used.
- `frankfurt_munich/weather/raw/`: DWD archives.
- Out-of-fold predictions (`results/**/oof_predictions.parquet`) are not committed. Those of the first models were copied into `frankfurt_munich/modeling/results/` on 10 November 2026 from the worktrees in which they were made.
