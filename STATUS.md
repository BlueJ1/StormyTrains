# Status

Updated 12 November 2026. This file records decisions and where things are. What is left to do is in [FINISHING_PLAN.md](FINISHING_PLAN.md); the report draft is in [report/](report/README.md).

`FINISHING_PLAN.md` and `finishing_plan.html` hold the same plan; the HTML is published at https://claude.ai/artifact/HKS8W5KAvZb3zty1MbaqBv. Keep the two in sync and republish when a decision is taken or a step is completed.

## Where things are

| Folder | Contents | State |
|---|---|---|
| `frankfurt_munich/` | The modelling dataset for Frankfurt (Main) Hbf → München Hbf: journeys, prediction rows, week split, weather features. See its README | current (rebuilt 12 November 2026) |
| `frankfurt_munich/weather/` | DWD station mapping, quality review and hourly observations for the route stations | current |
| `frankfurt_munich/modeling/` | Shared cross-validation, tuning, features and models; the first models (15–31 October). Their results use recorded-route inputs and `current_delay` from the final records; they stay out of the report and are to be deleted | code current, results to be deleted |
| `frankfurt_munich/advanced_modeling/` | Inputs known at the cutoff, 366 CV arms, finalist registry and the hold-out evaluation (PR #12, merged 8 November). [Plan](frankfurt_munich/advanced_modeling/PLAN.md), [results](frankfurt_munich/advanced_modeling/README.md) | results from before the corrections of 12 November; rebuild and rerun decided |
| `report/` | First draft of the final report (LaTeX) and its two figures | draft |
| `corridor_analysis/` | Comparison of candidate origin–destination pairs. `extract.py` is imported by `frankfurt_munich/build_dataset.py`, so the folder must stay | code in use, numbers historical |
| `berlin_munich_weather/`, `arrival_delays/`, `eda.ipynb` | Early work on Berlin → München and on delay distributions. `weather_features.py` imports an SSL helper from `berlin_munich_weather/fetch_weather.py`, which moves into `frankfurt_munich/` first | to be removed |
| `literature_review/` | Eight papers for the first two weeks, and the surveys behind the model choice | current |

The numbers in the three historical folders come from the monthly files as they were before the source rebuilt them on 6–8 October 2026 (26 months, to August 2026). Those files are no longer on disk, so a rerun gives other counts.

## Decisions

- **Railway data:** the monthly files, live-update snapshots and plan files of `piebro/deutsche-bahn-data` on Hugging Face (from July 2024), not the Bahn-Vorhersage dataset named in the brief: that data has not been made available to us by the data owners. The supervisor is aware (recorded 12 November 2026).
- **Origin–destination pair:** Frankfurt (Main) Hbf → München Hbf.
- **Prediction times:** 24 h, 6 h, 1 h, 20 min and 0 min before the scheduled Frankfurt departure.
- **Cancelled** means origin departure or destination arrival cancelled. A cancellation only at an intermediate stop does not count.
- **Four classes:** under 20 min, 20–59 min, 60+ min, cancelled. Exactly 20 min is in the 20–59 class.
- **Split** (12 October 2026, agreed with Qing He): by ISO week of the planned Frankfurt departure, not chronological, because we have only about two years of data. Assigned by a greedy stratified split on `severity_class` (`StratifiedGroupKFold`, seed 42) and committed in `frankfurt_munich/split_weeks.csv`. 23 of 118 weeks are the hold-out set (4,324 journeys); the other 95 are the CV set (17,507), in 5 folds. Changes over time are reported by period (before / from October 2025), not by split.
- **Hold-out set:** opened once, on 6 November 2026, for the registered finalists (Phase 9 of the modelling plan). After the corrections of 12 November, new finalists are registered from the corrected CV results and scored on it once more, with class shares and the training mean as references (Phase 10). No model, feature or configuration may be chosen on it.
- **Shared departures** (same planned Frankfurt departure and München arrival under two train numbers): keep one, from the passenger's view. If any of them completed, the departure counts as completed (often a replacement train).
- **Unusual journeys** are dropped before the split: planned travel time over 7 h (ICE 1223 and 2923), and journeys where Frankfurt or München was an additional stop. 193 journeys.
- **IC trains are kept. Negative `current_delay` values are kept.**
- **Metrics:** multi-class log loss chooses severity models, RMSE on completed journeys chooses delay models; macro-F1 is reported as well. One model per prediction time.
- **No prediction** is made for a journey while a cancellation of its Frankfurt departure or München arrival is in force (decided 12 November 2026). A cancellation holds from its issue time; a withdrawal ends it from the time the first snapshot that shows the withdrawal was requested, and the journey has prediction rows again after that. Journeys that ran after a withdrawn cancellation are flagged `cancellation_withdrawn` (59 in CV, 17 in the hold-out set). The dataset was rebuilt with this rule on 12 November 2026; before, these journeys kept every row.
- **`current_delay`** in `frankfurt_munich/data/` is the latest delay that had happened by the prediction time according to the final records. The advanced models use a second version, rebuilt from the snapshots, that requires the report to have been visible by then (`frankfurt_munich/data/advanced/history_*.parquet`). Since the rebuild of 12 November 2026 the first version no longer uses the departure from Frankfurt itself. Missing values: 0 plus an indicator for linear models, left missing for tree models. Not used at 24 h.
- **Weather** (17–19 October 2026): DWD observations for the latest full hour at or before the prediction time, aggregated over a fixed station list per branch (via Nürnberg or via Stuttgart). 14 features; nothing is filled. Observed weather, not forecasts.
- **Recent-reliability features** may use hold-out journeys, as long as the outcome was known at the prediction time.
- **Decisions of 12 November 2026** (all in `FINISHING_PLAN.md`, Q1–Q12): fix the network state; treat the timetable as static after 3 November 2025 as well; correct the dataset build for the selection effects R5–R8 (a journey has no prediction row while its cancellation is in force, and has one again after a withdrawal); rebuild every input and rerun all CV arms; register new finalists and evaluate them on the hold-out set; the report gives the new evaluation only and does not mention the first models, whose results are deleted; scope frozen for now; remove `arrival_delays/`, `berlin_munich_weather/` and `eda.ipynb`, keep `corridor_analysis/`; create a root README; the statement of who did what is left for later. State on 12 November: the code is corrected for the network state, the static timetable (in the plan rule shared by all input builders) and, in the dataset build, the cancellation rule, the first issue time and `current_delay`; tests are in `frankfurt_munich/tests/` and `frankfurt_munich/advanced_modeling/tests/`. The dataset in `frankfurt_munich/data/` was rebuilt on 12 November (85,826 CV and 21,236 hold-out prediction rows; journeys, folds and hold-out weeks unchanged), and its README is current. The advanced inputs in `frankfurt_munich/data/advanced/`, every result and the numbers in the two modelling READMEs are still the old ones; the inputs were built for the old prediction rows. For a kept replacement train (R5), `build_dataset.py` is not changed: the journey files keep the attributes of the train that ran, and the advanced models take `train_type` and `starts_at_origin` from the scheduled train through the timetable inputs. The rules written for the cancellation timeline and `current_delay` were confirmed on 12 November. Next: rebuild the advanced inputs (B11).
- **Advanced modelling** (4–8 November 2026): decisions D1–D16 and the rules are in `frankfurt_munich/advanced_modeling/PLAN.md`. `PLAN.md` and `plan.html` hold the same plan; the HTML is published at https://claude.ai/artifact/KAso5U9ubq5rFxMy7juUna. Keep the two in sync and republish when the plan changes.

## Main results so far

All from before the corrections decided on 12 November; they are replaced after the rerun.

- With inputs known at the cutoff, nested LightGBM with selected feature groups reaches a CV log loss of 0.920 at 24 h and 0.819 at 0 min (class shares 0.968 and 0.897) and an RMSE of 24.39 and 23.58 min (training mean 25.43).
- Recent reliability is the feature group that helps most. No other model family beat LightGBM by the minimum gain (D12).
- Weather did not help in cross-validation or on the hold-out set: all ten paired 95% intervals include zero.
- The hold-out evaluation ran before the review fixes to two input groups; the CV results include the fixes.
- The first models looked much better (0.759 and 21.55 min at 0 min) because of the recorded-route inputs and, mostly, `current_delay` from the final records.

## Local data, not in the repository

- `data/monthly_processed_data`, `data/monthly_processed_data_change` and `data/monthly_processed_data_plan` (about 19 GB), July 2024 to September 2026, as rebuilt by the source on 6–8 October 2026; the first two were downloaded on 8 October 2026 and checked against the SHA-256 sums. The plan files are pinned to a revision in `frankfurt_munich/advanced_modeling/README.md`.
- `frankfurt_munich/data/advanced/` and `frankfurt_munich/data/advanced_original_065b077/`: generated inputs of the advanced models, and the inputs the hold-out evaluation used.
- `frankfurt_munich/weather/raw/`: DWD archives.
- A copy of every file under `frankfurt_munich/` that is not in git, as of 12 November 2026 before any rebuild, is in `../Project in Data Science backup 2026-11-12/` (1.06 GB, with checksums). It is on this laptop only.
- Out-of-fold predictions (`results/**/oof_predictions.parquet`) are not committed. Those of the first models were copied into `frankfurt_munich/modeling/results/` on 10 November 2026 from the worktrees in which they were made.
