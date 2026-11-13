# Advanced modelling: Frankfurt (Main) Hbf → München Hbf

The experiments of the [advanced modelling plan](PLAN.md), added in PR #12: inputs known at the cutoff, new feature groups, nested tuning, other model families, ensembles, neural networks, transfer learning, and the one-time hold-out evaluation of railway-only against weather models. The plan is also published at https://claude.ai/artifact/KAso5U9ubq5rFxMy7juUna (same content as `PLAN.md` and `plan.html`).

**Outcome.** Most of this work is exploratory screening, scored on the same five CV week folds as [`../modeling/`](../modeling/README.md). New input features gave the clearest gains: recent reliability (step 2.1) reached the predeclared minimum gain (D12) in 17 of 20 comparisons, and all new groups together in 19 of 20. Changes to the learner rarely helped. Seed and configuration bagging reached D12 in 8 of 120 comparisons and pooling the horizons in 6 of 20. The two-stage models, CatBoost, the explainable boosting machine, the distributional models, calibration, stacking, weighted ensembles and corridor or network transfer never did, and the plain MLP never did against LightGBM. The registered final models are LightGBM and linear regression, the families already in `../modeling/`. On the 23 hold-out weeks, adding weather did not improve either task: all ten paired 95% intervals include zero.

## Layout

| Files | Purpose |
|---|---|
| `scheduled_timetable.py`, `history_features.py`, `recent_reliability.py`, `network_state.py`, `holiday_features.py` | Inputs known at the cutoff, built from the plan and change snapshots: planned timetable, live delay and its dynamics, recent reliability, network state, holidays |
| `build_advanced_weather.py`, `build_corridor_inputs.py`, `network_transfer.py`, `sequence_inputs.py` | Weather on the planned route, other corridors (8.3), network-wide delay pairs (8.4), upstream sequences (7.5) |
| `holdout_gate.py` | Refuses hold-out builds unless the committed registry matches the current source |
| `run_advanced.py`, `experiment_specs.py`, `experiment_models.py`, `advanced_data.py` | The CV steps of the plan: which arms exist, their models and inputs |
| `representations.py`, `group_selection.py`, `decomposed.py`, `ensembles.py`, `extra_models.py`, `distributions.py`, `neural.py`, `gated_neural.py`, `sequence_model.py`, `corridor_transfer.py` | Phase 1 and 3 to 8 methods |
| `compare.py`, `report_cv.py` | Paired fold comparisons and the CV tables below |
| `run_final.py`, `final_analysis.py`, `result_plots.py`, `finalists.json` | Registration of finalists and the hold-out evaluation |
| `results/` | One folder per arm, and `results/advanced_holdout/` |
| `tests/` | `.venv/bin/python -m pytest frankfurt_munich/advanced_modeling/tests` |
| `_paths.py` | Puts `../modeling` and `..` on the import path |

The cross-validation, tuning, feature sets and model factories are shared with the earlier models in `../modeling/` (`cv.py`, `tuning.py`, `features.py`, `models.py`, `data.py`). Generated inputs are written to `../data/advanced/`, which is not committed. The distributional learners need the separate environment in [requirements-distributions.txt](requirements-distributions.txt).

## Inputs and runs

The earlier scores in [`../modeling/`](../modeling/README.md) use recorded-route inputs. `run_advanced.py` runs the CV experiments in [PLAN.md](PLAN.md) with the new cutoff-specific inputs. It keeps the same prediction rows and week folds. The recorded route fields are available only in the step 0.7 leakage comparison and the historical reproduction commands.

The advanced runner reads generated CV inputs in `../data/advanced/`. It requires the timestamped timetable and reconstructed live-delay data and checks that they cover the same prediction keys. It reads advanced weather only for the weather arm.

Download the 27 `monthly_processed_data_plan` files for July 2024 through September 2026 into `data/monthly_processed_data_plan/`. [Pinned plan archive](https://huggingface.co/datasets/piebro/deutsche-bahn-data/tree/d27b9c413536bb5fe393954371f572050e55176b/monthly_processed_data_plan)

The legacy fetcher used the requested timetable hour in plan filenames. Following the user decision on 5 November 2026, the primary build accepts these legacy fields regardless of fetch time, using the earliest plan per stop as its static schedule. Following the decision of 12 November 2026, the plans with exact request stamps (from 3 November 2025) are treated the same way. The 45-minute bound applies to the separate change archive. [Legacy fetcher](https://github.com/piebro/deutsche-bahn-data/blob/f19888b770a68d5f9010adac33af70e400e2395d/fetch_data.py)

The prediction cohort supplies the planned Frankfurt departure and München arrival. The additional timetable fields come from the scheduled train's plan available at the cutoff. If the branch is unavailable, the weather pipeline retains the two endpoint temperatures and leaves the route aggregates missing. The live delay requires both a visible change report and a visible plan. For retained replacements, the scheduled service supplies live history until the link to the replacement becomes visible. When the monthly files record no replaced train (17 CV journeys), no other scheduled train is known, and the operated run supplies the history as for an ordinary train. Those runs have no plan snapshot, so `current_delay` stays missing for them; only the report count is filled. Seven further CV replacements name a replaced train that no plan matched, and their history stays missing. The CV results were computed before this change, which alters `upstream_stops_reported` and `minutes_since_last_report` in 22 CV rows and leaves `current_delay` unchanged.

Recent-reliability candidates come from the same target-free stop roster for CV and holdout. Both builds reconstruct past outcomes from snapshots available at each cutoff. The resulting rates describe journeys with observed outcomes. A missing arrival report stays unknown. An archive check found a destination arrival report somewhere in the snapshots for 16,550 of 16,551 completed CV journeys. That check measures archive coverage. At an individual cutoff, the builder still requires a visible report and an arrival time at or before the cutoff.

Input coverage after the rebuild of 13 November 2026 (corrected dataset, static plans in both archive periods). The result tables below are from before this rebuild.

| Input coverage | 24 h | 6 h | 1 h | 20 min | 0 min |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV prediction rows | 17,463 | 17,366 | 17,096 | 16,975 | 16,926 |
| Planned route available | 17,364 | 17,267 | 16,998 | 16,878 | 16,830 |
| Weather branch available | 17,363 | 17,266 | 16,997 | 16,877 | 16,829 |
| Snapshot-based `current_delay` used by models | — | — | 6,654 | 7,331 | 8,758 |

The snapshot-based `current_delay` is missing for about half the rows because the archive fetched live updates in blocks about six hours apart: for most journeys the last fetch before the cutoff came before the train had reported from any stop before Frankfurt (finishing plan B1). Since 13 November 2026 it does not use the departure from Frankfurt itself, as in the dataset's `current_delay`.

Inputs that follow the archive's fetch clock rather than the trains are not used (decisions of 13 November 2026): the network state (step 2.5; `network_state.py` is kept but not run), the corridor reliability over 3 h, the time since the last report and `current_delay_change_min`. When the latest report is an arrival, `current_delay` counts the time the train has stood there since its planned departure only if the same snapshot shows the departure after the cutoff, or the stop is Frankfurt. The scheduled train type is missing for 24 of the 230 kept replacement journeys, whose replaced train no plan matched; the loader sets it to `unknown`.

```bash
.venv/bin/python frankfurt_munich/advanced_modeling/scheduled_timetable.py
.venv/bin/python frankfurt_munich/advanced_modeling/history_features.py --feature all
.venv/bin/python frankfurt_munich/advanced_modeling/build_advanced_weather.py
.venv/bin/python frankfurt_munich/advanced_modeling/run_advanced.py 0.7 --jobs 4
.venv/bin/python frankfurt_munich/advanced_modeling/run_advanced.py 1.1 --jobs 4
.venv/bin/python frankfurt_munich/advanced_modeling/report_cv.py --write-readme
```

`run_advanced.py --help` lists the CV steps and filters. Fixed configurations come from the committed flat searches. Following the user decision on 5 November, remaining booster and CatBoost searches use 20 configurations inside each outer training set. Six completed 50-configuration baseline searches also supply budget-matched 20-configuration controls by reusing their fixed first 20 inner-score rows and refitting the outer models. The saved configurations link to the original searches. The reference table and each run configuration record the actual count. Final registration uses at most 20 configurations. Neural searches use eight. The `auto_groups` and `auto_representations` options to `--nested-variant` also select features inside each outer training set. A variant fixed after seeing the exploratory results retains that label even when its parameters are tuned again.

When running a subset of horizons with `--prediction-time`, set a separate `--results-dir`. Reports and registration select the latest timestamp in each run directory, so a partial run in the main results directory can replace the complete run in those reports.

The model-family screens use the cutoff-safe base columns for their learner comparisons. The nested booster feature-group policy evaluates the expanded Phase 2 inputs separately.
The corrected policy ranks each feature variant by its best inner-fold score across the first three configurations in the same search. It then tunes the selected variant over the full 20 configurations. The extra screening fits reuse configurations from that budget.

The PR review reruns updated 336 arms and 1,248 arm-horizon outputs. The other horizons retained their saved predictions exactly. All 366 active CV arms passed the output audit. The reviewed feature policy reached the predeclared minimum gain in 20 of 20 LightGBM comparisons and 13 of 20 XGBoost comparisons. All 12 network-transfer comparisons remained below the minimum gain after correcting progress to the furthest reported stop.

Steps 6.5 and 4.2 use a separate Python 3.13 environment with [requirements-distributions.txt](requirements-distributions.txt). NGBoost and LightGBMLSS constrain dependencies to older releases than the main environment uses.

The LightGBMLSS adapter standardizes delay within each training fold and returns the Gaussian mixture in minutes. Fitting uses the package's Gumbel-softmax relaxation. Prediction uses softmax weights from the fitted logits, giving a mean and CDF independent of prediction batch order. [LightGBMLSS API reference](https://statmixedml.github.io/LightGBMLSS/api/)

Step 8.3 uses railway inputs from the three auxiliary corridors. The builder drops ambiguous shared departures, replacements and additional endpoint stops. Each fit admits only source runs whose origin, run start, planned destination and reported outcome weeks are eligible. The inner and outer fits also exclude validation journey IDs. Step 8.4 applies the same run exclusion and checks the planned, reported and cutoff weeks of its network pairs.

```bash
.venv/bin/python frankfurt_munich/advanced_modeling/build_corridor_inputs.py
.venv/bin/python frankfurt_munich/advanced_modeling/network_transfer.py
.venv/bin/python frankfurt_munich/advanced_modeling/sequence_inputs.py
```

### Holdout evaluation

This folder preserves the registry and results of the original holdout evaluation, run at commit `065b077`. The source has changed since registration (the review fixes, then the move into this folder on 8 November 2026), so the holdout builders' source check refuses to run with this registry. The PR review changed the development pipeline after that evaluation. The scores below belong to the original inputs and source, and the review reruns use CV only. The original hold-out inputs predate the review fixes. Two of those fixes could change the scores: the recent-reliability features (2.1) of the hold-out rows drew past journeys from both splits, while the CV rows used CV journeys only; and before a replacement was announced, `current_delay` and the delay dynamics (2.4) of a train that was later replaced were set to missing. For the original protocol and source fingerprints, use commit `065b077` and its recorded inputs. The original feature inputs are archived locally in `../data/advanced_original_065b077/`, and the corrected CV inputs are in `../data/advanced/`. The evaluator refuses to overwrite the completed results.

The registration can re-tune arms in parallel after shared feature selection. It records the tuning environment and freezes the source and selected CV input fingerprints. Holdout builders verify the registered source. Evaluation verifies the registered CV inputs before scoring and records each arm’s interpreter and package versions. On retry, the scorer reuses completed arms when the committed registration and both splits' fingerprints match.

After a forced stop, resume with the same registry once the scorer and its arm workers have exited and the stale `.evaluation.lock` has been cleared.

After the CV steps are complete, `run_final.py register` ranks eligible pipelines by the railway-only CV score and writes one matched pair per task and horizon to `finalists.json`. Commit that file with the preregistration entry in the plan before generating holdout inputs.

```bash
.venv/bin/python frankfurt_munich/advanced_modeling/run_final.py register --jobs 4
# After committing the registry and the preregistration entry
.venv/bin/python frankfurt_munich/advanced_modeling/scheduled_timetable.py --split holdout
.venv/bin/python frankfurt_munich/advanced_modeling/history_features.py --split holdout
.venv/bin/python frankfurt_munich/advanced_modeling/build_advanced_weather.py --split holdout
```

Build any additional inputs named by the finalists with `--split holdout`. Use `history_features.py --feature reliability` for recent reliability and `sequence_inputs.py` for upstream sequences. Then run `run_final.py evaluate` once. If a distributional learner is registered, supply its pinned interpreter via `--distribution-python`. The evaluation saves scores, paired week-bootstrap intervals, subgroup errors, calibration, risk trajectories, and SHAP summaries in `results/advanced_holdout/`.

#### Holdout results

The registry fixed 20 arms before the holdout labels were read. Each model was refitted on the 95 CV weeks and scored on the same rows as its matched arm across 23 held-out ISO weeks. LightGBM was selected for all severity horizons and for delay at 24 h and departure. Full-group linear regression was selected for delay at 6 h, 1 h and 20 min. The LightGBM pairs share railway features, and each arm was tuned separately over the same 20 configurations. [Finalist registry](finalists.json)

Weather minus railway is reported below, with negative deltas favoring weather. The pointwise 95% intervals use 2,000 paired resamples of whole ISO weeks. All ten intervals include zero. The weather arm had higher delay RMSE at all five horizons, by 0.017 to 0.193 min. Severity log loss changed most at 24 h, increasing by 0.00486. These results cover the held-out weeks on this corridor under the stratified week split. Route weather aggregates were present for 57–65% of severity rows across horizons. This retrospective comparison uses DWD observation hours ending by the cutoff and assumes publication at the labeled hour. Actual publication latency was unmeasured.

| Horizon | Rows | Railway log loss | Weather log loss | Δ log loss | 95% interval | Railway macro-F1 | Weather macro-F1 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 24 h | 4,315 | 0.91979 | 0.92466 | +0.00486 | [−0.00144, +0.01119] | 0.2210 | 0.2149 |
| 6 h | 4,298 | 0.90663 | 0.90572 | −0.00091 | [−0.00622, +0.00378] | 0.2535 | 0.2132 |
| 1 h | 4,243 | 0.86349 | 0.86411 | +0.00062 | [−0.00279, +0.00386] | 0.2645 | 0.2551 |
| 20 min | 4,210 | 0.83962 | 0.84073 | +0.00111 | [−0.00283, +0.00486] | 0.2650 | 0.2662 |
| Departure | 4,194 | 0.82026 | 0.82025 | −0.00001 | [−0.00322, +0.00348] | 0.2852 | 0.2681 |

| Horizon | Completed journeys | Railway RMSE (min) | Weather RMSE (min) | Δ RMSE (min) | 95% interval (min) |
| --- | ---: | ---: | ---: | ---: | --- |
| 24 h | 4,090 | 25.147 | 25.340 | +0.193 | [−0.006, +0.401] |
| 6 h | 4,090 | 25.027 | 25.103 | +0.076 | [−0.026, +0.173] |
| 1 h | 4,090 | 24.614 | 24.689 | +0.076 | [−0.016, +0.179] |
| 20 min | 4,090 | 24.486 | 24.597 | +0.111 | [−0.008, +0.238] |
| Departure | 4,090 | 24.272 | 24.288 | +0.017 | [−0.132, +0.172] |

At departure, weather lowered log loss for normal or minor delays by 0.00866 and raised it for severe delays by 0.02732 and cancellations by 0.01909. It raised RMSE for severe delays by 0.490 min. Departure severity log loss moved by less than 0.0006 in either period and in opposite directions on the Nürnberg and Stuttgart branches. The heavy-rain groups contain two to four rows per horizon, and 1,486 departure severity rows lack a branch assignment. These subgroup differences are descriptive. [Severity subgroup scores](results/advanced_holdout/severity_errors.csv), [delay subgroup scores](results/advanced_holdout/delay_errors.csv)

For severe delays and cancellations, the calibration bins at or above 0.5 have at most five observations per nonempty bin. Departure hour led the SHAP rankings for all ten severity arms. Planned travel time and service reliability also ranked highly. The runner uses TreeSHAP for LightGBM and sampled permutation SHAP for linear regression. [Reliability plots](results/advanced_holdout/calibration.svg), [SHAP summaries](results/advanced_holdout/shap.csv)

The risk trajectories show the earliest journey with all five horizons in each realized class. The railway arm's severe-or-cancelled probability for the severe example rose from 10.4% at 24 h to 18.5% at 6 h and was 10.1% at departure. The cancelled example stayed between 5.9% and 8.3% in the railway arm. [Risk trajectories](results/advanced_holdout/risk.svg)

<!-- BEGIN ADVANCED CV -->

### Advanced CV results

Scores use the five committed CV week folds. Positive paired differences favor the reference.

#### Step 0.7 reference scores

| Model | Search configs | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [class_shares](results/advanced_0_7_class_shares/20261105-094255) | fixed | log_loss | 0.968037 | 0.958148 | 0.923149 | 0.904638 | 0.896917 |
| [lightgbm](results/advanced_0_7_lightgbm/20261107-033224) | fixed | rmse | 24.475782 | 24.475782 | 24.081690 | 24.118938 | 23.888618 |
| [lightgbm_classifier](results/advanced_0_7_lightgbm_classifier/20261107-033350) | fixed | log_loss | 0.920441 | 0.911750 | 0.867661 | 0.849127 | 0.833639 |
| [lightgbm_classifier + weather](results/advanced_0_7_lightgbm_classifier_weather/20261107-033440) | fixed | log_loss | 0.926648 | 0.920599 | 0.871240 | 0.853052 | 0.838042 |
| [lightgbm + weather](results/advanced_0_7_lightgbm_weather/20261107-033232) | fixed | rmse | 24.596359 | 24.636556 | 24.168355 | 24.135646 | 23.894385 |
| [linear_regression](results/advanced_0_7_linear_regression/20261107-033128) | fixed | rmse | 24.725467 | 24.725467 | 24.318993 | 24.297336 | 24.088508 |
| [linear_regression + weather](results/advanced_0_7_linear_regression_weather/20261107-033133) | fixed | rmse | 24.741384 | 24.745659 | 24.342394 | 24.314366 | 24.104920 |
| [logistic_regression](results/advanced_0_7_logistic_regression/20261107-033140) | fixed | log_loss | 0.934693 | 0.925578 | 0.881839 | 0.862961 | 0.848789 |
| [logistic_regression + weather](results/advanced_0_7_logistic_regression_weather/20261107-033152) | fixed | log_loss | 0.936857 | 0.926824 | 0.881742 | 0.862812 | 0.849881 |
| [training_mean](results/advanced_0_7_training_mean/20261105-094254) | fixed | rmse | 25.427529 | 25.427529 | 25.427529 | 25.427529 | 25.427529 |
| [xgboost](results/advanced_0_7_xgboost/20261107-033408) | fixed | rmse | 24.511355 | 24.511355 | 24.100710 | 24.070339 | 23.826258 |
| [xgboost_classifier](results/advanced_0_7_xgboost_classifier/20261107-033502) | fixed | log_loss | 0.921781 | 0.912863 | 0.868254 | 0.851611 | 0.836714 |
| [xgboost_classifier + weather](results/advanced_0_7_xgboost_classifier_weather/20261107-033616) | fixed | log_loss | 0.923900 | 0.916941 | 0.869928 | 0.851140 | 0.839592 |
| [xgboost + weather](results/advanced_0_7_xgboost_weather/20261107-033419) | fixed | rmse | 24.516292 | 24.541405 | 24.143383 | 24.125756 | 23.894956 |

#### Step 0.8 nested reference scores

| Model | Search configs | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [lightgbm](results/advanced_0_8_lightgbm/20261107-033524) | 20 | rmse | 24.558833 | 24.558978 | 24.172574 | 24.157333 | 23.889677 |
| [lightgbm_classifier](results/advanced_0_8_lightgbm_classifier/20261107-023339) | 20 | log_loss | 0.927132 | 0.917266 | 0.870323 | 0.850577 | 0.837399 |
| [lightgbm_classifier + weather](results/advanced_0_8_lightgbm_classifier_weather/20261107-001456) | 20 | log_loss | 0.927180 | 0.919514 | 0.874485 | 0.853337 | 0.839383 |
| [lightgbm + weather](results/advanced_0_8_lightgbm_weather/20261107-025742) | 20 | rmse | 24.544852 | 24.667386 | 24.164635 | 24.122381 | 23.906505 |
| [xgboost](results/advanced_0_8_xgboost/20261107-030627) | 20 | rmse | 24.536433 | 24.568876 | 24.121687 | 24.108471 | 23.876457 |
| [xgboost_classifier](results/advanced_0_8_xgboost_classifier/20261107-010725) | 20 | log_loss | 0.923019 | 0.914149 | 0.871321 | 0.851552 | 0.837121 |
| [xgboost_classifier + weather](results/advanced_0_8_xgboost_classifier_weather/20261106-233925) | 20 | log_loss | 0.923900 | 0.917211 | 0.870934 | 0.851439 | 0.837825 |
| [xgboost + weather](results/advanced_0_8_xgboost_weather/20261107-023238) | 20 | rmse | 24.548736 | 24.610447 | 24.138429 | 24.125756 | 23.894956 |

#### Smallest paired differences by step, task and horizon

| step | task | evaluation | prediction_time | method | variant | weather | contrast | mean | folds_improved | reaches_minimum_gain | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | delay | exploratory | 0min | lightgbm | recorded_route | True |  | -0.147810 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261107-033241/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 1h | linear_regression | recorded_route | True |  | -0.127056 | 4 | True | [folds](results/advanced_0_7_linear_regression_weather_recorded_route/20261107-033136/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 20min | lightgbm | recorded_route | True |  | -0.157135 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261107-033241/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 24h | linear_regression | recorded_route | True |  | -0.148750 | 5 | True | [folds](results/advanced_0_7_linear_regression_weather_recorded_route/20261107-033136/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 6h | lightgbm | recorded_route | True |  | -0.158027 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261107-033241/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 0min | logistic_regression | recorded_route | True |  | -0.007220 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261107-033216/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 1h | logistic_regression | recorded_route | True |  | -0.006654 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261107-033216/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 20min | logistic_regression | recorded_route | True |  | -0.007132 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261107-033216/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 24h | logistic_regression | recorded_route | True |  | -0.007309 | 4 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261107-033216/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 6h | logistic_regression | recorded_route | False |  | -0.006960 | 4 | True | [folds](results/advanced_0_7_logistic_regression_recorded_route/20261107-033146/paired_vs_base.csv) |
| 0.8 | delay | nested | 0min | lightgbm | auto_groups | True |  | -0.324119 | 5 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261107-023926/paired_vs_base.csv) |
| 0.8 | delay | nested | 1h | xgboost | auto_groups | True |  | -0.349025 | 5 | True | [folds](results/advanced_0_8_xgboost_weather_auto_groups/20261107-022102/paired_vs_base.csv) |
| 0.8 | delay | nested | 20min | lightgbm | auto_groups | False |  | -0.389667 | 5 | True | [folds](results/advanced_0_8_lightgbm_auto_groups/20261107-025849/paired_vs_base.csv) |
| 0.8 | delay | nested | 24h | lightgbm | auto_groups | False |  | -0.164367 | 4 | True | [folds](results/advanced_0_8_lightgbm_auto_groups/20261107-025849/paired_vs_base.csv) |
| 0.8 | delay | nested | 6h | lightgbm | auto_groups | True |  | -0.348582 | 5 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261107-023926/paired_vs_base.csv) |
| 0.8 | severity | nested | 0min | lightgbm_classifier | auto_groups | False |  | -0.017925 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_auto_groups/20261107-002245/paired_vs_base.csv) |
| 0.8 | severity | nested | 1h | lightgbm_classifier | auto_groups | True |  | -0.014109 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261106-230056/paired_vs_base.csv) |
| 0.8 | severity | nested | 20min | lightgbm_classifier | auto_groups | True |  | -0.013091 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261106-230056/paired_vs_base.csv) |
| 0.8 | severity | nested | 24h | lightgbm_classifier | auto_groups | True |  | -0.007022 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261106-230056/paired_vs_base.csv) |
| 0.8 | severity | nested | 6h | lightgbm_classifier | auto_groups | True |  | -0.013428 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261106-230056/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 0min | lightgbm | calendar_cyclic | False |  | -0.037984 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261107-033516/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 1h | lightgbm | calendar_integer | False |  | -0.045120 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_integer/20261107-033455/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 20min | lightgbm | calendar_cyclic | False |  | -0.024768 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261107-033516/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 24h | linear_regression | calendar_combined | False |  | -0.000093 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261107-033526/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 6h | linear_regression | calendar_combined | False |  | -0.000093 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261107-033526/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 0min | lightgbm_classifier | calendar_cyclic | False |  | -0.001188 | 4 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261107-033614/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 1h | lightgbm_classifier | calendar_cyclic | False |  | -0.000705 | 2 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261107-033614/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 20min | lightgbm_classifier | calendar_cyclic | False |  | -0.000989 | 3 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261107-033614/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 24h | logistic_regression | calendar_combined | False |  | +0.000032 | 3 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261107-033530/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 6h | logistic_regression | calendar_combined | False |  | +0.000004 | 3 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261107-033530/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 0min | linear_regression | day_of_year | False |  | -0.039930 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261107-033554/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 1h | linear_regression | day_of_year | False |  | -0.047304 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261107-033554/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 20min | linear_regression | day_of_year | False |  | -0.046643 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261107-033554/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 24h | linear_regression | day_of_year | False |  | -0.050984 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261107-033554/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 6h | linear_regression | day_of_year | False |  | -0.050984 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261107-033554/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 0min | logistic_regression | day_of_year | False |  | -0.002239 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261107-033558/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 1h | logistic_regression | day_of_year | False |  | -0.002171 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261107-033558/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 20min | logistic_regression | day_of_year | False |  | -0.002555 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261107-033558/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 24h | lightgbm_classifier | minute_of_day | False |  | -0.001847 | 3 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261107-033704/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 6h | lightgbm_classifier | minute_of_day | False |  | -0.001775 | 4 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261107-033704/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 0min | lightgbm | route_onehot | False |  | +0.010854 | 2 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261107-033634/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 1h | lightgbm | route_onehot | False |  | +0.004087 | 3 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261107-033634/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 20min | lightgbm | route_onehot | False |  | +0.015612 | 1 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261107-033634/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 24h | lightgbm | route_onehot | False |  | -0.062585 | 5 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261107-033634/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 6h | lightgbm | route_onehot | False |  | -0.062585 | 5 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261107-033634/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 0min | lightgbm_classifier | route_onehot | False |  | -0.001219 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261107-033754/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 1h | lightgbm_classifier | route_onehot | False |  | -0.001403 | 5 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261107-033754/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 20min | lightgbm_classifier | route_onehot | False |  | -0.001167 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261107-033754/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 24h | lightgbm_classifier | route_onehot | False |  | -0.001732 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261107-033754/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 6h | lightgbm_classifier | route_onehot | False |  | -0.001492 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261107-033754/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 0min | linear_regression | delay_clipped | False |  | -0.000092 | 4 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261107-033656/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 1h | linear_regression | delay_clipped | False |  | -0.001157 | 4 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261107-033656/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 20min | linear_regression | delay_clipped | False |  | -0.002168 | 3 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261107-033656/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 0min | logistic_regression | delay_bins | False |  | -0.000402 | 2 | False | [folds](results/advanced_1_4_logistic_regression_delay_bins/20261107-033728/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 1h | lightgbm_classifier | delay_missing | False |  | +0.000033 | 2 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_missing/20261107-033857/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 20min | lightgbm_classifier | delay_log | False |  | -0.000110 | 3 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_log/20261107-033832/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 0min | lightgbm | weather_extremes | True |  | +0.026827 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261107-033755/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 1h | lightgbm | weather_extremes | True |  | -0.023172 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261107-033755/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 20min | lightgbm | weather_extremes | True |  | +0.006327 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261107-033755/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 24h | lightgbm | weather_extremes | True |  | -0.043542 | 3 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261107-033755/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 6h | lightgbm | weather_extremes | True |  | +0.021090 | 1 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261107-033755/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 0min | lightgbm_classifier | weather_extremes | True |  | -0.002243 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261107-033904/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 1h | lightgbm_classifier | weather_extremes | True |  | -0.002543 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261107-033904/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 20min | lightgbm_classifier | weather_extremes | True |  | -0.002058 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261107-033904/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 24h | lightgbm_classifier | weather_extremes | True |  | -0.003114 | 4 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261107-033904/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 6h | lightgbm_classifier | weather_extremes | True |  | -0.000532 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261107-033904/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 0min | linear_regression | recent_reliability | False |  | -0.430351 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261107-033810/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 1h | linear_regression | recent_reliability | False |  | -0.439518 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261107-033810/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 20min | linear_regression | recent_reliability | False |  | -0.442265 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261107-033810/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 24h | linear_regression | recent_reliability | False |  | -0.297709 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261107-033810/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 6h | linear_regression | recent_reliability | False |  | -0.449056 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261107-033810/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 0min | logistic_regression | recent_reliability | False |  | -0.017685 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261107-033817/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 1h | logistic_regression | recent_reliability | False |  | -0.016648 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261107-033817/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 20min | logistic_regression | recent_reliability | False |  | -0.017531 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261107-033817/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 24h | logistic_regression | recent_reliability | False |  | -0.010196 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261107-033817/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 6h | logistic_regression | recent_reliability | False |  | -0.016665 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261107-033817/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 0min | lightgbm | planned_timetable | False |  | -0.097774 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261107-033844/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 1h | lightgbm | planned_timetable | False |  | -0.066995 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261107-033844/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 20min | lightgbm | planned_timetable | False |  | -0.115012 | 4 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261107-033844/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 24h | lightgbm | planned_timetable | False |  | -0.087388 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261107-033844/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 6h | lightgbm | planned_timetable | False |  | -0.087388 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261107-033844/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 0min | logistic_regression | planned_timetable | False |  | -0.002883 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261107-033832/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 1h | logistic_regression | planned_timetable | False |  | -0.002699 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261107-033832/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 20min | logistic_regression | planned_timetable | False |  | -0.002085 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261107-033832/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 24h | lightgbm_classifier | planned_timetable | False |  | -0.002021 | 3 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261107-033955/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 6h | lightgbm_classifier | planned_timetable | False |  | -0.001653 | 3 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261107-033955/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 0min | linear_regression | holidays | False |  | -0.005169 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261107-033844/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 1h | linear_regression | holidays | False |  | -0.010886 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261107-033844/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 20min | linear_regression | holidays | False |  | -0.008327 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261107-033844/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 24h | linear_regression | holidays | False |  | -0.014086 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261107-033844/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 6h | linear_regression | holidays | False |  | -0.014086 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261107-033844/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 0min | lightgbm_classifier | holidays | False |  | +0.000685 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261107-034057/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 1h | lightgbm_classifier | holidays | False |  | +0.000735 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261107-034057/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 20min | lightgbm_classifier | holidays | False |  | +0.001124 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261107-034057/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 24h | lightgbm_classifier | holidays | False |  | -0.000403 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261107-034057/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 6h | lightgbm_classifier | holidays | False |  | -0.000039 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261107-034057/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 0min | linear_regression | delay_dynamics | False |  | -0.082292 | 5 | False | [folds](results/advanced_2_4_linear_regression_delay_dynamics/20261107-033853/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 1h | linear_regression | delay_dynamics | False |  | -0.066352 | 5 | False | [folds](results/advanced_2_4_linear_regression_delay_dynamics/20261107-033853/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 20min | linear_regression | delay_dynamics | False |  | -0.104126 | 5 | True | [folds](results/advanced_2_4_linear_regression_delay_dynamics/20261107-033853/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 0min | logistic_regression | delay_dynamics | False |  | -0.003286 | 5 | True | [folds](results/advanced_2_4_logistic_regression_delay_dynamics/20261107-033859/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 1h | lightgbm_classifier | delay_dynamics | False |  | -0.002003 | 4 | False | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261107-034026/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 20min | logistic_regression | delay_dynamics | False |  | -0.002967 | 5 | False | [folds](results/advanced_2_4_logistic_regression_delay_dynamics/20261107-033859/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 0min | lightgbm | network_state | False |  | -0.010448 | 3 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261107-033921/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 1h | lightgbm | network_state | False |  | -0.019942 | 3 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261107-033921/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 20min | lightgbm | network_state | False |  | -0.026644 | 4 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261107-033921/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 24h | lightgbm | network_state | False |  | -0.003228 | 2 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261107-033921/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 6h | lightgbm | network_state | False |  | -0.013284 | 2 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261107-033921/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 0min | lightgbm_classifier | network_state | False |  | -0.001090 | 4 | False | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261107-034038/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 1h | lightgbm_classifier | network_state | False |  | -0.000176 | 3 | False | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261107-034038/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 20min | logistic_regression | network_state | False |  | -0.000185 | 3 | False | [folds](results/advanced_2_5_logistic_regression_network_state/20261107-033910/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 24h | logistic_regression | network_state | False |  | +0.000991 | 0 | False | [folds](results/advanced_2_5_logistic_regression_network_state/20261107-033910/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 6h | logistic_regression | network_state | False |  | +0.000640 | 1 | False | [folds](results/advanced_2_5_logistic_regression_network_state/20261107-033910/paired_vs_base.csv) |
| 2.all | delay | exploratory | 0min | linear_regression | all_groups | False |  | -0.485662 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261107-033916/paired_vs_base.csv) |
| 2.all | delay | exploratory | 1h | linear_regression | all_groups | False |  | -0.470487 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261107-033916/paired_vs_base.csv) |
| 2.all | delay | exploratory | 20min | linear_regression | all_groups | False |  | -0.508984 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261107-033916/paired_vs_base.csv) |
| 2.all | delay | exploratory | 24h | linear_regression | all_groups | False |  | -0.282641 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261107-033916/paired_vs_base.csv) |
| 2.all | delay | exploratory | 6h | linear_regression | all_groups | False |  | -0.424032 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261107-033916/paired_vs_base.csv) |
| 2.all | severity | exploratory | 0min | logistic_regression | all_groups | False |  | -0.018292 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261107-033948/paired_vs_base.csv) |
| 2.all | severity | exploratory | 1h | logistic_regression | all_groups | False |  | -0.015720 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261107-033948/paired_vs_base.csv) |
| 2.all | severity | exploratory | 20min | logistic_regression | all_groups | False |  | -0.016837 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261107-033948/paired_vs_base.csv) |
| 2.all | severity | exploratory | 24h | logistic_regression | all_groups | False |  | -0.006283 | 4 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261107-033948/paired_vs_base.csv) |
| 2.all | severity | exploratory | 6h | logistic_regression | all_groups | False |  | -0.013541 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261107-033948/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | lightgbm | drop_wind | True | full | -0.022209 | 4 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_wind/20261107-034327/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | linear_regression | add_recent_reliability | True | timetable-only | -0.909381 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261107-034008/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | linear_regression | drop_wind | True | full | -0.021573 | 3 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_wind/20261107-034020/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | linear_regression | add_recent_reliability | True | timetable-only | -0.902626 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261107-034008/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | linear_regression | drop_wind | True | full | -0.023254 | 3 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_wind/20261107-034020/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | linear_regression | add_recent_reliability | True | timetable-only | -0.909433 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261107-034008/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | linear_regression | drop_temperature | True | full | -0.030112 | 4 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_temperature/20261107-034019/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | linear_regression | add_recent_reliability | True | timetable-only | -0.638619 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261107-034008/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | drop_temperature | True | full | -0.034591 | 5 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_temperature/20261107-034019/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | add_recent_reliability | True | timetable-only | -0.836658 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261107-034008/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | drop_wind | True | full | -0.001237 | 2 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_wind/20261107-034243/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | add_recent_reliability | True | timetable-only | -0.035330 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261107-034046/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | drop_precipitation | True | full | -0.000706 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_precipitation/20261107-034236/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | add_recent_reliability | True | timetable-only | -0.034265 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261107-034046/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | drop_wind | True | full | -0.001052 | 2 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_wind/20261107-034243/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | add_recent_reliability | True | timetable-only | -0.034702 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261107-034046/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | drop_wind | True | full | -0.001457 | 5 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_wind/20261107-034243/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | add_recent_reliability | True | timetable-only | -0.024544 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261107-034046/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | drop_temperature | True | full | -0.001276 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_temperature/20261107-034234/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | add_recent_reliability | True | timetable-only | -0.033200 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261107-034046/paired_vs_base.csv) |
| 3.3 | delay | nested | 0min | linear_regression | backward | False |  | +0.035562 | 0 | False | [folds](results/advanced_3_3_linear_regression_backward/20261107-034938/paired_vs_base.csv) |
| 3.3 | delay | nested | 1h | linear_regression | backward | False |  | +0.037007 | 0 | False | [folds](results/advanced_3_3_linear_regression_backward/20261107-034938/paired_vs_base.csv) |
| 3.3 | delay | nested | 20min | linear_regression | forward | True |  | -0.011949 | 2 | False | [folds](results/advanced_3_3_linear_regression_weather_forward/20261107-035353/paired_vs_base.csv) |
| 3.3 | delay | nested | 24h | linear_regression | backward | True |  | -0.048137 | 3 | False | [folds](results/advanced_3_3_linear_regression_weather_backward/20261107-035712/paired_vs_base.csv) |
| 3.3 | delay | nested | 6h | linear_regression | backward | True |  | +0.011571 | 2 | False | [folds](results/advanced_3_3_linear_regression_weather_backward/20261107-035712/paired_vs_base.csv) |
| 3.3 | severity | nested | 0min | logistic_regression | forward | True |  | +0.000315 | 2 | False | [folds](results/advanced_3_3_logistic_regression_weather_forward/20261107-040752/paired_vs_base.csv) |
| 3.3 | severity | nested | 1h | logistic_regression | backward | True |  | +0.000726 | 2 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261107-042415/paired_vs_base.csv) |
| 3.3 | severity | nested | 20min | logistic_regression | backward | True |  | +0.001037 | 1 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261107-042415/paired_vs_base.csv) |
| 3.3 | severity | nested | 24h | logistic_regression | forward | True |  | -0.000349 | 2 | False | [folds](results/advanced_3_3_logistic_regression_weather_forward/20261107-040752/paired_vs_base.csv) |
| 3.3 | severity | nested | 6h | logistic_regression | backward | True |  | +0.000863 | 2 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261107-042415/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 0min | logistic_regression | l1 | True |  | -0.003448 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261107-042331/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 1h | logistic_regression | l1 | True |  | -0.003512 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261107-042331/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 20min | logistic_regression | l1 | True |  | -0.003651 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261107-042331/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 24h | logistic_regression | l1 | True |  | -0.003417 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261107-042331/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 6h | logistic_regression | l1 | True |  | -0.003738 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261107-042331/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 0min | lightgbm | temperature | True |  | -0.042357 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261107-035739/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 1h | lightgbm | temperature | True |  | -0.040966 | 5 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261107-035739/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 20min | lightgbm | temperature | True |  | -0.045709 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261107-035739/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 24h | lightgbm | snow | True |  | -0.027227 | 3 | False | [folds](results/advanced_3_4_lightgbm_weather_snow/20261107-035759/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 6h | lightgbm | temperature | True |  | -0.010874 | 3 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261107-035739/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 0min | lightgbm_classifier | temperature | True |  | -0.001279 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261107-040123/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 1h | lightgbm_classifier | temperature | True |  | -0.000927 | 5 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261107-040123/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 20min | lightgbm_classifier | temperature | True |  | -0.001003 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261107-040123/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 24h | lightgbm_classifier | temperature | True |  | +0.000366 | 2 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261107-040123/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 6h | lightgbm_classifier | temperature | True |  | -0.000484 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261107-040123/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_class | True |  | +0.000215 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261107-040548/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_class | True |  | -0.000837 | 3 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261107-040548/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_class | False |  | -0.000278 | 3 | False | [folds](results/advanced_4_1_lightgbm_classifier_cancellation_then_class/20261107-040405/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_class | True |  | -0.000007 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261107-040548/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_class | True |  | +0.000206 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261107-040548/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 0min | quantile_forest_severity |  | True |  | +0.029828 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261107-041736/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 1h | quantile_forest_severity |  | True |  | +0.026506 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261107-041736/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 20min | quantile_forest_severity |  | True |  | +0.039486 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261107-041736/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 24h | quantile_forest_severity |  | True |  | +0.042189 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261107-041736/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 6h | quantile_forest_severity |  | True |  | +0.022049 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261107-041736/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_ordinal | False |  | +0.001998 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_cancellation_then_ordinal/20261107-040516/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_ordinal | True |  | -0.000408 | 3 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261107-040604/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_ordinal | False |  | +0.001251 | 1 | False | [folds](results/advanced_4_3_lightgbm_classifier_cancellation_then_ordinal/20261107-040516/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.000793 | 2 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261107-040604/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.001689 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261107-040604/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 0min | lightgbm | mixture_of_experts | True |  | -0.026326 | 3 | False | [folds](results/advanced_4_4_lightgbm_weather_mixture_of_experts/20261107-040643/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 1h | lightgbm | mixture_of_experts | False |  | -0.047743 | 4 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261107-040602/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 20min | lightgbm | mixture_of_experts | True |  | -0.046101 | 4 | False | [folds](results/advanced_4_4_lightgbm_weather_mixture_of_experts/20261107-040643/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 24h | lightgbm | mixture_of_experts | False |  | -0.027983 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261107-040602/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 6h | lightgbm | mixture_of_experts | False |  | -0.027983 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261107-040602/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 0min | lightgbm | change_then_level | False |  | +0.028198 | 1 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261107-040616/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 1h | lightgbm | change_then_level | False |  | +0.015311 | 2 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261107-040616/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 20min | lightgbm | change_then_level | False |  | +0.049355 | 2 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261107-040616/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 0min | lightgbm | plus_neural | True |  | -0.059409 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261107-062016/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 1h | lightgbm | plus_neural | True |  | -0.080313 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261107-062016/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 20min | lightgbm | plus_catboost | True |  | -0.046848 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_catboost/20261107-062015/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 24h | lightgbm | plus_neural | True |  | -0.025077 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261107-062016/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 6h | lightgbm | plus_neural | True |  | -0.045861 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261107-062016/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 0min | lightgbm_classifier | plus_neural | True |  | -0.002057 | 2 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_neural/20261107-062017/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 1h | lightgbm_classifier | plus_catboost | True |  | -0.001713 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261107-062017/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 20min | lightgbm_classifier | plus_catboost | True |  | -0.002147 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261107-062017/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 24h | lightgbm_classifier | mean | True |  | -0.000810 | 3 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_mean/20261107-062016/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 6h | lightgbm_classifier | plus_neural | True |  | -0.001334 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_neural/20261107-062017/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 0min | xgboost | top5_flat | True |  | -0.071664 | 5 | False | [folds](results/advanced_5_2_xgboost_weather_top5_flat/20261107-042124/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 1h | lightgbm | top5_flat | True |  | -0.072463 | 5 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_flat/20261107-041848/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 20min | lightgbm | top5_flat | False |  | -0.064965 | 5 | False | [folds](results/advanced_5_2_lightgbm_top5_flat/20261107-041300/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 24h | xgboost | top5_flat | False |  | -0.074564 | 4 | False | [folds](results/advanced_5_2_xgboost_top5_flat/20261107-041618/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 6h | lightgbm | top5_flat | True |  | -0.103264 | 5 | True | [folds](results/advanced_5_2_lightgbm_weather_top5_flat/20261107-041848/paired_vs_base.csv) |
| 5.2 | delay | nested | 0min | lightgbm | top5_nested | False |  | -0.097227 | 5 | False | [folds](results/advanced_5_2_lightgbm_top5_nested/20261107-054311/paired_vs_base.csv) |
| 5.2 | delay | nested | 1h | lightgbm | top5_nested | False |  | -0.076237 | 4 | False | [folds](results/advanced_5_2_lightgbm_top5_nested/20261107-054311/paired_vs_base.csv) |
| 5.2 | delay | nested | 20min | lightgbm | top5_nested | False |  | -0.091889 | 5 | False | [folds](results/advanced_5_2_lightgbm_top5_nested/20261107-054311/paired_vs_base.csv) |
| 5.2 | delay | nested | 24h | xgboost | top5_nested | True |  | -0.052861 | 3 | False | [folds](results/advanced_5_2_xgboost_weather_top5_nested/20261107-054557/paired_vs_base.csv) |
| 5.2 | delay | nested | 6h | xgboost | top5_nested | True |  | -0.102523 | 5 | True | [folds](results/advanced_5_2_xgboost_weather_top5_nested/20261107-054557/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 0min | xgboost_classifier | top5_flat | True |  | -0.003673 | 5 | True | [folds](results/advanced_5_2_xgboost_classifier_weather_top5_flat/20261107-042949/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 1h | lightgbm_classifier | top5_flat | False |  | -0.001733 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_flat/20261107-041622/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 20min | xgboost_classifier | top5_flat | False |  | -0.003228 | 5 | True | [folds](results/advanced_5_2_xgboost_classifier_top5_flat/20261107-042103/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 24h | lightgbm_classifier | top5_flat | True |  | -0.002975 | 4 | False | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_flat/20261107-042401/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 6h | lightgbm_classifier | top5_flat | True |  | -0.003670 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_flat/20261107-042401/paired_vs_base.csv) |
| 5.2 | severity | nested | 0min | lightgbm_classifier | top5_nested | False |  | -0.004471 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261107-054354/paired_vs_base.csv) |
| 5.2 | severity | nested | 1h | xgboost_classifier | top5_nested | False |  | -0.002685 | 5 | False | [folds](results/advanced_5_2_xgboost_classifier_top5_nested/20261107-054753/paired_vs_base.csv) |
| 5.2 | severity | nested | 20min | lightgbm_classifier | top5_nested | False |  | -0.003004 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261107-054354/paired_vs_base.csv) |
| 5.2 | severity | nested | 24h | lightgbm_classifier | top5_nested | False |  | -0.004607 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261107-054354/paired_vs_base.csv) |
| 5.2 | severity | nested | 6h | lightgbm_classifier | top5_nested | False |  | -0.002238 | 4 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261107-054354/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 0min | lightgbm | greedy | True |  | -0.032853 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261107-042622/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 1h | lightgbm | greedy | True |  | -0.069353 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261107-042622/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 20min | lightgbm | greedy | True |  | -0.018224 | 5 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261107-042622/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 24h | lightgbm | greedy | True |  | +0.019919 | 2 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261107-042622/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 6h | lightgbm | greedy | True |  | +0.015918 | 1 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261107-042622/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 0min | lightgbm_classifier | greedy | True |  | -0.001681 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261107-044000/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 1h | lightgbm_classifier | greedy | True |  | -0.000871 | 3 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261107-044000/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 20min | lightgbm_classifier | greedy | True |  | -0.001721 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261107-044000/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 24h | lightgbm_classifier | greedy | True |  | -0.000346 | 3 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261107-044000/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 6h | lightgbm_classifier | greedy | True |  | +0.000391 | 1 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261107-044000/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 0min | lightgbm | stack | True |  | -0.037618 | 4 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261107-042905/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 1h | lightgbm | stack | True |  | -0.081735 | 4 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261107-042905/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 20min | lightgbm | stack | True |  | -0.034046 | 5 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261107-042905/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 24h | lightgbm | stack | True |  | +0.033195 | 0 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261107-042905/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 6h | lightgbm | stack | True |  | +0.014701 | 2 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261107-042905/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 0min | lightgbm_classifier | stack | True |  | +0.000892 | 1 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261107-044239/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 1h | lightgbm_classifier | stack | False |  | +0.001846 | 2 | False | [folds](results/advanced_5_4_lightgbm_classifier_stack/20261107-043458/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 20min | lightgbm_classifier | stack | True |  | +0.000830 | 1 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261107-044239/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 24h | lightgbm_classifier | stack | True |  | +0.002962 | 1 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261107-044239/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 6h | lightgbm_classifier | stack | True |  | +0.002317 | 0 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261107-044239/paired_vs_base.csv) |
| 6.1 | delay | nested | 0min | catboost |  | False |  | -0.014213 | 3 | False | [folds](results/advanced_6_1_catboost/20261107-004133/paired_vs_base.csv) |
| 6.1 | delay | nested | 1h | catboost |  | False |  | -0.008175 | 2 | False | [folds](results/advanced_6_1_catboost/20261107-004133/paired_vs_base.csv) |
| 6.1 | delay | nested | 20min | catboost |  | False |  | +0.003098 | 2 | False | [folds](results/advanced_6_1_catboost/20261107-004133/paired_vs_base.csv) |
| 6.1 | delay | nested | 24h | catboost |  | True |  | +0.009378 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261107-001301/paired_vs_base.csv) |
| 6.1 | delay | nested | 6h | catboost |  | True |  | -0.076428 | 4 | False | [folds](results/advanced_6_1_catboost_weather/20261107-001301/paired_vs_base.csv) |
| 6.1 | severity | nested | 0min | catboost_classifier |  | True |  | +0.001941 | 0 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-225306/paired_vs_base.csv) |
| 6.1 | severity | nested | 1h | catboost_classifier |  | True |  | -0.000446 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-225306/paired_vs_base.csv) |
| 6.1 | severity | nested | 20min | catboost_classifier |  | True |  | +0.000639 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-225306/paired_vs_base.csv) |
| 6.1 | severity | nested | 24h | catboost_classifier |  | True |  | +0.000336 | 1 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-225306/paired_vs_base.csv) |
| 6.1 | severity | nested | 6h | catboost_classifier |  | True |  | +0.000928 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-225306/paired_vs_base.csv) |
| 6.2 | severity | nested | 0min | catboost_classifier | temperature | True |  | -0.000467 | 4 | False | [folds](results/advanced_6_2_catboost_classifier_weather_temperature/20261107-061916/paired_vs_base.csv) |
| 6.2 | severity | nested | 1h | xgboost_classifier | temperature | False |  | -0.000392 | 4 | False | [folds](results/advanced_6_2_xgboost_classifier_temperature/20261107-054908/paired_vs_base.csv) |
| 6.2 | severity | nested | 20min | xgboost_classifier | temperature | False |  | -0.000413 | 3 | False | [folds](results/advanced_6_2_xgboost_classifier_temperature/20261107-054908/paired_vs_base.csv) |
| 6.2 | severity | nested | 24h | catboost_classifier | temperature | False |  | -0.000285 | 4 | False | [folds](results/advanced_6_2_catboost_classifier_temperature/20261107-060004/paired_vs_base.csv) |
| 6.2 | severity | nested | 6h | catboost_classifier | temperature | False |  | -0.000304 | 5 | False | [folds](results/advanced_6_2_catboost_classifier_temperature/20261107-060004/paired_vs_base.csv) |
| 6.3 | delay | nested | 0min | ebm |  | False |  | +0.200796 | 0 | False | [folds](results/advanced_6_3_ebm/20261107-042953/paired_vs_base.csv) |
| 6.3 | delay | nested | 1h | ebm |  | False |  | +0.098728 | 1 | False | [folds](results/advanced_6_3_ebm/20261107-042953/paired_vs_base.csv) |
| 6.3 | delay | nested | 20min | ebm |  | False |  | +0.176091 | 1 | False | [folds](results/advanced_6_3_ebm/20261107-042953/paired_vs_base.csv) |
| 6.3 | delay | nested | 24h | ebm |  | False |  | +0.087528 | 2 | False | [folds](results/advanced_6_3_ebm/20261107-042953/paired_vs_base.csv) |
| 6.3 | delay | nested | 6h | ebm |  | False |  | +0.087383 | 2 | False | [folds](results/advanced_6_3_ebm/20261107-042953/paired_vs_base.csv) |
| 6.3 | severity | nested | 0min | ebm_classifier |  | False |  | +0.011346 | 1 | False | [folds](results/advanced_6_3_ebm_classifier/20261107-043320/paired_vs_base.csv) |
| 6.3 | severity | nested | 1h | ebm_classifier |  | False |  | +0.010845 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261107-043320/paired_vs_base.csv) |
| 6.3 | severity | nested | 20min | ebm_classifier |  | False |  | +0.013835 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261107-043320/paired_vs_base.csv) |
| 6.3 | severity | nested | 24h | ebm_classifier |  | False |  | +0.006941 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261107-043320/paired_vs_base.csv) |
| 6.3 | severity | nested | 6h | ebm_classifier |  | False |  | +0.007665 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261107-043320/paired_vs_base.csv) |
| 6.5 | delay | nested | 0min | ngboost |  | False |  | +0.070853 | 1 | False | [folds](results/advanced_6_5_ngboost/20261107-043457/paired_vs_base.csv) |
| 6.5 | delay | nested | 1h | ngboost |  | False |  | +0.027578 | 3 | False | [folds](results/advanced_6_5_ngboost/20261107-043457/paired_vs_base.csv) |
| 6.5 | delay | nested | 20min | ngboost |  | False |  | +0.035588 | 2 | False | [folds](results/advanced_6_5_ngboost/20261107-043457/paired_vs_base.csv) |
| 6.5 | delay | nested | 24h | ngboost |  | False |  | +0.038672 | 2 | False | [folds](results/advanced_6_5_ngboost/20261107-043457/paired_vs_base.csv) |
| 6.5 | delay | nested | 6h | ngboost |  | True |  | +0.019298 | 2 | False | [folds](results/advanced_6_5_ngboost_weather/20261107-044048/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 0min | lightgbm | log_shift | True |  | +0.114453 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261107-043550/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 1h | lightgbm | log_shift | True |  | +0.105628 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261107-043550/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 20min | lightgbm | log_shift | True |  | +0.090547 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261107-043550/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 24h | lightgbm | log_shift | True |  | +0.131838 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261107-043550/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 6h | lightgbm | log_shift | False |  | +0.218975 | 0 | False | [folds](results/advanced_6_6_lightgbm_log_shift/20261107-043542/paired_vs_base.csv) |
| 7.1 | delay | nested | 0min | mlp |  | False |  | +0.486034 | 0 | False | [folds](results/advanced_7_1_mlp/20261107-033342/paired_vs_base.csv) |
| 7.1 | delay | nested | 1h | mlp |  | False |  | +0.383368 | 0 | False | [folds](results/advanced_7_1_mlp/20261107-033342/paired_vs_base.csv) |
| 7.1 | delay | nested | 20min | mlp |  | False |  | +0.458559 | 0 | False | [folds](results/advanced_7_1_mlp/20261107-033342/paired_vs_base.csv) |
| 7.1 | delay | nested | 24h | mlp |  | False |  | +0.531205 | 0 | False | [folds](results/advanced_7_1_mlp/20261107-033342/paired_vs_base.csv) |
| 7.1 | delay | nested | 6h | mlp |  | True |  | +0.459027 | 1 | False | [folds](results/advanced_7_1_mlp_weather/20261107-033343/paired_vs_base.csv) |
| 7.1 | severity | nested | 0min | mlp_classifier |  | False |  | +0.014631 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261107-033219/paired_vs_base.csv) |
| 7.1 | severity | nested | 1h | mlp_classifier |  | True |  | +0.014438 | 0 | False | [folds](results/advanced_7_1_mlp_classifier_weather/20261107-032548/paired_vs_base.csv) |
| 7.1 | severity | nested | 20min | mlp_classifier |  | False |  | +0.020193 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261107-033219/paired_vs_base.csv) |
| 7.1 | severity | nested | 24h | mlp_classifier |  | False |  | +0.016402 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261107-033219/paired_vs_base.csv) |
| 7.1 | severity | nested | 6h | mlp_classifier |  | True |  | +0.012044 | 1 | False | [folds](results/advanced_7_1_mlp_classifier_weather/20261107-032548/paired_vs_base.csv) |
| 7.2 | delay | nested | 0min | multitask |  | False |  | -0.077622 | 3 | False | [folds](results/advanced_7_2_multitask/20261107-033027/paired_vs_base.csv) |
| 7.2 | delay | nested | 1h | multitask |  | True |  | -0.196541 | 4 | True | [folds](results/advanced_7_2_multitask_weather/20261107-032029/paired_vs_base.csv) |
| 7.2 | delay | nested | 20min | multitask |  | True |  | -0.081195 | 3 | False | [folds](results/advanced_7_2_multitask_weather/20261107-032029/paired_vs_base.csv) |
| 7.2 | delay | nested | 24h | multitask |  | False |  | -0.136672 | 4 | True | [folds](results/advanced_7_2_multitask/20261107-033027/paired_vs_base.csv) |
| 7.2 | delay | nested | 6h | multitask |  | False |  | -0.171043 | 4 | True | [folds](results/advanced_7_2_multitask/20261107-033027/paired_vs_base.csv) |
| 7.2 | severity | nested | 0min | multitask_classifier |  | False |  | +0.002917 | 2 | False | [folds](results/advanced_7_2_multitask_classifier/20261107-031335/paired_vs_base.csv) |
| 7.2 | severity | nested | 1h | multitask_classifier |  | False |  | -0.003813 | 4 | True | [folds](results/advanced_7_2_multitask_classifier/20261107-031335/paired_vs_base.csv) |
| 7.2 | severity | nested | 20min | multitask_classifier |  | True |  | -0.005371 | 2 | True | [folds](results/advanced_7_2_multitask_classifier_weather/20261107-030856/paired_vs_base.csv) |
| 7.2 | severity | nested | 24h | multitask_classifier |  | False |  | -0.001112 | 3 | False | [folds](results/advanced_7_2_multitask_classifier/20261107-031335/paired_vs_base.csv) |
| 7.2 | severity | nested | 6h | multitask_classifier |  | False |  | -0.001310 | 3 | False | [folds](results/advanced_7_2_multitask_classifier/20261107-031335/paired_vs_base.csv) |
| 7.3 | delay | nested | 0min | tabgns |  | True |  | -0.298751 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261107-054244/paired_vs_base.csv) |
| 7.3 | delay | nested | 1h | tabgns |  | True |  | -0.539986 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261107-054244/paired_vs_base.csv) |
| 7.3 | delay | nested | 20min | tabgns |  | False |  | -0.274791 | 4 | True | [folds](results/advanced_7_3_tabgns/20261107-054331/paired_vs_base.csv) |
| 7.3 | delay | nested | 24h | tabgns |  | True |  | -0.424493 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261107-054244/paired_vs_base.csv) |
| 7.3 | delay | nested | 6h | tabgns |  | False |  | -0.390689 | 4 | True | [folds](results/advanced_7_3_tabgns/20261107-054331/paired_vs_base.csv) |
| 7.3 | severity | nested | 0min | tabgns_classifier |  | False |  | -0.003752 | 4 | True | [folds](results/advanced_7_3_tabgns_classifier/20261107-054519/paired_vs_base.csv) |
| 7.3 | severity | nested | 1h | tabgns_classifier |  | False |  | -0.001970 | 3 | False | [folds](results/advanced_7_3_tabgns_classifier/20261107-054519/paired_vs_base.csv) |
| 7.3 | severity | nested | 20min | tabgns_classifier |  | True |  | -0.010955 | 4 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261107-054523/paired_vs_base.csv) |
| 7.3 | severity | nested | 24h | tabgns_classifier |  | True |  | -0.004528 | 4 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261107-054523/paired_vs_base.csv) |
| 7.3 | severity | nested | 6h | tabgns_classifier |  | True |  | -0.000144 | 4 | False | [folds](results/advanced_7_3_tabgns_classifier_weather/20261107-054523/paired_vs_base.csv) |
| 7.4 | delay | nested | 0min | realmlp |  | True |  | -0.208790 | 4 | True | [folds](results/advanced_7_4_realmlp_weather/20261107-044548/paired_vs_base.csv) |
| 7.4 | delay | nested | 1h | realmlp |  | True |  | -0.563915 | 4 | True | [folds](results/advanced_7_4_realmlp_weather/20261107-044548/paired_vs_base.csv) |
| 7.4 | delay | nested | 20min | realmlp |  | False |  | -0.124496 | 3 | True | [folds](results/advanced_7_4_realmlp/20261107-044430/paired_vs_base.csv) |
| 7.4 | delay | nested | 24h | tabm |  | True |  | -0.499979 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261107-051659/paired_vs_base.csv) |
| 7.4 | delay | nested | 6h | tabm |  | True |  | -0.427632 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261107-051659/paired_vs_base.csv) |
| 7.4 | severity | nested | 0min | realmlp_classifier |  | True |  | +0.002574 | 3 | False | [folds](results/advanced_7_4_realmlp_classifier_weather/20261107-044651/paired_vs_base.csv) |
| 7.4 | severity | nested | 1h | realmlp_classifier |  | False |  | -0.000404 | 2 | False | [folds](results/advanced_7_4_realmlp_classifier/20261107-044408/paired_vs_base.csv) |
| 7.4 | severity | nested | 20min | tabm_classifier |  | True |  | -0.006785 | 2 | True | [folds](results/advanced_7_4_tabm_classifier_weather/20261107-053327/paired_vs_base.csv) |
| 7.4 | severity | nested | 24h | realmlp_classifier |  | True |  | -0.007390 | 3 | True | [folds](results/advanced_7_4_realmlp_classifier_weather/20261107-044651/paired_vs_base.csv) |
| 7.4 | severity | nested | 6h | realmlp_classifier |  | False |  | +0.000358 | 4 | False | [folds](results/advanced_7_4_realmlp_classifier/20261107-044408/paired_vs_base.csv) |
| 7.5 | delay | nested | 0min | sequence_gru |  | True |  | +0.096420 | 2 | False | [folds](results/advanced_7_5_sequence_gru_weather/20261107-023324/paired_vs_base.csv) |
| 7.5 | delay | nested | 1h | sequence_gru |  | True |  | +0.024262 | 2 | False | [folds](results/advanced_7_5_sequence_gru_weather/20261107-023324/paired_vs_base.csv) |
| 7.5 | delay | nested | 20min | sequence_gru |  | True |  | +0.106878 | 2 | False | [folds](results/advanced_7_5_sequence_gru_weather/20261107-023324/paired_vs_base.csv) |
| 7.5 | severity | nested | 0min | sequence_gru_classifier |  | True |  | -0.000333 | 3 | False | [folds](results/advanced_7_5_sequence_gru_classifier_weather/20261107-021453/paired_vs_base.csv) |
| 7.5 | severity | nested | 1h | sequence_gru_classifier |  | True |  | +0.000391 | 2 | False | [folds](results/advanced_7_5_sequence_gru_classifier_weather/20261107-021453/paired_vs_base.csv) |
| 7.5 | severity | nested | 20min | sequence_gru_classifier |  | True |  | -0.009078 | 3 | True | [folds](results/advanced_7_5_sequence_gru_classifier_weather/20261107-021453/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 0min | lightgbm |  | False |  | -0.029674 | 4 | False | [folds](results/advanced_8_2_lightgbm_pooled/20261107-044435/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 1h | lightgbm |  | True |  | -0.072234 | 5 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261107-044449/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 20min | lightgbm |  | True |  | -0.065466 | 5 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261107-044449/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 24h | lightgbm |  | True |  | -0.057443 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261107-044449/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 6h | lightgbm |  | True |  | -0.137448 | 5 | True | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261107-044449/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 0min | lightgbm_classifier |  | True |  | -0.003362 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261107-044642/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 1h | lightgbm_classifier |  | True |  | -0.003306 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261107-044642/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 20min | lightgbm_classifier |  | True |  | -0.005288 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261107-044642/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 24h | lightgbm_classifier |  | True |  | -0.004244 | 4 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261107-044642/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 6h | lightgbm_classifier |  | True |  | -0.006102 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261107-044642/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 0min | lightgbm | init_model | False |  | -0.011737 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261107-044719/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 1h | lightgbm | init_model | False |  | +0.009429 | 3 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261107-044719/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 20min | lightgbm | init_model | False |  | -0.014767 | 3 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261107-044719/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 24h | lightgbm | stacked | False |  | +0.031896 | 1 | False | [folds](results/advanced_8_3_lightgbm_stacked/20261107-044849/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 6h | lightgbm | stacked | False |  | +0.036389 | 2 | False | [folds](results/advanced_8_3_lightgbm_stacked/20261107-044849/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 0min | lightgbm_classifier | init_model | False |  | +0.001790 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261107-045330/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 1h | lightgbm_classifier | init_model | False |  | +0.001403 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261107-045330/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 20min | lightgbm_classifier | init_model | False |  | +0.000247 | 2 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261107-045330/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 24h | lightgbm_classifier | init_model | False |  | +0.001035 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261107-045330/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 6h | lightgbm_classifier | init_model | False |  | +0.000598 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261107-045330/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 0min | lightgbm | network_transfer | True |  | -0.026107 | 4 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261107-045318/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 1h | lightgbm | network_transfer | True |  | -0.004131 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261107-045318/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 20min | lightgbm | network_transfer | True |  | -0.019617 | 4 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261107-045318/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 0min | lightgbm_classifier | network_transfer | True |  | -0.000601 | 4 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261107-045703/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 1h | lightgbm_classifier | network_transfer | True |  | -0.000402 | 3 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261107-045703/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 20min | lightgbm_classifier | network_transfer | True |  | +0.000707 | 1 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261107-045703/paired_vs_base.csv) |

#### Step 3.2 leading permutation groups and TreeSHAP features

| task | prediction_time | group | score_increase | feature | mean_abs_shap | details |
| --- | --- | --- | --- | --- | --- | --- |
| severity | 0min | delay | 0.046587 | hour | 0.151402 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261107-043030) |
| severity | 1h | recent_reliability | 0.029224 | hour | 0.127759 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261107-043030) |
| severity | 20min | delay | 0.030662 | hour | 0.153809 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261107-043030) |
| severity | 24h | calendar | 0.030103 | hour | 0.156797 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261107-043030) |
| severity | 6h | recent_reliability | 0.041252 | hour | 0.112883 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261107-043030) |
| delay | 0min | delay | 1.503805 | current_delay | 2.138394 | [folds](results/advanced_3_2_lightgbm_weather/20261107-035035) |
| delay | 1h | delay | 0.889239 | hour | 2.013026 | [folds](results/advanced_3_2_lightgbm_weather/20261107-035035) |
| delay | 20min | delay | 1.017541 | hour | 1.963051 | [folds](results/advanced_3_2_lightgbm_weather/20261107-035035) |
| delay | 24h | calendar | 0.589756 | hour | 2.417021 | [folds](results/advanced_3_2_lightgbm_weather/20261107-035035) |
| delay | 6h | recent_reliability | 0.956969 | service_mean_delay_min_28d | 1.988083 | [folds](results/advanced_3_2_lightgbm_weather/20261107-035035) |

#### Step 3.5 group choices at 24 h and 0 min

| task | weather | direction | prediction_time | group | folds_selected |
| --- | --- | --- | --- | --- | --- |
| delay | False | backward | 24h | calendar | 5 |
| delay | False | backward | 24h | recent_reliability | 5 |
| delay | False | backward | 24h | timetable | 5 |
| delay | False | backward | 24h | network_state | 1 |
| delay | False | backward | 0min | calendar | 5 |
| delay | False | backward | 0min | delay | 5 |
| delay | False | backward | 0min | recent_reliability | 5 |
| delay | False | backward | 0min | timetable | 5 |
| delay | False | backward | 0min | network_state | 3 |
| delay | False | backward | 0min | route | 2 |
| delay | False | forward | 24h | calendar | 5 |
| delay | False | forward | 24h | recent_reliability | 5 |
| delay | False | forward | 24h | timetable | 5 |
| delay | False | forward | 24h | network_state | 1 |
| delay | False | forward | 0min | calendar | 5 |
| delay | False | forward | 0min | delay | 5 |
| delay | False | forward | 0min | recent_reliability | 5 |
| delay | False | forward | 0min | timetable | 5 |
| delay | False | forward | 0min | network_state | 3 |
| delay | False | forward | 0min | route | 2 |
| delay | True | backward | 24h | calendar | 5 |
| delay | True | backward | 24h | recent_reliability | 5 |
| delay | True | backward | 24h | timetable | 5 |
| delay | True | backward | 24h | network_state | 1 |
| delay | True | backward | 24h | wind | 1 |
| delay | True | backward | 24h | snow | 1 |
| delay | True | backward | 0min | calendar | 5 |
| delay | True | backward | 0min | delay | 5 |
| delay | True | backward | 0min | recent_reliability | 5 |
| delay | True | backward | 0min | timetable | 5 |
| delay | True | backward | 0min | wind | 3 |
| delay | True | backward | 0min | network_state | 3 |
| delay | True | backward | 0min | temperature | 2 |
| delay | True | backward | 0min | route | 2 |
| delay | True | forward | 24h | calendar | 5 |
| delay | True | forward | 24h | recent_reliability | 5 |
| delay | True | forward | 24h | timetable | 5 |
| delay | True | forward | 24h | network_state | 1 |
| delay | True | forward | 24h | wind | 1 |
| delay | True | forward | 24h | snow | 1 |
| delay | True | forward | 0min | calendar | 5 |
| delay | True | forward | 0min | delay | 5 |
| delay | True | forward | 0min | recent_reliability | 5 |
| delay | True | forward | 0min | timetable | 5 |
| delay | True | forward | 0min | wind | 3 |
| delay | True | forward | 0min | network_state | 3 |
| delay | True | forward | 0min | temperature | 2 |
| delay | True | forward | 0min | route | 2 |
| severity | False | backward | 24h | recent_reliability | 5 |
| severity | False | backward | 24h | timetable | 5 |
| severity | False | backward | 24h | calendar | 2 |
| severity | False | backward | 24h | route | 3 |
| severity | False | backward | 0min | delay | 5 |
| severity | False | backward | 0min | network_state | 5 |
| severity | False | backward | 0min | recent_reliability | 5 |
| severity | False | backward | 0min | timetable | 5 |
| severity | False | backward | 0min | route | 4 |
| severity | False | backward | 0min | calendar | 2 |
| severity | False | forward | 24h | recent_reliability | 5 |
| severity | False | forward | 24h | timetable | 5 |
| severity | False | forward | 24h | calendar | 2 |
| severity | False | forward | 24h | route | 3 |
| severity | False | forward | 0min | delay | 5 |
| severity | False | forward | 0min | network_state | 5 |
| severity | False | forward | 0min | recent_reliability | 5 |
| severity | False | forward | 0min | timetable | 5 |
| severity | False | forward | 0min | route | 4 |
| severity | False | forward | 0min | calendar | 2 |
| severity | True | backward | 24h | recent_reliability | 5 |
| severity | True | backward | 24h | temperature | 4 |
| severity | True | backward | 24h | timetable | 5 |
| severity | True | backward | 24h | route | 3 |
| severity | True | backward | 24h | snow | 2 |
| severity | True | backward | 24h | calendar | 1 |
| severity | True | backward | 0min | delay | 5 |
| severity | True | backward | 0min | network_state | 5 |
| severity | True | backward | 0min | recent_reliability | 5 |
| severity | True | backward | 0min | timetable | 5 |
| severity | True | backward | 0min | wind | 2 |
| severity | True | backward | 0min | route | 4 |
| severity | True | backward | 0min | temperature | 2 |
| severity | True | backward | 0min | calendar | 1 |
| severity | True | forward | 24h | recent_reliability | 5 |
| severity | True | forward | 24h | temperature | 2 |
| severity | True | forward | 24h | timetable | 5 |
| severity | True | forward | 24h | calendar | 2 |
| severity | True | forward | 24h | route | 3 |
| severity | True | forward | 24h | wind | 1 |
| severity | True | forward | 0min | delay | 5 |
| severity | True | forward | 0min | network_state | 5 |
| severity | True | forward | 0min | recent_reliability | 5 |
| severity | True | forward | 0min | timetable | 5 |
| severity | True | forward | 0min | wind | 1 |
| severity | True | forward | 0min | route | 4 |
| severity | True | forward | 0min | temperature | 2 |
| severity | True | forward | 0min | calendar | 2 |

<!-- END ADVANCED CV -->
