# Advanced modelling: Frankfurt (Main) Hbf → München Hbf

The experiments of the [advanced modelling plan](PLAN.md), added in PR #12: inputs known at the cutoff, new feature groups, nested tuning, other model families, ensembles, neural networks, transfer learning, and the one-time hold-out evaluation of railway-only against weather models. The plan is also published at https://claude.ai/artifact/KAso5U9ubq5rFxMy7juUna (same content as `PLAN.md` and `plan.html`).

**Outcome.** Most of this work is exploratory screening, scored on the same five CV week folds as [`../modeling/`](../modeling/README.md). New input features gave the clearest gains: recent reliability (step 2.1) reached the predeclared minimum gain (D12) in 17 of 20 comparisons, and all new groups together in 19 of 20. Changes to the learner rarely helped. Seed and configuration bagging reached D12 in 8 of 120 comparisons and pooling the horizons in 6 of 20. The two-stage models, CatBoost, the explainable boosting machine, the distributional models, calibration, stacking, weighted ensembles and corridor or network transfer never did, and the plain MLP never did against LightGBM. The registered final models are LightGBM and linear regression, the families already in `../modeling/`. On the 23 hold-out weeks, adding weather did not improve either task: all ten paired 95% intervals include zero.

## Layout

| Files | Purpose |
|---|---|
| `scheduled_timetable.py`, `history_features.py`, `recent_reliability.py`, `holiday_features.py` | Inputs known at the cutoff, built from the plan and change snapshots: planned timetable, live delay and its dynamics, recent reliability, network state, holidays |
| `build_advanced_weather.py`, `build_corridor_inputs.py`, `network_transfer.py`, `sequence_inputs.py` | Weather on the planned route, other corridors (8.3), network-wide delay pairs (8.4), upstream sequences (7.5) |
| `build_forecast_weather.py` | GFS forecast weather for the hours of the planned journey (step 10.8, decision D17), alongside the observed weather |
| `holdout_gate.py` | Refuses hold-out builds unless the committed registry matches the current source |
| `run_advanced.py`, `experiment_specs.py`, `experiment_models.py`, `advanced_data.py` | The CV steps of the plan: which arms exist, their models and inputs |
| `representations.py`, `group_selection.py`, `decomposed.py`, `ensembles.py`, `extra_models.py`, `distributions.py`, `neural.py`, `gated_neural.py`, `sequence_model.py`, `corridor_transfer.py` | Phase 1 and 3 to 8 methods |
| `compare.py`, `report_cv.py`, `audit_cv_results.py` | Paired fold comparisons, the CV tables below, and the output audit of the CV runs |
| `run_final.py`, `final_analysis.py`, `result_plots.py`, `finalists.json` | Registration of finalists and the hold-out evaluation |
| `flat_searches/` | The eight flat searches of the first models (LightGBM and XGBoost, with and without weather, both tasks), moved from `../modeling/results/` on 16 November 2026 |
| `results/` | One folder per arm, and `results/advanced_holdout/` |
| `tests/` | `.venv/bin/python -m pytest frankfurt_munich/advanced_modeling/tests` |
| `_paths.py` | Puts `../modeling` and `..` on the import path |

The cross-validation, tuning, feature sets and model factories are shared with the earlier models in `../modeling/` (`cv.py`, `tuning.py`, `features.py`, `models.py`, `data.py`). Generated inputs are written to `../data/advanced/`, which is not committed. The distributional learners need the separate environment in [requirements-distributions.txt](requirements-distributions.txt).

## Inputs and runs

The earlier scores in [`../modeling/`](../modeling/README.md) use recorded-route inputs. `run_advanced.py` runs the CV experiments in [PLAN.md](PLAN.md) with the new cutoff-specific inputs. It keeps the same prediction rows and week folds. The recorded route fields are available only in the step 0.7 leakage comparison and the historical reproduction commands.

The advanced runner reads generated CV inputs in `../data/advanced/`. It requires the timestamped timetable and reconstructed live-delay data and checks that they cover the same prediction keys. It reads advanced weather only for the weather arm.

Download the 27 `monthly_processed_data_plan` files for July 2024 through September 2026 into `data/monthly_processed_data_plan/`. [Pinned plan archive](https://huggingface.co/datasets/piebro/deutsche-bahn-data/tree/d27b9c413536bb5fe393954371f572050e55176b/monthly_processed_data_plan)

The legacy fetcher used the requested timetable hour in plan filenames. Following the user decision on 5 November 2026, the primary build accepts these legacy fields regardless of fetch time, using the earliest plan per stop as its static schedule. Following the decision of 12 November 2026, the plans with exact request stamps (from 3 November 2025) are treated the same way. The 45-minute bound applies to the separate change archive. [Legacy fetcher](https://github.com/piebro/deutsche-bahn-data/blob/f19888b770a68d5f9010adac33af70e400e2395d/fetch_data.py)

The prediction cohort supplies the planned Frankfurt departure and München arrival. The additional timetable fields come from the scheduled train's plan available at the cutoff. If the branch is unavailable, the weather pipeline retains the two endpoint temperatures and leaves the route aggregates missing. The live delay requires both a visible change report and a visible plan. For retained replacements, the scheduled service supplies live history until the link to the replacement becomes visible. When the monthly files record no replaced train (17 CV journeys), no other scheduled train is known, and the operated run supplies the history as for an ordinary train. Those runs have no plan snapshot, so `current_delay` stays missing for them; only the report count is filled. Seven further CV replacements name a replaced train that no plan matched, and their history stays missing.

Recent-reliability candidates come from the same target-free stop roster for CV and holdout. Both builds reconstruct past outcomes from snapshots available at each cutoff. The resulting rates describe journeys with observed outcomes. A missing arrival report stays unknown. An archive check found a destination arrival report somewhere in the snapshots for 16,550 of 16,551 completed CV journeys. That check measures archive coverage. At an individual cutoff, the builder still requires a visible report and an arrival time at or before the cutoff.

Input coverage after the rebuild of 13 November 2026 (corrected dataset, static plans in both archive periods). The CV results below were rerun on these inputs (finishing plan B4, 13–15 November 2026).

| Input coverage | 24 h | 6 h | 1 h | 20 min | 0 min |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV prediction rows | 17,463 | 17,366 | 17,096 | 16,975 | 16,926 |
| Planned route available | 17,364 | 17,267 | 16,998 | 16,878 | 16,830 |
| Weather branch available | 17,363 | 17,266 | 16,997 | 16,877 | 16,829 |
| Snapshot-based `current_delay` used by models | — | — | 6,654 | 7,331 | 8,758 |
| GFS forecast, route features (built 17 November 2026) | 17,363 | 17,266 | 16,997 | 16,877 | 16,829 |

The GFS forecast Frankfurt temperature is known for every CV row and the München temperature for all but 10 (two journeys on 30 September 2026 arrive after the end of the extract); the route features need the weather branch, like the observed weather. Each row uses the latest GFS run that started by its cutoff (a run counts as available from its start, decision D17) and the hours from the planned departure to the planned arrival. No model uses the forecasts yet (step 10.9).

The snapshot-based `current_delay` is missing for about half the rows because the archive fetched live updates in blocks about six hours apart: for most journeys the last fetch before the cutoff came before the train had reported from any stop before Frankfurt (finishing plan B1). Since 13 November 2026 it does not use the departure from Frankfurt itself, as in the dataset's `current_delay`.

Inputs that follow the archive's fetch clock rather than the trains are not used (decisions of 13 November 2026): the network state (step 2.5; its builder `network_state.py` was deleted, see the git history), the corridor reliability over 3 h, the time since the last report and `current_delay_change_min`. When the latest report is an arrival, `current_delay` counts the time the train has stood there since its planned departure only if the same snapshot shows the departure after the cutoff, or the stop is Frankfurt. The scheduled train type is missing for 24 of the 230 kept replacement journeys, whose replaced train no plan matched; the loader sets it to `unknown`.

```bash
.venv/bin/python frankfurt_munich/advanced_modeling/scheduled_timetable.py
.venv/bin/python frankfurt_munich/advanced_modeling/history_features.py --feature all
.venv/bin/python frankfurt_munich/advanced_modeling/build_advanced_weather.py
.venv/bin/python frankfurt_munich/advanced_modeling/build_forecast_weather.py
.venv/bin/python frankfurt_munich/advanced_modeling/run_advanced.py 0.7 --jobs 4
.venv/bin/python frankfurt_munich/advanced_modeling/run_advanced.py 1.1 --jobs 4
.venv/bin/python frankfurt_munich/advanced_modeling/report_cv.py --write-readme
```

`run_advanced.py --help` lists the CV steps and filters. Each weather arm has two forecast arms (D17, step 10.9): `_forecast` (GFS forecasts in place of the observations) and `_weather_forecast` (observations plus forecasts), except nested XGBoost, CatBoost and the arms built on them. `--weather-arm` picks arms (`railway`, `weather`, `forecast`, `weather_forecast`), and `audit_cv_results.py --arm` audits only those. Both forecast arms reuse the observed-weather flat configuration. Fixed configurations come from the eight flat searches of the first models in `flat_searches/`: `config.json` gives the configuration chosen per prediction time, and `search_results.csv` supplies the top-five averages of 5.2. The arms record these sources relative to `flat_searches/`. Following the user decision on 5 November, remaining booster and CatBoost searches use 20 configurations inside each outer training set. Six completed 50-configuration baseline searches also supply budget-matched 20-configuration controls by reusing their fixed first 20 inner-score rows and refitting the outer models. The saved configurations link to the original searches. The reference table and each run configuration record the actual count. Final registration uses at most 20 configurations. Neural searches use eight. The `auto_groups` and `auto_representations` options to `--nested-variant` also select features inside each outer training set. A variant fixed after seeing the exploratory results retains that label even when its parameters are tuned again.

When running a subset of horizons with `--prediction-time`, set a separate `--results-dir`. Reports and registration select the latest timestamp in each run directory, so a partial run in the main results directory can replace the complete run in those reports.

The model-family screens use the cutoff-safe base columns for their learner comparisons. The nested booster feature-group policy evaluates the expanded Phase 2 inputs separately.
The corrected policy ranks each feature variant by its best inner-fold score across the first three configurations in the same search. It then tunes the selected variant over the full 20 configurations. The extra screening fits reuse configurations from that budget.

All 354 active CV arms were rerun on the rebuilt inputs at commit `7e0e96a` (finishing plan B4, 13–15 November 2026): the 366 arms of D14 less the four network-state arms (2.5) and the eight network-state ablations in 3.1, dropped on 13 November. All 354 passed the output audit (`audit_cv_results.py --commit 7e0e96a2569f4722c545856aeae96c0170d7ee80`). It checks that each run has its files, was made from a clean commit, has the expected horizons with five folds each, and has summaries equal to its fold means. It also checks that the saved predictions cover every CV row once in its committed fold, are valid, and reproduce the fold scores. The feature policy reached the predeclared minimum gain in 20 of 20 LightGBM comparisons and 13 of 20 XGBoost comparisons. None of the 12 network-transfer comparisons reached it.

The review of 10 November asked for the neural networks and the pooled model to be compared with nested LightGBM, not only with the MLP or the fixed-configuration booster. These comparisons use the same paired fold differences (`compare.py`), against the step 0.8 LightGBM run of the same task and weather arm on the same horizons:

| Step | Comparisons | Lower mean error than nested LightGBM | Reached minimum gain |
| --- | ---: | ---: | ---: |
| 7.1 MLP | 20 | 0 | 0 |
| 7.2 multi-task MLP | 20 | 0 | 0 |
| 7.3 TabGNS | 20 | 0 | 0 |
| 7.4 RealMLP and TabM | 40 | 1 | 0 |
| 7.5 GRU (1 h on) | 12 | 0 | 0 |
| 8.2 pooled prediction times (exploratory) | 20 | 19 | 4 |

The one neural arm with a lower mean error is TabM with weather for delay at 24 h (0.010 min, three folds improved). The pooled model reached the minimum gain for severity at 0 min, for severity with weather at 1 h and 20 min, and for delay with weather at 6 h. The pooled models reuse flat-search configurations, so these comparisons are exploratory.

Steps 6.5 and 4.2 use a separate Python 3.13 environment with [requirements-distributions.txt](requirements-distributions.txt). NGBoost and LightGBMLSS constrain dependencies to older releases than the main environment uses. In the B4 rerun they ran in that environment with `OMP_NUM_THREADS=1`, for example `OMP_NUM_THREADS=1 <distribution-python> frankfurt_munich/advanced_modeling/run_advanced.py 6.5 --jobs 3`. Set it as well when `run_final.py evaluate` is given `--distribution-python`. Without that setting LightGBMLSS crashes on macOS, because scikit-learn, LightGBM and torch each load their own OpenMP runtime. The models are single-threaded, so the setting does not change their results. The run of 4.2 and 6.5 before the rebuild had recorded the main environment's numpy, pandas and scikit-learn versions, not these pins.

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
| [class_shares](results/advanced_0_7_class_shares/20261113-212614) | fixed | log_loss | 0.967992 | 0.958166 | 0.922901 | 0.904186 | 0.896396 |
| [lightgbm](results/advanced_0_7_lightgbm/20261113-212635) | fixed | rmse | 24.495471 | 24.505243 | 24.080572 | 24.020343 | 23.721266 |
| [lightgbm_classifier](results/advanced_0_7_lightgbm_classifier/20261113-212808) | fixed | log_loss | 0.920320 | 0.911833 | 0.865819 | 0.845798 | 0.828542 |
| [lightgbm_classifier + weather](results/advanced_0_7_lightgbm_classifier_weather/20261113-212856) | fixed | log_loss | 0.922384 | 0.916049 | 0.869455 | 0.850772 | 0.834340 |
| [lightgbm + weather](results/advanced_0_7_lightgbm_weather/20261113-212638) | fixed | rmse | 24.513819 | 24.501714 | 24.163462 | 24.090464 | 23.787463 |
| [linear_regression](results/advanced_0_7_linear_regression/20261113-212614) | fixed | rmse | 24.628921 | 24.634678 | 24.191326 | 24.148835 | 23.906206 |
| [linear_regression + weather](results/advanced_0_7_linear_regression_weather/20261113-212615) | fixed | rmse | 24.605495 | 24.644969 | 24.208634 | 24.161579 | 23.918108 |
| [logistic_regression](results/advanced_0_7_logistic_regression/20261113-212619) | fixed | log_loss | 0.930236 | 0.920970 | 0.876390 | 0.856842 | 0.841432 |
| [logistic_regression + weather](results/advanced_0_7_logistic_regression_weather/20261113-212624) | fixed | log_loss | 0.931270 | 0.924145 | 0.877911 | 0.857848 | 0.842388 |
| [training_mean](results/advanced_0_7_training_mean/20261113-212613) | fixed | rmse | 25.423325 | 25.431851 | 25.403242 | 25.404544 | 25.404926 |
| [xgboost](results/advanced_0_7_xgboost/20261113-212830) | fixed | rmse | 24.478576 | 24.484618 | 24.085617 | 23.999789 | 23.697279 |
| [xgboost_classifier](results/advanced_0_7_xgboost_classifier/20261113-212944) | fixed | log_loss | 0.920912 | 0.911975 | 0.865809 | 0.848441 | 0.831868 |
| [xgboost_classifier + weather](results/advanced_0_7_xgboost_classifier_weather/20261113-213111) | fixed | log_loss | 0.920159 | 0.913658 | 0.869298 | 0.850174 | 0.833422 |
| [xgboost + weather](results/advanced_0_7_xgboost_weather/20261113-212859) | fixed | rmse | 24.445143 | 24.478797 | 24.209807 | 24.065709 | 23.766262 |

#### Step 0.8 nested reference scores

| Model | Search configs | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [lightgbm](results/advanced_0_8_lightgbm/20261113-221438) | 20 | rmse | 24.521828 | 24.555725 | 24.118380 | 24.022716 | 23.776330 |
| [lightgbm_classifier](results/advanced_0_8_lightgbm_classifier/20261113-230146) | 20 | log_loss | 0.924414 | 0.915752 | 0.868344 | 0.845636 | 0.833885 |
| [lightgbm_classifier + weather](results/advanced_0_8_lightgbm_classifier_weather/20261114-000751) | 20 | log_loss | 0.922686 | 0.915433 | 0.870172 | 0.849489 | 0.833584 |
| [lightgbm + weather](results/advanced_0_8_lightgbm_weather/20261113-223957) | 20 | rmse | 24.545003 | 24.574248 | 24.131414 | 24.071326 | 23.822866 |
| [xgboost](results/advanced_0_8_xgboost/20261113-224228) | 20 | rmse | 24.508199 | 24.520966 | 24.081305 | 24.000247 | 23.692860 |
| [xgboost_classifier](results/advanced_0_8_xgboost_classifier/20261114-001244) | 20 | log_loss | 0.920901 | 0.912369 | 0.867531 | 0.847626 | 0.832320 |
| [xgboost_classifier + weather](results/advanced_0_8_xgboost_classifier_weather/20261114-013919) | 20 | log_loss | 0.920159 | 0.913658 | 0.866848 | 0.847307 | 0.831830 |
| [xgboost + weather](results/advanced_0_8_xgboost_weather/20261113-233729) | 20 | rmse | 24.469813 | 24.486058 | 24.077068 | 24.065709 | 23.766262 |

#### Smallest paired differences by step, task and horizon

| step | task | evaluation | prediction_time | method | variant | weather | contrast | mean | folds_improved | reaches_minimum_gain | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | delay | exploratory | 0min | lightgbm | recorded_route | True |  | -0.102627 | 4 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261113-212651/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 1h | xgboost | recorded_route | True |  | -0.107646 | 4 | True | [folds](results/advanced_0_7_xgboost_weather_recorded_route/20261113-212921/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 20min | lightgbm | recorded_route | True |  | -0.098248 | 5 | False | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261113-212651/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 24h | lightgbm | recorded_route | True |  | -0.129485 | 4 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261113-212651/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 6h | lightgbm | recorded_route | False |  | -0.107571 | 5 | True | [folds](results/advanced_0_7_lightgbm_recorded_route/20261113-212642/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 0min | xgboost_classifier | recorded_route | True |  | -0.004221 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_recorded_route/20261113-213135/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 1h | xgboost_classifier | recorded_route | True |  | -0.003060 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_recorded_route/20261113-213135/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 20min | xgboost_classifier | recorded_route | True |  | -0.003986 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_recorded_route/20261113-213135/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 24h | logistic_regression | recorded_route | False |  | -0.003255 | 4 | True | [folds](results/advanced_0_7_logistic_regression_recorded_route/20261113-212622/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 6h | xgboost_classifier | recorded_route | True |  | -0.003323 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_recorded_route/20261113-213135/paired_vs_base.csv) |
| 0.8 | delay | nested | 0min | xgboost | auto_groups | True |  | -0.376702 | 5 | True | [folds](results/advanced_0_8_xgboost_weather_auto_groups/20261114-041458/paired_vs_base.csv) |
| 0.8 | delay | nested | 1h | lightgbm | auto_groups | False |  | -0.372655 | 4 | True | [folds](results/advanced_0_8_lightgbm_auto_groups/20261114-022430/paired_vs_base.csv) |
| 0.8 | delay | nested | 20min | lightgbm | auto_groups | False |  | -0.352300 | 5 | True | [folds](results/advanced_0_8_lightgbm_auto_groups/20261114-022430/paired_vs_base.csv) |
| 0.8 | delay | nested | 24h | lightgbm | auto_groups | False |  | -0.245612 | 4 | True | [folds](results/advanced_0_8_lightgbm_auto_groups/20261114-022430/paired_vs_base.csv) |
| 0.8 | delay | nested | 6h | lightgbm | auto_groups | False |  | -0.347379 | 5 | True | [folds](results/advanced_0_8_lightgbm_auto_groups/20261114-022430/paired_vs_base.csv) |
| 0.8 | severity | nested | 0min | xgboost_classifier | auto_groups | True |  | -0.014405 | 5 | True | [folds](results/advanced_0_8_xgboost_classifier_weather_auto_groups/20261114-081135/paired_vs_base.csv) |
| 0.8 | severity | nested | 1h | lightgbm_classifier | auto_groups | True |  | -0.011954 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261114-045420/paired_vs_base.csv) |
| 0.8 | severity | nested | 20min | lightgbm_classifier | auto_groups | False |  | -0.011308 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_auto_groups/20261114-040128/paired_vs_base.csv) |
| 0.8 | severity | nested | 24h | lightgbm_classifier | auto_groups | False |  | -0.006831 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_auto_groups/20261114-040128/paired_vs_base.csv) |
| 0.8 | severity | nested | 6h | lightgbm_classifier | auto_groups | False |  | -0.014266 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_auto_groups/20261114-040128/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 0min | lightgbm | calendar_cyclic | False |  | -0.037526 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261114-081158/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 1h | lightgbm | calendar_cyclic | False |  | -0.060085 | 5 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261114-081158/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 20min | lightgbm | calendar_cyclic | False |  | -0.037351 | 5 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261114-081158/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 24h | linear_regression | calendar_combined | False |  | -0.000006 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261114-081157/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 6h | linear_regression | calendar_combined | False |  | -0.000003 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261114-081157/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 0min | lightgbm_classifier | calendar_cyclic | False |  | -0.001720 | 4 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261114-081302/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 1h | lightgbm_classifier | calendar_cyclic | False |  | -0.002585 | 5 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261114-081302/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 20min | lightgbm_classifier | calendar_cyclic | False |  | -0.001793 | 5 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261114-081302/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 24h | logistic_regression | calendar_combined | False |  | +0.000078 | 2 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261114-081202/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 6h | logistic_regression | calendar_combined | False |  | +0.000043 | 2 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261114-081202/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 0min | lightgbm | minute_of_day | False |  | -0.049968 | 4 | False | [folds](results/advanced_1_2_lightgbm_minute_of_day/20261114-081354/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 1h | lightgbm | minute_of_day | False |  | -0.041076 | 5 | False | [folds](results/advanced_1_2_lightgbm_minute_of_day/20261114-081354/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 20min | linear_regression | day_of_year | False |  | -0.036690 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261114-081338/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 24h | linear_regression | day_of_year | False |  | -0.043554 | 3 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261114-081338/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 6h | linear_regression | day_of_year | False |  | -0.043075 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261114-081338/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 0min | lightgbm_classifier | minute_of_day | False |  | -0.002075 | 4 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261114-081503/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 1h | logistic_regression | day_of_year | False |  | -0.001733 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261114-081342/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 20min | logistic_regression | day_of_year | False |  | -0.001853 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261114-081342/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 24h | lightgbm_classifier | minute_of_day | False |  | -0.001301 | 3 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261114-081503/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 6h | lightgbm_classifier | minute_of_day | False |  | -0.001252 | 3 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261114-081503/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 0min | lightgbm | route_onehot | False |  | -0.013491 | 4 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 1h | lightgbm | route_onehot | False |  | -0.015656 | 3 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 20min | linear_regression | route_target | False |  | -0.019671 | 2 | False | [folds](results/advanced_1_3_linear_regression_route_target/20261114-081531/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 24h | lightgbm | route_onehot | False |  | -0.109568 | 5 | True | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 6h | lightgbm | route_onehot | False |  | -0.112356 | 5 | True | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 0min | lightgbm_classifier | route_onehot | False |  | -0.001761 | 5 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 1h | lightgbm_classifier | route_onehot | False |  | -0.001457 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 20min | lightgbm_classifier | route_onehot | False |  | -0.001523 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 24h | lightgbm_classifier | route_onehot | False |  | -0.001387 | 3 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 6h | lightgbm_classifier | route_onehot | False |  | -0.001393 | 3 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 0min | lightgbm | delay_missing | False |  | -0.005123 | 3 | False | [folds](results/advanced_1_4_lightgbm_delay_missing/20261114-081741/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 1h | lightgbm | delay_clipped | False |  | -0.004195 | 5 | False | [folds](results/advanced_1_4_lightgbm_delay_clipped/20261114-081732/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 20min | linear_regression | delay_clipped | False |  | -0.002789 | 3 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261114-081723/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 0min | logistic_regression | delay_bins | False |  | -0.000137 | 2 | False | [folds](results/advanced_1_4_logistic_regression_delay_bins/20261114-081728/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 1h | lightgbm_classifier | delay_missing | False |  | -0.000599 | 5 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_missing/20261114-081834/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 20min | lightgbm_classifier | delay_missing | False |  | -0.000106 | 3 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_missing/20261114-081834/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 0min | lightgbm | weather_extremes | True |  | -0.092755 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261114-081848/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 1h | lightgbm | weather_extremes | True |  | -0.124748 | 3 | True | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261114-081848/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 20min | lightgbm | weather_extremes | True |  | -0.104239 | 3 | True | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261114-081848/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 24h | lightgbm | weather_extremes | True |  | -0.030373 | 3 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261114-081848/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 6h | lightgbm | weather_extremes | True |  | +0.062867 | 1 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261114-081848/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 0min | lightgbm_classifier | weather_extremes | True |  | -0.005847 | 4 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261114-081958/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 1h | lightgbm_classifier | weather_extremes | True |  | -0.006355 | 5 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261114-081958/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 20min | lightgbm_classifier | weather_extremes | True |  | -0.005509 | 4 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261114-081958/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 24h | lightgbm_classifier | weather_extremes | True |  | -0.000688 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261114-081958/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 6h | lightgbm_classifier | weather_extremes | True |  | +0.000072 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261114-081958/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 0min | linear_regression | recent_reliability | False |  | -0.383250 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 1h | linear_regression | recent_reliability | False |  | -0.392831 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 20min | linear_regression | recent_reliability | False |  | -0.395945 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 24h | linear_regression | recent_reliability | False |  | -0.249050 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 6h | linear_regression | recent_reliability | False |  | -0.402704 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 0min | logistic_regression | recent_reliability | False |  | -0.014766 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 1h | logistic_regression | recent_reliability | False |  | -0.014679 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 20min | logistic_regression | recent_reliability | False |  | -0.014624 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 24h | logistic_regression | recent_reliability | False |  | -0.008698 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 6h | logistic_regression | recent_reliability | False |  | -0.014694 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 0min | lightgbm | planned_timetable | False |  | -0.117382 | 4 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 1h | lightgbm | planned_timetable | False |  | -0.118120 | 4 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 20min | lightgbm | planned_timetable | False |  | -0.126810 | 5 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 24h | lightgbm | planned_timetable | False |  | -0.207707 | 5 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 6h | lightgbm | planned_timetable | False |  | -0.223698 | 5 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 0min | logistic_regression | planned_timetable | False |  | -0.002254 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261114-082248/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 1h | lightgbm_classifier | planned_timetable | False |  | -0.002090 | 4 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261114-082412/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 20min | logistic_regression | planned_timetable | False |  | -0.001681 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261114-082248/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 24h | lightgbm_classifier | planned_timetable | False |  | -0.004379 | 4 | True | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261114-082412/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 6h | lightgbm_classifier | planned_timetable | False |  | -0.004231 | 4 | True | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261114-082412/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 0min | lightgbm | holidays | False |  | -0.002987 | 3 | False | [folds](results/advanced_2_3_lightgbm_holidays/20261114-082433/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 1h | linear_regression | holidays | False |  | -0.004998 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261114-082417/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 20min | lightgbm | holidays | False |  | -0.007948 | 2 | False | [folds](results/advanced_2_3_lightgbm_holidays/20261114-082433/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 24h | linear_regression | holidays | False |  | -0.010071 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261114-082417/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 6h | linear_regression | holidays | False |  | -0.010398 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261114-082417/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 0min | lightgbm_classifier | holidays | False |  | +0.001639 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 1h | lightgbm_classifier | holidays | False |  | +0.001081 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 20min | lightgbm_classifier | holidays | False |  | +0.001629 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 24h | lightgbm_classifier | holidays | False |  | +0.000274 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 6h | lightgbm_classifier | holidays | False |  | +0.000178 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 0min | lightgbm | delay_dynamics | False |  | -0.043798 | 5 | False | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261114-082600/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 1h | lightgbm | delay_dynamics | False |  | -0.036659 | 4 | False | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261114-082600/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 20min | lightgbm | delay_dynamics | False |  | -0.060889 | 5 | False | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261114-082600/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 0min | lightgbm_classifier | delay_dynamics | False |  | -0.002985 | 5 | False | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261114-082653/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 1h | lightgbm_classifier | delay_dynamics | False |  | -0.003230 | 5 | True | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261114-082653/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 20min | lightgbm_classifier | delay_dynamics | False |  | -0.002352 | 5 | False | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261114-082653/paired_vs_base.csv) |
| 2.all | delay | exploratory | 0min | lightgbm | all_groups | False |  | -0.386688 | 5 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261114-082725/paired_vs_base.csv) |
| 2.all | delay | exploratory | 1h | linear_regression | all_groups | False |  | -0.333216 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261114-082701/paired_vs_base.csv) |
| 2.all | delay | exploratory | 20min | lightgbm | all_groups | False |  | -0.378661 | 5 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261114-082725/paired_vs_base.csv) |
| 2.all | delay | exploratory | 24h | linear_regression | all_groups | False |  | -0.244677 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261114-082701/paired_vs_base.csv) |
| 2.all | delay | exploratory | 6h | linear_regression | all_groups | False |  | -0.382828 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261114-082701/paired_vs_base.csv) |
| 2.all | severity | exploratory | 0min | lightgbm_classifier | all_groups | False |  | -0.011990 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 1h | lightgbm_classifier | all_groups | False |  | -0.010606 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 20min | lightgbm_classifier | all_groups | False |  | -0.010677 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 24h | lightgbm_classifier | all_groups | False |  | -0.005692 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 6h | logistic_regression | all_groups | False |  | -0.011190 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261114-082717/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | linear_regression | drop_route | True | full | -0.032215 | 2 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_route/20261114-082918/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | lightgbm | add_delay | True | timetable-only | -0.965828 | 5 | True | [folds](results/advanced_3_1_lightgbm_weather_add_delay/20261114-083105/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | lightgbm | drop_temperature | True | full | -0.033886 | 2 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_temperature/20261114-083124/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | linear_regression | add_recent_reliability | True | timetable-only | -0.819301 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261114-082917/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | lightgbm | drop_temperature | True | full | -0.046254 | 3 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_temperature/20261114-083124/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | linear_regression | add_recent_reliability | True | timetable-only | -0.812501 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261114-082917/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | lightgbm | drop_precipitation | True | full | -0.048063 | 5 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_precipitation/20261114-083133/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | linear_regression | add_calendar | True | timetable-only | -0.560817 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_calendar/20261114-082915/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | drop_temperature | True | full | -0.014465 | 3 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_temperature/20261114-082925/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | add_recent_reliability | True | timetable-only | -0.767714 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261114-082917/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | drop_route | True | full | -0.001468 | 2 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_route/20261114-082946/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | add_recent_reliability | True | timetable-only | -0.032062 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261114-082938/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | drop_route | True | full | -0.001925 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_route/20261114-082946/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | add_recent_reliability | True | timetable-only | -0.031938 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261114-082938/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | drop_route | True | full | -0.002094 | 5 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_route/20261114-082946/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | add_recent_reliability | True | timetable-only | -0.031606 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261114-082938/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | drop_route | True | full | -0.000970 | 2 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_route/20261114-082946/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | add_recent_reliability | True | timetable-only | -0.022182 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261114-082938/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | drop_temperature | True | full | -0.001016 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_temperature/20261114-083034/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | add_recent_reliability | True | timetable-only | -0.030013 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261114-082938/paired_vs_base.csv) |
| 3.3 | delay | nested | 0min | linear_regression | backward | False |  | -0.042898 | 3 | False | [folds](results/advanced_3_3_linear_regression_backward/20261114-090529/paired_vs_base.csv) |
| 3.3 | delay | nested | 1h | linear_regression | backward | False |  | -0.037013 | 3 | False | [folds](results/advanced_3_3_linear_regression_backward/20261114-090529/paired_vs_base.csv) |
| 3.3 | delay | nested | 20min | linear_regression | backward | False |  | -0.043734 | 3 | False | [folds](results/advanced_3_3_linear_regression_backward/20261114-090529/paired_vs_base.csv) |
| 3.3 | delay | nested | 24h | linear_regression | forward | True |  | +0.017868 | 3 | False | [folds](results/advanced_3_3_linear_regression_weather_forward/20261114-090701/paired_vs_base.csv) |
| 3.3 | delay | nested | 6h | linear_regression | backward | True |  | +0.018640 | 1 | False | [folds](results/advanced_3_3_linear_regression_weather_backward/20261114-090837/paired_vs_base.csv) |
| 3.3 | severity | nested | 0min | logistic_regression | backward | True |  | -0.000807 | 3 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261114-092848/paired_vs_base.csv) |
| 3.3 | severity | nested | 1h | logistic_regression | forward | True |  | -0.002890 | 4 | False | [folds](results/advanced_3_3_logistic_regression_weather_forward/20261114-091602/paired_vs_base.csv) |
| 3.3 | severity | nested | 20min | logistic_regression | forward | True |  | -0.002792 | 5 | False | [folds](results/advanced_3_3_logistic_regression_weather_forward/20261114-091602/paired_vs_base.csv) |
| 3.3 | severity | nested | 24h | logistic_regression | forward | True |  | +0.002030 | 2 | False | [folds](results/advanced_3_3_logistic_regression_weather_forward/20261114-091602/paired_vs_base.csv) |
| 3.3 | severity | nested | 6h | logistic_regression | forward | True |  | -0.001289 | 4 | False | [folds](results/advanced_3_3_logistic_regression_weather_forward/20261114-091602/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 0min | logistic_regression | l1 | True |  | -0.004427 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261114-092911/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 1h | logistic_regression | l1 | True |  | -0.004593 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261114-092911/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 20min | logistic_regression | l1 | True |  | -0.004926 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261114-092911/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 24h | logistic_regression | l1 | True |  | -0.004180 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261114-092911/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 6h | logistic_regression | l1 | True |  | -0.004534 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261114-092911/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 0min | lightgbm | wind | True |  | -0.056555 | 2 | False | [folds](results/advanced_3_4_lightgbm_weather_wind/20261114-092932/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 1h | lightgbm | wind | True |  | -0.063169 | 2 | False | [folds](results/advanced_3_4_lightgbm_weather_wind/20261114-092932/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 20min | lightgbm | weather_extremes | True |  | -0.083057 | 2 | False | [folds](results/advanced_3_4_lightgbm_weather_weather_extremes/20261114-092931/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 24h | lightgbm | weather_extremes | True |  | -0.080722 | 3 | False | [folds](results/advanced_3_4_lightgbm_weather_weather_extremes/20261114-092931/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 6h | lightgbm | temperature | True |  | -0.016882 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261114-092933/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 0min | lightgbm_classifier | weather_extremes | True |  | -0.002346 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_weather_extremes/20261114-093138/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 1h | lightgbm_classifier | temperature | True |  | -0.000961 | 3 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261114-093202/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 20min | lightgbm_classifier | weather_extremes | True |  | -0.002011 | 3 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_weather_extremes/20261114-093138/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 24h | lightgbm_classifier | temperature | True |  | -0.001430 | 5 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261114-093202/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 6h | lightgbm_classifier | weather_extremes | True |  | -0.000827 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_weather_extremes/20261114-093138/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_class | True |  | +0.000105 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261114-093418/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_class | False |  | -0.000317 | 4 | False | [folds](results/advanced_4_1_lightgbm_classifier_cancellation_then_class/20261114-093341/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_class | True |  | -0.000312 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261114-093418/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_class | False |  | +0.000471 | 1 | False | [folds](results/advanced_4_1_lightgbm_classifier_cancellation_then_class/20261114-093341/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_class | True |  | +0.000224 | 3 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261114-093418/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 0min | quantile_forest_severity |  | True |  | +0.036841 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261114-222012/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 1h | quantile_forest_severity |  | True |  | +0.023919 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261114-222012/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 20min | quantile_forest_severity |  | True |  | +0.035857 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261114-222012/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 24h | quantile_forest_severity |  | True |  | +0.021106 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261114-222012/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 6h | quantile_forest_severity |  | True |  | +0.027903 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261114-222012/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.001280 | 1 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261114-093557/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_ordinal | False |  | +0.001324 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_cancellation_then_ordinal/20261114-093529/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.000391 | 2 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261114-093557/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.002034 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261114-093557/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.001602 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261114-093557/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 0min | lightgbm | mixture_of_experts | True |  | -0.029178 | 2 | False | [folds](results/advanced_4_4_lightgbm_weather_mixture_of_experts/20261114-093709/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 1h | lightgbm | mixture_of_experts | False |  | -0.081268 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261114-093712/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 20min | lightgbm | mixture_of_experts | True |  | -0.048304 | 2 | False | [folds](results/advanced_4_4_lightgbm_weather_mixture_of_experts/20261114-093709/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 24h | lightgbm | mixture_of_experts | False |  | -0.033654 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261114-093712/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 6h | lightgbm | mixture_of_experts | False |  | -0.036355 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261114-093712/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 0min | lightgbm | change_then_level | False |  | +0.010127 | 1 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261114-093727/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 1h | lightgbm | change_then_level | False |  | -0.009455 | 3 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261114-093727/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 20min | lightgbm | change_then_level | False |  | +0.017280 | 2 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261114-093727/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 0min | lightgbm | plus_catboost | True |  | -0.063649 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_catboost/20261115-055728/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 1h | lightgbm | plus_linear | True |  | -0.104592 | 3 | True | [folds](results/advanced_5_1_lightgbm_weather_plus_linear/20261115-055728/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 20min | lightgbm | plus_linear | True |  | -0.076194 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_linear/20261115-055728/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 24h | lightgbm | plus_neural | True |  | -0.014854 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261115-055733/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 6h | lightgbm | plus_linear | True |  | -0.011991 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_linear/20261115-055728/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 0min | lightgbm_classifier | plus_catboost | True |  | -0.003296 | 4 | True | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 1h | lightgbm_classifier | plus_catboost | True |  | -0.003655 | 5 | True | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 20min | lightgbm_classifier | plus_catboost | True |  | -0.004283 | 5 | True | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 24h | lightgbm_classifier | plus_catboost | True |  | -0.001245 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 6h | lightgbm_classifier | geometric | False |  | -0.000596 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_geometric/20261115-055728/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 0min | xgboost | top5_flat | False |  | -0.044814 | 5 | False | [folds](results/advanced_5_2_xgboost_top5_flat/20261115-055915/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 1h | xgboost | top5_flat | True |  | -0.114345 | 5 | True | [folds](results/advanced_5_2_xgboost_weather_top5_flat/20261115-060640/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 20min | lightgbm | top5_flat | False |  | -0.052738 | 5 | False | [folds](results/advanced_5_2_lightgbm_top5_flat/20261115-055907/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 24h | lightgbm | top5_flat | True |  | -0.057413 | 4 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_flat/20261115-060140/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 6h | lightgbm | seeds | False |  | -0.037025 | 4 | False | [folds](results/advanced_5_2_lightgbm_seeds/20261115-055922/paired_vs_base.csv) |
| 5.2 | delay | nested | 0min | lightgbm | top5_nested | True |  | -0.097550 | 4 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_nested/20261115-060105/paired_vs_base.csv) |
| 5.2 | delay | nested | 1h | lightgbm | top5_nested | True |  | -0.062464 | 4 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_nested/20261115-060105/paired_vs_base.csv) |
| 5.2 | delay | nested | 20min | xgboost | top5_nested | True |  | -0.075812 | 5 | False | [folds](results/advanced_5_2_xgboost_weather_top5_nested/20261115-060715/paired_vs_base.csv) |
| 5.2 | delay | nested | 24h | lightgbm | top5_nested | True |  | -0.036370 | 5 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_nested/20261115-060105/paired_vs_base.csv) |
| 5.2 | delay | nested | 6h | lightgbm | top5_nested | True |  | -0.075722 | 4 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_nested/20261115-060105/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 0min | xgboost_classifier | top5_flat | False |  | -0.002645 | 5 | False | [folds](results/advanced_5_2_xgboost_classifier_top5_flat/20261115-060637/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 1h | lightgbm_classifier | top5_flat | False |  | -0.002322 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_flat/20261115-060209/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 20min | xgboost_classifier | top5_flat | True |  | -0.003479 | 5 | True | [folds](results/advanced_5_2_xgboost_classifier_weather_top5_flat/20261115-061557/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 24h | lightgbm_classifier | top5_flat | True |  | -0.003242 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_flat/20261115-060817/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 6h | lightgbm_classifier | top5_flat | True |  | -0.001888 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_flat/20261115-060817/paired_vs_base.csv) |
| 5.2 | severity | nested | 0min | lightgbm_classifier | top5_nested | False |  | -0.004957 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.2 | severity | nested | 1h | lightgbm_classifier | top5_nested | False |  | -0.002698 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.2 | severity | nested | 20min | xgboost_classifier | top5_nested | False |  | -0.001477 | 5 | False | [folds](results/advanced_5_2_xgboost_classifier_top5_nested/20261115-060637/paired_vs_base.csv) |
| 5.2 | severity | nested | 24h | lightgbm_classifier | top5_nested | False |  | -0.003319 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.2 | severity | nested | 6h | lightgbm_classifier | top5_nested | False |  | -0.003799 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 0min | lightgbm | greedy | True |  | -0.036564 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 1h | lightgbm | greedy | True |  | -0.100464 | 4 | True | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 20min | lightgbm | greedy | True |  | -0.071008 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 24h | lightgbm | greedy | True |  | -0.002944 | 3 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 6h | lightgbm | greedy | True |  | -0.015177 | 2 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 0min | lightgbm_classifier | greedy | True |  | -0.002335 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261115-063353/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 1h | lightgbm_classifier | greedy | True |  | -0.003028 | 5 | True | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261115-063353/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 20min | lightgbm_classifier | greedy | True |  | -0.003252 | 5 | True | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261115-063353/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 24h | lightgbm_classifier | greedy | True |  | -0.000790 | 3 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261115-063353/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 6h | lightgbm_classifier | greedy | False |  | -0.000368 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_greedy/20261115-062543/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 0min | lightgbm | stack | True |  | -0.037648 | 4 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 1h | lightgbm | stack | True |  | -0.118885 | 4 | True | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 20min | lightgbm | stack | True |  | -0.082396 | 4 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 24h | lightgbm | stack | True |  | +0.003978 | 3 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 6h | lightgbm | stack | True |  | +0.000284 | 2 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 0min | lightgbm_classifier | stack | True |  | +0.000477 | 3 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 1h | lightgbm_classifier | stack | True |  | +0.000165 | 2 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 20min | lightgbm_classifier | stack | True |  | -0.000743 | 3 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 24h | lightgbm_classifier | stack | True |  | +0.002419 | 1 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 6h | lightgbm_classifier | stack | True |  | +0.002416 | 2 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 6.1 | delay | nested | 0min | catboost |  | True |  | -0.016223 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | delay | nested | 1h | catboost |  | True |  | -0.111373 | 5 | True | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | delay | nested | 20min | catboost |  | False |  | +0.012945 | 2 | False | [folds](results/advanced_6_1_catboost/20261114-113030/paired_vs_base.csv) |
| 6.1 | delay | nested | 24h | catboost |  | True |  | -0.034199 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | delay | nested | 6h | catboost |  | True |  | -0.029309 | 4 | False | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | severity | nested | 0min | catboost_classifier |  | True |  | -0.000027 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 1h | catboost_classifier |  | True |  | +0.000227 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 20min | catboost_classifier |  | True |  | +0.001187 | 2 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 24h | catboost_classifier |  | True |  | +0.000487 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 6h | catboost_classifier |  | True |  | +0.000960 | 1 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.2 | severity | nested | 0min | lightgbm_classifier | temperature | False |  | -0.000581 | 4 | False | [folds](results/advanced_6_2_lightgbm_classifier_temperature/20261115-065325/paired_vs_base.csv) |
| 6.2 | severity | nested | 1h | lightgbm_classifier | temperature | True |  | -0.000378 | 4 | False | [folds](results/advanced_6_2_lightgbm_classifier_weather_temperature/20261115-065502/paired_vs_base.csv) |
| 6.2 | severity | nested | 20min | lightgbm_classifier | temperature | False |  | -0.000366 | 3 | False | [folds](results/advanced_6_2_lightgbm_classifier_temperature/20261115-065325/paired_vs_base.csv) |
| 6.2 | severity | nested | 24h | lightgbm_classifier | temperature | False |  | -0.000196 | 4 | False | [folds](results/advanced_6_2_lightgbm_classifier_temperature/20261115-065325/paired_vs_base.csv) |
| 6.2 | severity | nested | 6h | catboost_classifier | temperature | False |  | -0.000459 | 4 | False | [folds](results/advanced_6_2_catboost_classifier_temperature/20261115-072723/paired_vs_base.csv) |
| 6.3 | delay | nested | 0min | ebm |  | False |  | +0.154187 | 0 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 1h | ebm |  | False |  | +0.214240 | 1 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 20min | ebm |  | False |  | +0.257095 | 1 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 24h | ebm |  | False |  | +0.112140 | 0 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 6h | ebm |  | False |  | +0.086063 | 1 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | severity | nested | 0min | ebm_classifier |  | False |  | +0.009253 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 1h | ebm_classifier |  | False |  | +0.011103 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 20min | ebm_classifier |  | False |  | +0.015673 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 24h | ebm_classifier |  | False |  | +0.010166 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 6h | ebm_classifier |  | False |  | +0.009921 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.5 | delay | nested | 0min | ngboost |  | False |  | -0.016746 | 2 | False | [folds](results/advanced_6_5_ngboost/20261114-220700/paired_vs_base.csv) |
| 6.5 | delay | nested | 1h | ngboost |  | False |  | -0.006736 | 3 | False | [folds](results/advanced_6_5_ngboost/20261114-220700/paired_vs_base.csv) |
| 6.5 | delay | nested | 20min | ngboost |  | False |  | +0.032981 | 3 | False | [folds](results/advanced_6_5_ngboost/20261114-220700/paired_vs_base.csv) |
| 6.5 | delay | nested | 24h | ngboost |  | False |  | +0.041729 | 0 | False | [folds](results/advanced_6_5_ngboost/20261114-220700/paired_vs_base.csv) |
| 6.5 | delay | nested | 6h | ngboost |  | False |  | +0.013887 | 2 | False | [folds](results/advanced_6_5_ngboost/20261114-220700/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 0min | lightgbm | log_shift | True |  | +0.088552 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261114-163707/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 1h | lightgbm | log_shift | True |  | +0.047583 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261114-163707/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 20min | lightgbm | log_shift | True |  | +0.056928 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261114-163707/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 24h | lightgbm | log_shift | True |  | +0.113728 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261114-163707/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 6h | lightgbm | log_shift | False |  | +0.204537 | 0 | False | [folds](results/advanced_6_6_lightgbm_log_shift/20261114-163652/paired_vs_base.csv) |
| 7.1 | delay | nested | 0min | mlp |  | False |  | +0.425626 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 1h | mlp |  | False |  | +0.349914 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 20min | mlp |  | False |  | +0.437380 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 24h | mlp |  | False |  | +0.461818 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 6h | mlp |  | False |  | +0.380262 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | severity | nested | 0min | mlp_classifier |  | False |  | +0.012711 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 1h | mlp_classifier |  | False |  | +0.015032 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 20min | mlp_classifier |  | False |  | +0.017340 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 24h | mlp_classifier |  | False |  | +0.016576 | 1 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 6h | mlp_classifier |  | False |  | +0.014627 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.2 | delay | nested | 0min | multitask |  | True |  | -0.379830 | 4 | True | [folds](results/advanced_7_2_multitask_weather/20261114-173952/paired_vs_base.csv) |
| 7.2 | delay | nested | 1h | multitask |  | False |  | -0.104996 | 4 | True | [folds](results/advanced_7_2_multitask/20261114-173930/paired_vs_base.csv) |
| 7.2 | delay | nested | 20min | multitask |  | True |  | -0.112804 | 3 | True | [folds](results/advanced_7_2_multitask_weather/20261114-173952/paired_vs_base.csv) |
| 7.2 | delay | nested | 24h | multitask |  | True |  | -0.156891 | 3 | True | [folds](results/advanced_7_2_multitask_weather/20261114-173952/paired_vs_base.csv) |
| 7.2 | delay | nested | 6h | multitask |  | True |  | -0.172642 | 3 | True | [folds](results/advanced_7_2_multitask_weather/20261114-173952/paired_vs_base.csv) |
| 7.2 | severity | nested | 0min | multitask_classifier |  | True |  | -0.000295 | 3 | False | [folds](results/advanced_7_2_multitask_classifier_weather/20261114-180244/paired_vs_base.csv) |
| 7.2 | severity | nested | 1h | multitask_classifier |  | False |  | -0.001589 | 2 | False | [folds](results/advanced_7_2_multitask_classifier/20261114-173937/paired_vs_base.csv) |
| 7.2 | severity | nested | 20min | multitask_classifier |  | False |  | +0.002582 | 1 | False | [folds](results/advanced_7_2_multitask_classifier/20261114-173937/paired_vs_base.csv) |
| 7.2 | severity | nested | 24h | multitask_classifier |  | False |  | -0.001436 | 2 | False | [folds](results/advanced_7_2_multitask_classifier/20261114-173937/paired_vs_base.csv) |
| 7.2 | severity | nested | 6h | multitask_classifier |  | False |  | -0.000931 | 2 | False | [folds](results/advanced_7_2_multitask_classifier/20261114-173937/paired_vs_base.csv) |
| 7.3 | delay | nested | 0min | tabgns |  | True |  | -0.654078 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261114-181426/paired_vs_base.csv) |
| 7.3 | delay | nested | 1h | tabgns |  | True |  | -0.602559 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261114-181426/paired_vs_base.csv) |
| 7.3 | delay | nested | 20min | tabgns |  | True |  | -0.423127 | 4 | True | [folds](results/advanced_7_3_tabgns_weather/20261114-181426/paired_vs_base.csv) |
| 7.3 | delay | nested | 24h | tabgns |  | True |  | -0.579759 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261114-181426/paired_vs_base.csv) |
| 7.3 | delay | nested | 6h | tabgns |  | True |  | -0.583640 | 4 | True | [folds](results/advanced_7_3_tabgns_weather/20261114-181426/paired_vs_base.csv) |
| 7.3 | severity | nested | 0min | tabgns_classifier |  | True |  | -0.006994 | 4 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261114-182100/paired_vs_base.csv) |
| 7.3 | severity | nested | 1h | tabgns_classifier |  | True |  | -0.003611 | 3 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261114-182100/paired_vs_base.csv) |
| 7.3 | severity | nested | 20min | tabgns_classifier |  | True |  | -0.007565 | 4 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261114-182100/paired_vs_base.csv) |
| 7.3 | severity | nested | 24h | tabgns_classifier |  | True |  | -0.011485 | 5 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261114-182100/paired_vs_base.csv) |
| 7.3 | severity | nested | 6h | tabgns_classifier |  | True |  | -0.013428 | 5 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261114-182100/paired_vs_base.csv) |
| 7.4 | delay | nested | 0min | tabm |  | True |  | -0.875879 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261114-192011/paired_vs_base.csv) |
| 7.4 | delay | nested | 1h | tabm |  | True |  | -0.738786 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261114-192011/paired_vs_base.csv) |
| 7.4 | delay | nested | 20min | realmlp |  | True |  | -0.910161 | 5 | True | [folds](results/advanced_7_4_realmlp_weather/20261114-183252/paired_vs_base.csv) |
| 7.4 | delay | nested | 24h | tabm |  | True |  | -0.720759 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261114-192011/paired_vs_base.csv) |
| 7.4 | delay | nested | 6h | tabm |  | True |  | -0.963230 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261114-192011/paired_vs_base.csv) |
| 7.4 | severity | nested | 0min | realmlp_classifier |  | True |  | -0.010289 | 4 | True | [folds](results/advanced_7_4_realmlp_classifier_weather/20261114-184206/paired_vs_base.csv) |
| 7.4 | severity | nested | 1h | tabm_classifier |  | True |  | -0.005026 | 3 | True | [folds](results/advanced_7_4_tabm_classifier_weather/20261114-202212/paired_vs_base.csv) |
| 7.4 | severity | nested | 20min | realmlp_classifier |  | True |  | -0.011573 | 3 | True | [folds](results/advanced_7_4_realmlp_classifier_weather/20261114-184206/paired_vs_base.csv) |
| 7.4 | severity | nested | 24h | realmlp_classifier |  | False |  | -0.005108 | 4 | True | [folds](results/advanced_7_4_realmlp_classifier/20261114-183023/paired_vs_base.csv) |
| 7.4 | severity | nested | 6h | realmlp_classifier |  | True |  | -0.019117 | 5 | True | [folds](results/advanced_7_4_realmlp_classifier_weather/20261114-184206/paired_vs_base.csv) |
| 7.5 | delay | nested | 0min | sequence_gru |  | False |  | +0.051973 | 2 | False | [folds](results/advanced_7_5_sequence_gru/20261114-210059/paired_vs_base.csv) |
| 7.5 | delay | nested | 1h | sequence_gru |  | False |  | +0.145940 | 1 | False | [folds](results/advanced_7_5_sequence_gru/20261114-210059/paired_vs_base.csv) |
| 7.5 | delay | nested | 20min | sequence_gru |  | True |  | +0.100473 | 2 | False | [folds](results/advanced_7_5_sequence_gru_weather/20261114-210128/paired_vs_base.csv) |
| 7.5 | severity | nested | 0min | sequence_gru_classifier |  | False |  | +0.000505 | 2 | False | [folds](results/advanced_7_5_sequence_gru_classifier/20261114-210139/paired_vs_base.csv) |
| 7.5 | severity | nested | 1h | sequence_gru_classifier |  | False |  | +0.000431 | 3 | False | [folds](results/advanced_7_5_sequence_gru_classifier/20261114-210139/paired_vs_base.csv) |
| 7.5 | severity | nested | 20min | sequence_gru_classifier |  | False |  | -0.000690 | 2 | False | [folds](results/advanced_7_5_sequence_gru_classifier/20261114-210139/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 0min | lightgbm |  | True |  | -0.014614 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 1h | lightgbm |  | True |  | -0.088032 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 20min | lightgbm |  | True |  | -0.075066 | 5 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 24h | lightgbm |  | True |  | -0.015705 | 3 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 6h | lightgbm |  | True |  | -0.034767 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 0min | lightgbm_classifier |  | True |  | -0.003529 | 4 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 1h | lightgbm_classifier |  | True |  | -0.003110 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 20min | lightgbm_classifier |  | True |  | -0.005264 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 24h | lightgbm_classifier |  | True |  | -0.001431 | 4 | False | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 6h | lightgbm_classifier |  | True |  | -0.002808 | 5 | False | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 0min | lightgbm | stacked | False |  | +0.023619 | 1 | False | [folds](results/advanced_8_3_lightgbm_stacked/20261114-164223/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 1h | lightgbm | init_model | False |  | +0.018429 | 1 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 20min | lightgbm | init_model | False |  | +0.016414 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 24h | lightgbm | init_model | False |  | +0.023085 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 6h | lightgbm | init_model | False |  | +0.022009 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 0min | lightgbm_classifier | stacked | False |  | +0.001886 | 0 | False | [folds](results/advanced_8_3_lightgbm_classifier_stacked/20261114-165804/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 1h | lightgbm_classifier | init_model | False |  | +0.001445 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 20min | lightgbm_classifier | init_model | False |  | +0.001088 | 2 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 24h | lightgbm_classifier | init_model | False |  | +0.000088 | 3 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 6h | lightgbm_classifier | init_model | False |  | -0.000295 | 4 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 0min | lightgbm | network_transfer | True |  | -0.005895 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261114-170025/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 1h | lightgbm | network_transfer | True |  | -0.002974 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261114-170025/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 20min | lightgbm | network_transfer | True |  | -0.009739 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261114-170025/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 0min | lightgbm_classifier | network_transfer | True |  | +0.000019 | 3 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261114-170353/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 1h | lightgbm_classifier | network_transfer | True |  | +0.000234 | 1 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261114-170353/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 20min | lightgbm_classifier | network_transfer | True |  | -0.000319 | 3 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261114-170353/paired_vs_base.csv) |

#### Step 3.2 leading permutation groups and TreeSHAP features

| task | prediction_time | group | score_increase | feature | mean_abs_shap | details |
| --- | --- | --- | --- | --- | --- | --- |
| severity | 0min | delay | 0.047600 | route_id | 0.179296 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 1h | route | 0.026975 | route_id | 0.146786 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 20min | route | 0.027343 | route_id | 0.167452 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 24h | route | 0.032388 | route_id | 0.162560 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 6h | recent_reliability | 0.030979 | route_id | 0.108209 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| delay | 0min | delay | 1.520474 | hour | 1.878026 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 1h | delay | 0.772937 | hour | 1.956902 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 20min | delay | 0.930739 | hour | 1.970728 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 24h | calendar | 0.628347 | hour | 2.583592 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 6h | recent_reliability | 0.748895 | service_mean_delay_min_28d | 1.701739 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |

#### Step 3.5 group choices at 24 h and 0 min

| task | weather | direction | prediction_time | group | folds_selected |
| --- | --- | --- | --- | --- | --- |
| delay | False | backward | 24h | calendar | 5 |
| delay | False | backward | 24h | recent_reliability | 5 |
| delay | False | backward | 24h | timetable | 5 |
| delay | False | backward | 24h | route | 1 |
| delay | False | backward | 0min | calendar | 5 |
| delay | False | backward | 0min | delay | 5 |
| delay | False | backward | 0min | recent_reliability | 5 |
| delay | False | backward | 0min | timetable | 5 |
| delay | False | forward | 24h | calendar | 5 |
| delay | False | forward | 24h | recent_reliability | 5 |
| delay | False | forward | 24h | timetable | 5 |
| delay | False | forward | 24h | route | 1 |
| delay | False | forward | 0min | calendar | 5 |
| delay | False | forward | 0min | delay | 5 |
| delay | False | forward | 0min | recent_reliability | 5 |
| delay | False | forward | 0min | timetable | 5 |
| delay | True | backward | 24h | recent_reliability | 5 |
| delay | True | backward | 24h | route | 3 |
| delay | True | backward | 24h | temperature | 3 |
| delay | True | backward | 24h | timetable | 5 |
| delay | True | backward | 24h | calendar | 3 |
| delay | True | backward | 24h | wind | 1 |
| delay | True | backward | 0min | delay | 5 |
| delay | True | backward | 0min | recent_reliability | 5 |
| delay | True | backward | 0min | temperature | 4 |
| delay | True | backward | 0min | timetable | 5 |
| delay | True | backward | 0min | calendar | 4 |
| delay | True | backward | 0min | wind | 1 |
| delay | True | forward | 24h | recent_reliability | 5 |
| delay | True | forward | 24h | route | 2 |
| delay | True | forward | 24h | temperature | 2 |
| delay | True | forward | 24h | timetable | 5 |
| delay | True | forward | 24h | calendar | 4 |
| delay | True | forward | 24h | wind | 1 |
| delay | True | forward | 0min | delay | 5 |
| delay | True | forward | 0min | recent_reliability | 5 |
| delay | True | forward | 0min | temperature | 4 |
| delay | True | forward | 0min | timetable | 5 |
| delay | True | forward | 0min | calendar | 4 |
| delay | True | forward | 0min | wind | 1 |
| severity | False | backward | 24h | calendar | 4 |
| severity | False | backward | 24h | recent_reliability | 5 |
| severity | False | backward | 24h | timetable | 5 |
| severity | False | backward | 24h | route | 2 |
| severity | False | backward | 0min | calendar | 1 |
| severity | False | backward | 0min | delay | 5 |
| severity | False | backward | 0min | recent_reliability | 5 |
| severity | False | backward | 0min | timetable | 5 |
| severity | False | backward | 0min | route | 3 |
| severity | False | forward | 24h | calendar | 4 |
| severity | False | forward | 24h | recent_reliability | 5 |
| severity | False | forward | 24h | timetable | 5 |
| severity | False | forward | 24h | route | 2 |
| severity | False | forward | 0min | calendar | 1 |
| severity | False | forward | 0min | delay | 5 |
| severity | False | forward | 0min | recent_reliability | 5 |
| severity | False | forward | 0min | timetable | 5 |
| severity | False | forward | 0min | route | 3 |
| severity | True | backward | 24h | calendar | 1 |
| severity | True | backward | 24h | recent_reliability | 5 |
| severity | True | backward | 24h | timetable | 5 |
| severity | True | backward | 24h | route | 3 |
| severity | True | backward | 24h | temperature | 4 |
| severity | True | backward | 24h | wind | 1 |
| severity | True | backward | 0min | delay | 5 |
| severity | True | backward | 0min | precipitation | 2 |
| severity | True | backward | 0min | recent_reliability | 5 |
| severity | True | backward | 0min | route | 4 |
| severity | True | backward | 0min | temperature | 4 |
| severity | True | backward | 0min | timetable | 5 |
| severity | True | backward | 0min | wind | 1 |
| severity | True | forward | 24h | calendar | 3 |
| severity | True | forward | 24h | recent_reliability | 5 |
| severity | True | forward | 24h | timetable | 5 |
| severity | True | forward | 24h | route | 2 |
| severity | True | forward | 24h | temperature | 2 |
| severity | True | forward | 24h | wind | 1 |
| severity | True | forward | 0min | delay | 5 |
| severity | True | forward | 0min | precipitation | 2 |
| severity | True | forward | 0min | recent_reliability | 5 |
| severity | True | forward | 0min | route | 4 |
| severity | True | forward | 0min | temperature | 4 |
| severity | True | forward | 0min | timetable | 5 |
| severity | True | forward | 0min | wind | 1 |

<!-- END ADVANCED CV -->
