# Modelling: shared cross-validation

Every model for Frankfurt (Main) Hbf → München Hbf is trained and scored by the same procedure, so that models differ only in their features and estimator. A model is a function that returns a fresh, unfitted scikit-learn estimator; `cv.py` does the rest.

Development uses the CV set (`../data/prediction_rows_cv.parquet`). The original weather models merge route weather from `../data/prediction_rows_weather_cv.parquet`. Loading the hold-out set needs `allow_holdout=True`.

## Advanced modelling

The earlier scores below use recorded-route inputs. `run_advanced.py` runs the CV experiments in [PLAN.md](PLAN.md) with the new cutoff-specific inputs. It keeps the same prediction rows and week folds. The recorded route fields are available only in the step 0.7 leakage comparison and the historical reproduction commands.

The advanced runner reads generated CV inputs in `../data/advanced/`. It requires the timestamped timetable and reconstructed live-delay data and checks that they cover the same prediction keys. It reads advanced weather only for the weather arm.

Download the 27 `monthly_processed_data_plan` files for July 2024 through September 2026 into `data/monthly_processed_data_plan/`. [Pinned plan archive](https://huggingface.co/datasets/piebro/deutsche-bahn-data/tree/d27b9c413536bb5fe393954371f572050e55176b/monthly_processed_data_plan)

The legacy fetcher used the requested timetable hour in plan filenames. Following the user decision on 5 November 2026, the primary build accepts these legacy fields regardless of fetch time, using the earliest plan per stop as its static schedule. Exact request stamps from 3 November 2025 remain cutoff-bound. The 45-minute bound applies to the separate change archive. [Legacy fetcher](https://github.com/piebro/deutsche-bahn-data/blob/f19888b770a68d5f9010adac33af70e400e2395d/fetch_data.py)

The prediction cohort supplies the planned Frankfurt departure and München arrival. The additional timetable fields come from the scheduled train's plan available at the cutoff. If the branch is unavailable, the weather pipeline retains the two endpoint temperatures and leaves the route aggregates missing. The live delay requires both a visible change report and a visible plan. For retained replacement trains it also requires a visible link to the scheduled service.

Recent-reliability candidates come from the CV roster during development. The holdout feature build adds its roster for the historical lookup. Both builds reconstruct past outcomes from snapshots available at each cutoff.

| Input coverage | 24 h | 6 h | 1 h | 20 min | 0 min |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV prediction rows | 17,466 | 17,380 | 17,114 | 16,994 | 16,947 |
| Planned route available | 10,302 | 10,269 | 10,861 | 11,223 | 11,419 |
| Weather branch available | 10,301 | 10,268 | 10,860 | 11,222 | 11,418 |
| Snapshot-based `current_delay` used by models | — | — | 6,046 | 6,788 | 8,323 |

```bash
.venv/bin/python frankfurt_munich/scheduled_timetable.py
.venv/bin/python frankfurt_munich/history_features.py --feature all
.venv/bin/python frankfurt_munich/network_state.py
.venv/bin/python frankfurt_munich/modeling/build_advanced_weather.py
.venv/bin/python frankfurt_munich/modeling/run_advanced.py 0.7 --jobs 4
.venv/bin/python frankfurt_munich/modeling/run_advanced.py 1.1 --jobs 4
.venv/bin/python frankfurt_munich/modeling/report_cv.py --write-readme
```

`run_advanced.py --help` lists the CV steps and filters. Fixed configurations come from the committed flat searches. Following the user decision on 5 November, remaining booster and CatBoost searches use 20 configurations inside each outer training set. Six completed 50-configuration baseline searches also supply budget-matched 20-configuration controls by reusing their fixed first 20 inner-score rows and refitting the outer models. The saved configurations link to the original searches. The reference table and each run configuration record the actual count. Final registration uses at most 20 configurations. Neural searches use eight. The `auto_groups` and `auto_representations` options to `--nested-variant` also select features inside each outer training set. A variant fixed after seeing the exploratory results retains that label even when its parameters are tuned again.

When running a subset of horizons with `--prediction-time`, set a separate `--results-dir`. Reports and registration select the latest timestamp in each run directory, so a partial run in the main results directory can replace the complete run in those reports.

The model-family screens use the cutoff-safe base columns for their learner comparisons. The nested booster feature-group policy evaluates the expanded Phase 2 inputs separately.

Steps 6.5 and 4.2 use a separate Python 3.13 environment with [requirements-distributions.txt](requirements-distributions.txt). NGBoost and LightGBMLSS constrain dependencies to older releases than the main environment uses.

The LightGBMLSS adapter standardizes delay within each training fold and returns the Gaussian mixture in minutes. Fitting uses the package's Gumbel-softmax relaxation. Prediction uses softmax weights from the fitted logits, giving a mean and CDF independent of prediction batch order. [LightGBMLSS API reference](https://statmixedml.github.io/LightGBMLSS/api/)

Step 8.3 uses railway inputs from the three auxiliary corridors. The builder drops ambiguous shared departures, replacements and additional endpoint stops. Each fit admits only source runs whose origin, run start, planned destination and reported outcome weeks are eligible. The inner and outer fits also exclude validation journey IDs. Step 8.4 applies the same run exclusion and checks the planned, reported and cutoff weeks of its network pairs.

```bash
.venv/bin/python frankfurt_munich/modeling/build_corridor_inputs.py
.venv/bin/python frankfurt_munich/modeling/network_transfer.py
.venv/bin/python frankfurt_munich/modeling/sequence_inputs.py
```

### Holdout evaluation

This branch contains the completed registry and evaluation. The commands below document the procedure used for that run. The registration command requires an unused registry path, and the evaluator refuses to overwrite the completed results.

The registration can re-tune arms in parallel after shared feature selection. It records the tuning environment and freezes the source and selected CV input fingerprints. Holdout builders verify the registered source. Evaluation verifies the registered CV inputs before scoring and records each arm’s interpreter and package versions. On retry, the scorer reuses completed arms when the committed registration and both splits' fingerprints match.

After a forced stop, resume with the same registry once the scorer and its arm workers have exited and the stale `.evaluation.lock` has been cleared.

After the CV steps are complete, `run_final.py register` ranks eligible pipelines by the railway-only CV score and writes one matched pair per task and horizon to `finalists.json`. Commit that file with the preregistration entry in the plan before generating holdout inputs.

```bash
.venv/bin/python frankfurt_munich/modeling/run_final.py register --jobs 4
# After committing the registry and the preregistration entry
.venv/bin/python frankfurt_munich/scheduled_timetable.py --split holdout
.venv/bin/python frankfurt_munich/history_features.py --split holdout
.venv/bin/python frankfurt_munich/modeling/build_advanced_weather.py --split holdout
```

Build any additional inputs named by the finalists with `--split holdout`. Use `history_features.py --feature reliability` for recent reliability, `network_state.py` for network state, and `sequence_inputs.py` for upstream sequences. Then run `run_final.py evaluate` once. If a distributional learner is registered, supply its pinned interpreter via `--distribution-python`. The evaluation saves scores, paired week-bootstrap intervals, subgroup errors, calibration, risk trajectories, and SHAP summaries in `results/advanced_holdout/`.

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
| [lightgbm](results/advanced_0_7_lightgbm/20261105-094317) | fixed | rmse | 24.475782 | 24.475782 | 24.061917 | 24.110516 | 23.871876 |
| [lightgbm_classifier](results/advanced_0_7_lightgbm_classifier/20261105-094449) | fixed | log_loss | 0.920441 | 0.911750 | 0.867040 | 0.848649 | 0.833305 |
| [lightgbm_classifier + weather](results/advanced_0_7_lightgbm_classifier_weather/20261105-094543) | fixed | log_loss | 0.926648 | 0.920599 | 0.870853 | 0.852692 | 0.837792 |
| [lightgbm + weather](results/advanced_0_7_lightgbm_weather/20261105-094320) | fixed | rmse | 24.596359 | 24.636556 | 24.162988 | 24.118915 | 23.886333 |
| [linear_regression](results/advanced_0_7_linear_regression/20261105-094255) | fixed | rmse | 24.725467 | 24.725467 | 24.306489 | 24.287113 | 24.075382 |
| [linear_regression + weather](results/advanced_0_7_linear_regression_weather/20261105-094257) | fixed | rmse | 24.741384 | 24.745659 | 24.329880 | 24.303906 | 24.091288 |
| [logistic_regression](results/advanced_0_7_logistic_regression/20261105-094300) | fixed | log_loss | 0.934693 | 0.925578 | 0.881563 | 0.862618 | 0.848278 |
| [logistic_regression + weather](results/advanced_0_7_logistic_regression_weather/20261105-094306) | fixed | log_loss | 0.936857 | 0.926824 | 0.881350 | 0.862513 | 0.849384 |
| [training_mean](results/advanced_0_7_training_mean/20261105-094254) | fixed | rmse | 25.427529 | 25.427529 | 25.427529 | 25.427529 | 25.427529 |
| [xgboost](results/advanced_0_7_xgboost/20261105-094512) | fixed | rmse | 24.511355 | 24.511355 | 24.085740 | 24.059483 | 23.813927 |
| [xgboost_classifier](results/advanced_0_7_xgboost_classifier/20261105-094634) | fixed | log_loss | 0.921781 | 0.912863 | 0.867956 | 0.851121 | 0.836399 |
| [xgboost_classifier + weather](results/advanced_0_7_xgboost_classifier_weather/20261105-094757) | fixed | log_loss | 0.923900 | 0.916941 | 0.869718 | 0.850681 | 0.839115 |
| [xgboost + weather](results/advanced_0_7_xgboost_weather/20261105-094544) | fixed | rmse | 24.516292 | 24.541405 | 24.131985 | 24.115664 | 23.883132 |

#### Step 0.8 nested reference scores

| Model | Search configs | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [lightgbm](results/advanced_0_8_lightgbm/20261105-170728) | 20 | rmse | 24.558833 | 24.558978 | 24.158170 | 24.148049 | 23.851491 |
| [lightgbm_classifier](results/advanced_0_8_lightgbm_classifier/20261105-170810) | 20 | log_loss | 0.927132 | 0.917266 | 0.870015 | 0.850262 | 0.837066 |
| [lightgbm_classifier + weather](results/advanced_0_8_lightgbm_classifier_weather/20261105-170925) | 20 | log_loss | 0.927180 | 0.919514 | 0.874094 | 0.852796 | 0.838998 |
| [lightgbm + weather](results/advanced_0_8_lightgbm_weather/20261105-170745) | 20 | rmse | 24.544852 | 24.667386 | 24.154075 | 24.117928 | 23.947489 |
| [xgboost](results/advanced_0_8_xgboost/20261105-170959) | 20 | rmse | 24.536433 | 24.568876 | 24.107252 | 24.096992 | 23.862371 |
| [xgboost_classifier](results/advanced_0_8_xgboost_classifier/20261105-185949) | 20 | log_loss | 0.923019 | 0.914149 | 0.870932 | 0.851310 | 0.836797 |
| [xgboost_classifier + weather](results/advanced_0_8_xgboost_classifier_weather/20261105-222105) | 20 | log_loss | 0.923900 | 0.917211 | 0.870635 | 0.851156 | 0.837468 |
| [xgboost + weather](results/advanced_0_8_xgboost_weather/20261105-171041) | 20 | rmse | 24.548736 | 24.610447 | 24.123494 | 24.115664 | 23.883132 |

#### Smallest paired differences by step, task and horizon

| step | task | evaluation | prediction_time | method | variant | weather | contrast | mean | folds_improved | reaches_minimum_gain | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | delay | exploratory | 0min | lightgbm | recorded_route | True |  | -0.151353 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261105-094332/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 1h | linear_regression | recorded_route | True |  | -0.126682 | 4 | True | [folds](results/advanced_0_7_linear_regression_weather_recorded_route/20261105-094300/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 20min | lightgbm | recorded_route | True |  | -0.152879 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261105-094332/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 24h | linear_regression | recorded_route | True |  | -0.148750 | 5 | True | [folds](results/advanced_0_7_linear_regression_weather_recorded_route/20261105-094300/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 6h | lightgbm | recorded_route | True |  | -0.158027 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261105-094332/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 0min | logistic_regression | recorded_route | True |  | -0.007223 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261105-094332/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 1h | logistic_regression | recorded_route | True |  | -0.006681 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261105-094332/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 20min | logistic_regression | recorded_route | True |  | -0.007240 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261105-094332/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 24h | logistic_regression | recorded_route | True |  | -0.007309 | 4 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261105-094332/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 6h | logistic_regression | recorded_route | False |  | -0.006960 | 4 | True | [folds](results/advanced_0_7_logistic_regression_recorded_route/20261105-094304/paired_vs_base.csv) |
| 0.8 | delay | nested | 0min | lightgbm | auto_groups | True |  | -0.359541 | 5 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | delay | nested | 1h | lightgbm | auto_groups | True |  | -0.305157 | 5 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | delay | nested | 20min | lightgbm | auto_groups | False |  | -0.348110 | 4 | True | [folds](results/advanced_0_8_lightgbm_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | delay | nested | 24h | xgboost | auto_groups | True |  | -0.154070 | 5 | True | [folds](results/advanced_0_8_xgboost_weather_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | delay | nested | 6h | lightgbm | auto_groups | True |  | -0.329586 | 5 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | severity | nested | 0min | lightgbm_classifier | auto_groups | False |  | -0.016190 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | severity | nested | 1h | lightgbm_classifier | auto_groups | True |  | -0.013546 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | severity | nested | 20min | lightgbm_classifier | auto_groups | True |  | -0.012910 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | severity | nested | 24h | lightgbm_classifier | auto_groups | False |  | -0.007606 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_auto_groups/20261106-133246/paired_vs_base.csv) |
| 0.8 | severity | nested | 6h | lightgbm_classifier | auto_groups | True |  | -0.013128 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261106-133246/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 0min | lightgbm | calendar_cyclic | False |  | -0.036489 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261105-095024/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 1h | lightgbm | calendar_integer | False |  | -0.048392 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_integer/20261105-094954/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 20min | lightgbm | calendar_cyclic | False |  | -0.036861 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261105-095024/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 24h | linear_regression | calendar_combined | False |  | -0.000093 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261105-095217/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 6h | linear_regression | calendar_combined | False |  | -0.000093 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261105-095217/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 0min | lightgbm_classifier | calendar_cyclic | False |  | -0.001084 | 3 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261105-095215/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 1h | lightgbm_classifier | calendar_cyclic | False |  | -0.000931 | 3 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261105-095215/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 20min | lightgbm_classifier | calendar_cyclic | False |  | -0.000935 | 3 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261105-095215/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 24h | logistic_regression | calendar_combined | False |  | +0.000032 | 3 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261105-095224/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 6h | logistic_regression | calendar_combined | False |  | +0.000004 | 3 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261105-095224/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 0min | linear_regression | day_of_year | False |  | -0.038747 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261105-095506/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 1h | linear_regression | day_of_year | False |  | -0.046288 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261105-095506/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 20min | linear_regression | day_of_year | False |  | -0.045714 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261105-095506/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 24h | linear_regression | day_of_year | False |  | -0.050984 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261105-095506/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 6h | linear_regression | day_of_year | False |  | -0.050984 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261105-095506/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 0min | logistic_regression | day_of_year | False |  | -0.002209 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261105-095512/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 1h | logistic_regression | day_of_year | False |  | -0.002301 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261105-095512/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 20min | logistic_regression | day_of_year | False |  | -0.002459 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261105-095512/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 24h | lightgbm_classifier | minute_of_day | False |  | -0.001847 | 3 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261105-095626/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 6h | lightgbm_classifier | minute_of_day | False |  | -0.001775 | 4 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261105-095626/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 0min | lightgbm | route_onehot | False |  | +0.007323 | 3 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261105-103821/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 1h | lightgbm | route_onehot | False |  | +0.005672 | 2 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261105-103821/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 20min | lightgbm | route_onehot | False |  | +0.025017 | 1 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261105-103821/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 24h | lightgbm | route_onehot | False |  | -0.062585 | 5 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261105-103821/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 6h | lightgbm | route_onehot | False |  | -0.062585 | 5 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261105-103821/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 0min | lightgbm_classifier | route_onehot | False |  | -0.001082 | 3 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261105-104006/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 1h | lightgbm_classifier | route_onehot | False |  | -0.001114 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261105-104006/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 20min | lightgbm_classifier | route_onehot | False |  | -0.001276 | 5 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261105-104006/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 24h | lightgbm_classifier | route_onehot | False |  | -0.001732 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261105-104006/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 6h | lightgbm_classifier | route_onehot | False |  | -0.001492 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261105-104006/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 0min | linear_regression | delay_clipped | False |  | -0.000096 | 4 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261105-095727/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 1h | linear_regression | delay_clipped | False |  | -0.001293 | 3 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261105-095727/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 20min | linear_regression | delay_clipped | False |  | -0.002367 | 3 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261105-095727/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 0min | logistic_regression | delay_bins | False |  | -0.000164 | 2 | False | [folds](results/advanced_1_4_logistic_regression_delay_bins/20261105-095934/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 1h | logistic_regression | delay_clipped | False |  | -0.000088 | 2 | False | [folds](results/advanced_1_4_logistic_regression_delay_clipped/20261105-095731/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 20min | logistic_regression | delay_clipped | False |  | -0.000171 | 4 | False | [folds](results/advanced_1_4_logistic_regression_delay_clipped/20261105-095731/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 0min | lightgbm | weather_extremes | True |  | +0.006412 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261105-100344/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 1h | lightgbm | weather_extremes | True |  | -0.033245 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261105-100344/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 20min | lightgbm | weather_extremes | True |  | +0.012658 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261105-100344/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 24h | lightgbm | weather_extremes | True |  | -0.043542 | 3 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261105-100344/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 6h | lightgbm | weather_extremes | True |  | +0.021090 | 1 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261105-100344/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 0min | lightgbm_classifier | weather_extremes | True |  | -0.002211 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261105-100506/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 1h | lightgbm_classifier | weather_extremes | True |  | -0.002805 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261105-100506/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 20min | lightgbm_classifier | weather_extremes | True |  | -0.002106 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261105-100506/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 24h | lightgbm_classifier | weather_extremes | True |  | -0.003114 | 4 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261105-100506/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 6h | lightgbm_classifier | weather_extremes | True |  | -0.000532 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261105-100506/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 0min | linear_regression | recent_reliability | False |  | -0.429938 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261105-105841/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 1h | linear_regression | recent_reliability | False |  | -0.436979 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261105-105841/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 20min | linear_regression | recent_reliability | False |  | -0.445106 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261105-105841/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 24h | linear_regression | recent_reliability | False |  | -0.261842 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261105-105841/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 6h | linear_regression | recent_reliability | False |  | -0.444428 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261105-105841/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 0min | logistic_regression | recent_reliability | False |  | -0.016383 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261105-105845/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 1h | logistic_regression | recent_reliability | False |  | -0.015453 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261105-105845/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 20min | logistic_regression | recent_reliability | False |  | -0.016412 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261105-105845/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 24h | logistic_regression | recent_reliability | False |  | -0.008616 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261105-105845/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 6h | logistic_regression | recent_reliability | False |  | -0.015793 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261105-105845/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 0min | lightgbm | planned_timetable | False |  | -0.102172 | 4 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261105-102639/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 1h | lightgbm | planned_timetable | False |  | -0.075283 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261105-102639/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 20min | lightgbm | planned_timetable | False |  | -0.111171 | 4 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261105-102639/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 24h | lightgbm | planned_timetable | False |  | -0.087388 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261105-102639/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 6h | lightgbm | planned_timetable | False |  | -0.087388 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261105-102639/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 0min | logistic_regression | planned_timetable | False |  | -0.002867 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261105-102627/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 1h | logistic_regression | planned_timetable | False |  | -0.002954 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261105-102627/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 20min | logistic_regression | planned_timetable | False |  | -0.002134 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261105-102627/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 24h | lightgbm_classifier | planned_timetable | False |  | -0.002021 | 3 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261105-102814/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 6h | lightgbm_classifier | planned_timetable | False |  | -0.001653 | 3 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261105-102814/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 0min | linear_regression | holidays | False |  | -0.004916 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261105-102820/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 1h | linear_regression | holidays | False |  | -0.010927 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261105-102820/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 20min | linear_regression | holidays | False |  | -0.008203 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261105-102820/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 24h | linear_regression | holidays | False |  | -0.014001 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261105-102820/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 6h | linear_regression | holidays | False |  | -0.014001 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261105-102820/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 0min | lightgbm_classifier | holidays | False |  | +0.000686 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261105-103009/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 1h | lightgbm_classifier | holidays | False |  | +0.000846 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261105-103009/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 20min | lightgbm_classifier | holidays | False |  | +0.001192 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261105-103009/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 24h | lightgbm_classifier | holidays | False |  | -0.000439 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261105-103009/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 6h | lightgbm_classifier | holidays | False |  | -0.000075 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261105-103009/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 0min | linear_regression | delay_dynamics | False |  | -0.082550 | 5 | False | [folds](results/advanced_2_4_linear_regression_delay_dynamics/20261105-103015/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 1h | linear_regression | delay_dynamics | False |  | -0.066089 | 5 | False | [folds](results/advanced_2_4_linear_regression_delay_dynamics/20261105-103015/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 20min | linear_regression | delay_dynamics | False |  | -0.105360 | 5 | True | [folds](results/advanced_2_4_linear_regression_delay_dynamics/20261105-103015/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 0min | logistic_regression | delay_dynamics | False |  | -0.003210 | 5 | True | [folds](results/advanced_2_4_logistic_regression_delay_dynamics/20261105-103018/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 1h | lightgbm_classifier | delay_dynamics | False |  | -0.001912 | 4 | False | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261105-103126/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 20min | logistic_regression | delay_dynamics | False |  | -0.002928 | 5 | False | [folds](results/advanced_2_4_logistic_regression_delay_dynamics/20261105-103018/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 0min | lightgbm | network_state | False |  | -0.006713 | 3 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261106-124214/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 1h | lightgbm | network_state | False |  | -0.023117 | 3 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261106-124214/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 20min | lightgbm | network_state | False |  | -0.025650 | 4 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261106-124214/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 24h | lightgbm | network_state | False |  | -0.003228 | 2 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261106-124214/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 6h | lightgbm | network_state | False |  | -0.013284 | 2 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261106-124214/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 0min | lightgbm_classifier | network_state | False |  | -0.001010 | 4 | False | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261106-124418/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 1h | lightgbm_classifier | network_state | False |  | -0.000280 | 3 | False | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261106-124418/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 20min | logistic_regression | network_state | False |  | -0.000330 | 4 | False | [folds](results/advanced_2_5_logistic_regression_network_state/20261106-124152/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 24h | logistic_regression | network_state | False |  | +0.000991 | 0 | False | [folds](results/advanced_2_5_logistic_regression_network_state/20261106-124152/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 6h | logistic_regression | network_state | False |  | +0.000640 | 1 | False | [folds](results/advanced_2_5_logistic_regression_network_state/20261106-124152/paired_vs_base.csv) |
| 2.all | delay | exploratory | 0min | linear_regression | all_groups | False |  | -0.491510 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261106-124258/paired_vs_base.csv) |
| 2.all | delay | exploratory | 1h | linear_regression | all_groups | False |  | -0.471916 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261106-124258/paired_vs_base.csv) |
| 2.all | delay | exploratory | 20min | linear_regression | all_groups | False |  | -0.516662 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261106-124258/paired_vs_base.csv) |
| 2.all | delay | exploratory | 24h | linear_regression | all_groups | False |  | -0.261990 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261106-124258/paired_vs_base.csv) |
| 2.all | delay | exploratory | 6h | linear_regression | all_groups | False |  | -0.425707 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261106-124258/paired_vs_base.csv) |
| 2.all | severity | exploratory | 0min | logistic_regression | all_groups | False |  | -0.017040 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261106-124328/paired_vs_base.csv) |
| 2.all | severity | exploratory | 1h | logistic_regression | all_groups | False |  | -0.014724 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261106-124328/paired_vs_base.csv) |
| 2.all | severity | exploratory | 20min | logistic_regression | all_groups | False |  | -0.016045 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261106-124328/paired_vs_base.csv) |
| 2.all | severity | exploratory | 24h | logistic_regression | all_groups | False |  | -0.004936 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261106-124328/paired_vs_base.csv) |
| 2.all | severity | exploratory | 6h | logistic_regression | all_groups | False |  | -0.013155 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261106-124328/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | lightgbm | drop_wind | True | full | -0.022800 | 4 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_wind/20261106-124713/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | lightgbm | add_delay | True | timetable-only | -0.918203 | 5 | True | [folds](results/advanced_3_1_lightgbm_weather_add_delay/20261105-111017/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | lightgbm | drop_wind | True | full | -0.026554 | 4 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_wind/20261106-124713/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | linear_regression | add_recent_reliability | True | timetable-only | -0.881939 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261105-110409/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | lightgbm | drop_wind | True | full | -0.024868 | 4 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_wind/20261106-124713/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | linear_regression | add_recent_reliability | True | timetable-only | -0.895016 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261105-110409/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | linear_regression | drop_temperature | True | full | -0.024725 | 4 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_temperature/20261106-124349/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | linear_regression | add_recent_reliability | True | timetable-only | -0.591364 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261105-110409/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | drop_temperature | True | full | -0.031307 | 5 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_temperature/20261106-124349/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | add_recent_reliability | True | timetable-only | -0.813482 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_recent_reliability/20261105-110409/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | drop_wind | True | full | -0.001296 | 3 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_wind/20261106-124617/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | add_recent_reliability | True | timetable-only | -0.033232 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261105-110548/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | lightgbm_classifier | drop_wind | True | full | -0.000790 | 3 | False | [folds](results/advanced_3_1_lightgbm_classifier_weather_drop_wind/20261106-125436/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | add_recent_reliability | True | timetable-only | -0.031956 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261105-110548/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | drop_wind | True | full | -0.001151 | 2 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_wind/20261106-124617/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | add_recent_reliability | True | timetable-only | -0.032649 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261105-110548/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | drop_wind | True | full | -0.001400 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_wind/20261106-124617/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | add_recent_reliability | True | timetable-only | -0.022537 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261105-110548/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | drop_temperature | True | full | -0.001158 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_temperature/20261106-124556/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | add_recent_reliability | True | timetable-only | -0.031509 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_recent_reliability/20261105-110548/paired_vs_base.csv) |
| 3.3 | delay | nested | 0min | linear_regression | backward | False |  | +0.041796 | 0 | False | [folds](results/advanced_3_3_linear_regression_backward/20261106-125257/paired_vs_base.csv) |
| 3.3 | delay | nested | 1h | linear_regression | forward | True |  | +0.036991 | 2 | False | [folds](results/advanced_3_3_linear_regression_weather_forward/20261106-125709/paired_vs_base.csv) |
| 3.3 | delay | nested | 20min | linear_regression | forward | True |  | -0.000928 | 2 | False | [folds](results/advanced_3_3_linear_regression_weather_forward/20261106-125709/paired_vs_base.csv) |
| 3.3 | delay | nested | 24h | linear_regression | backward | True |  | -0.032768 | 2 | False | [folds](results/advanced_3_3_linear_regression_weather_backward/20261106-130014/paired_vs_base.csv) |
| 3.3 | delay | nested | 6h | linear_regression | backward | False |  | +0.021515 | 2 | False | [folds](results/advanced_3_3_linear_regression_backward/20261106-125257/paired_vs_base.csv) |
| 3.3 | severity | nested | 0min | logistic_regression | forward | True |  | -0.001064 | 2 | False | [folds](results/advanced_3_3_logistic_regression_weather_forward/20261106-131029/paired_vs_base.csv) |
| 3.3 | severity | nested | 1h | logistic_regression | backward | True |  | +0.002325 | 1 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261106-132627/paired_vs_base.csv) |
| 3.3 | severity | nested | 20min | logistic_regression | backward | True |  | +0.001925 | 0 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261106-132627/paired_vs_base.csv) |
| 3.3 | severity | nested | 24h | logistic_regression | backward | True |  | -0.000113 | 2 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261106-132627/paired_vs_base.csv) |
| 3.3 | severity | nested | 6h | logistic_regression | backward | True |  | +0.002931 | 1 | False | [folds](results/advanced_3_3_logistic_regression_weather_backward/20261106-132627/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 0min | logistic_regression | l1 | True |  | -0.003192 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261106-131829/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 1h | logistic_regression | l1 | True |  | -0.003312 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261106-131829/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 20min | logistic_regression | l1 | True |  | -0.003201 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261106-131829/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 24h | logistic_regression | l1 | True |  | -0.003378 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261106-131829/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 6h | logistic_regression | l1 | True |  | -0.003159 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_l1/20261106-131829/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 0min | lightgbm | temperature | True |  | -0.037779 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261106-130052/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 1h | lightgbm | temperature | True |  | -0.031678 | 5 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261106-130052/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 20min | lightgbm | temperature | True |  | -0.048172 | 5 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261106-130052/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 24h | lightgbm | snow | True |  | -0.048509 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_snow/20261106-130112/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 6h | lightgbm | temperature | True |  | -0.011058 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261106-130052/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 0min | lightgbm_classifier | temperature | True |  | -0.001506 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261106-130457/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 1h | lightgbm_classifier | temperature | True |  | -0.000789 | 3 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261106-130457/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 20min | lightgbm_classifier | temperature | True |  | -0.000979 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261106-130457/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 24h | lightgbm_classifier | temperature | True |  | +0.000072 | 2 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261106-130457/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 6h | lightgbm_classifier | temperature | True |  | -0.000499 | 5 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_temperature/20261106-130457/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_class | False |  | +0.000170 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_cancellation_then_class/20261105-232913/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_class | True |  | -0.000957 | 3 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261105-233143/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_class | False |  | -0.000051 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_cancellation_then_class/20261105-232913/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_class | True |  | -0.000007 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261105-233143/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_class | True |  | +0.000206 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261105-233143/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 0min | quantile_forest_severity |  | True |  | +0.039528 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261105-232419/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 1h | quantile_forest_severity |  | True |  | +0.022594 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261105-232419/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 20min | quantile_forest_severity |  | True |  | +0.034061 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261105-232419/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 24h | quantile_forest_severity |  | True |  | +0.042189 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261105-232419/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 6h | quantile_forest_severity |  | True |  | +0.022049 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261105-232419/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_ordinal | False |  | +0.002039 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_cancellation_then_ordinal/20261105-233315/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_ordinal | True |  | -0.000208 | 3 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261105-233516/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.001519 | 1 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261105-233516/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.000793 | 2 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261105-233516/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_ordinal | True |  | +0.001689 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261105-233516/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 0min | lightgbm | mixture_of_experts | True |  | -0.030342 | 3 | False | [folds](results/advanced_4_4_lightgbm_weather_mixture_of_experts/20261105-233810/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 1h | lightgbm | mixture_of_experts | False |  | -0.044434 | 4 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261105-233645/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 20min | lightgbm | mixture_of_experts | True |  | -0.040355 | 4 | False | [folds](results/advanced_4_4_lightgbm_weather_mixture_of_experts/20261105-233810/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 24h | lightgbm | mixture_of_experts | False |  | -0.027983 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261105-233645/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 6h | lightgbm | mixture_of_experts | False |  | -0.027983 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261105-233645/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 0min | lightgbm | change_then_level | False |  | +0.024842 | 1 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261105-233828/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 1h | lightgbm | change_then_level | False |  | +0.020137 | 2 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261105-233828/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 20min | lightgbm | change_then_level | False |  | +0.037482 | 1 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261105-233828/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 0min | lightgbm | plus_neural | True |  | -0.055261 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261106-055745/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 1h | lightgbm | plus_neural | True |  | -0.075697 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261106-055745/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 20min | lightgbm | plus_catboost | True |  | -0.048021 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_catboost/20261106-055744/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 24h | lightgbm | plus_neural | True |  | -0.026422 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261106-055745/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 6h | lightgbm | plus_neural | True |  | -0.057906 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_neural/20261106-055745/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 0min | lightgbm_classifier | plus_neural | True |  | -0.003150 | 4 | True | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_neural/20261106-081317/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 1h | lightgbm_classifier | plus_catboost | True |  | -0.001617 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261106-081316/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 20min | lightgbm_classifier | plus_catboost | True |  | -0.002205 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261106-081316/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 24h | lightgbm_classifier | mean | True |  | -0.000810 | 3 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_mean/20261105-134208/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 6h | lightgbm_classifier | plus_neural | True |  | -0.001002 | 3 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_neural/20261106-081317/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 0min | xgboost | top5_flat | True |  | -0.073421 | 5 | False | [folds](results/advanced_5_2_xgboost_weather_top5_flat/20261106-003348/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 1h | lightgbm | top5_flat | True |  | -0.081333 | 5 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_flat/20261106-000304/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 20min | lightgbm | top5_flat | False |  | -0.071109 | 5 | False | [folds](results/advanced_5_2_lightgbm_top5_flat/20261105-231802/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 24h | xgboost | top5_flat | False |  | -0.074564 | 4 | False | [folds](results/advanced_5_2_xgboost_top5_flat/20261105-233714/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 6h | lightgbm | top5_flat | True |  | -0.103264 | 5 | True | [folds](results/advanced_5_2_lightgbm_weather_top5_flat/20261106-000304/paired_vs_base.csv) |
| 5.2 | delay | nested | 0min | lightgbm | top5_nested | True |  | -0.122527 | 5 | True | [folds](results/advanced_5_2_lightgbm_weather_top5_nested/20261106-000418/paired_vs_base.csv) |
| 5.2 | delay | nested | 1h | lightgbm | top5_nested | False |  | -0.074463 | 4 | False | [folds](results/advanced_5_2_lightgbm_top5_nested/20261105-231840/paired_vs_base.csv) |
| 5.2 | delay | nested | 20min | lightgbm | top5_nested | False |  | -0.114004 | 5 | True | [folds](results/advanced_5_2_lightgbm_top5_nested/20261105-231840/paired_vs_base.csv) |
| 5.2 | delay | nested | 24h | xgboost | top5_nested | True |  | -0.052861 | 3 | False | [folds](results/advanced_5_2_xgboost_weather_top5_nested/20261106-003647/paired_vs_base.csv) |
| 5.2 | delay | nested | 6h | xgboost | top5_nested | True |  | -0.102523 | 5 | True | [folds](results/advanced_5_2_xgboost_weather_top5_nested/20261106-003647/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 0min | xgboost_classifier | top5_flat | True |  | -0.003561 | 5 | True | [folds](results/advanced_5_2_xgboost_classifier_weather_top5_flat/20261106-010442/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 1h | lightgbm_classifier | top5_flat | False |  | -0.001575 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_flat/20261105-233206/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 20min | xgboost_classifier | top5_flat | False |  | -0.003051 | 5 | True | [folds](results/advanced_5_2_xgboost_classifier_top5_flat/20261105-235147/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 24h | lightgbm_classifier | top5_flat | True |  | -0.002975 | 4 | False | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_flat/20261106-002431/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 6h | lightgbm_classifier | top5_flat | True |  | -0.003670 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_flat/20261106-002431/paired_vs_base.csv) |
| 5.2 | severity | nested | 0min | lightgbm_classifier | top5_nested | False |  | -0.004472 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261105-233335/paired_vs_base.csv) |
| 5.2 | severity | nested | 1h | lightgbm_classifier | top5_nested | True |  | -0.002774 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_nested/20261106-002801/paired_vs_base.csv) |
| 5.2 | severity | nested | 20min | lightgbm_classifier | top5_nested | False |  | -0.002951 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261105-233335/paired_vs_base.csv) |
| 5.2 | severity | nested | 24h | lightgbm_classifier | top5_nested | False |  | -0.004607 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261105-233335/paired_vs_base.csv) |
| 5.2 | severity | nested | 6h | lightgbm_classifier | top5_nested | False |  | -0.002238 | 4 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261105-233335/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 0min | lightgbm | greedy | True |  | -0.032543 | 5 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261106-013617/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 1h | lightgbm | greedy | True |  | -0.069346 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261106-013617/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 20min | lightgbm | greedy | True |  | -0.019694 | 5 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261106-013617/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 24h | lightgbm | greedy | True |  | +0.019919 | 2 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261106-013617/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 6h | lightgbm | greedy | True |  | +0.015918 | 1 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261106-013617/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 0min | lightgbm_classifier | greedy | True |  | -0.001740 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261106-020147/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 1h | lightgbm_classifier | greedy | True |  | -0.000954 | 3 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261106-020147/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 20min | lightgbm_classifier | greedy | True |  | -0.001669 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261106-020147/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 24h | lightgbm_classifier | greedy | True |  | -0.000346 | 3 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261106-020147/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 6h | lightgbm_classifier | greedy | True |  | +0.000391 | 1 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261106-020147/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 0min | lightgbm | stack | False |  | -0.038289 | 3 | False | [folds](results/advanced_5_4_lightgbm_stack/20261106-020523/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 1h | lightgbm | stack | True |  | -0.081902 | 3 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261106-022319/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 20min | lightgbm | stack | True |  | -0.036484 | 4 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261106-022319/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 24h | lightgbm | stack | True |  | +0.033195 | 0 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261106-022319/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 6h | lightgbm | stack | True |  | +0.014701 | 2 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261106-022319/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 0min | lightgbm_classifier | stack | True |  | +0.000710 | 1 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261106-024834/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 1h | lightgbm_classifier | stack | False |  | +0.001856 | 2 | False | [folds](results/advanced_5_4_lightgbm_classifier_stack/20261106-021857/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 20min | lightgbm_classifier | stack | True |  | +0.000877 | 2 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261106-024834/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 24h | lightgbm_classifier | stack | True |  | +0.002962 | 1 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261106-024834/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 6h | lightgbm_classifier | stack | True |  | +0.002317 | 0 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261106-024834/paired_vs_base.csv) |
| 6.1 | delay | nested | 0min | catboost |  | True |  | -0.039996 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261105-233407/paired_vs_base.csv) |
| 6.1 | delay | nested | 1h | catboost |  | True |  | -0.007192 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261105-233407/paired_vs_base.csv) |
| 6.1 | delay | nested | 20min | catboost |  | False |  | -0.036251 | 4 | False | [folds](results/advanced_6_1_catboost/20261105-233457/paired_vs_base.csv) |
| 6.1 | delay | nested | 24h | catboost |  | True |  | +0.009378 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261105-233407/paired_vs_base.csv) |
| 6.1 | delay | nested | 6h | catboost |  | True |  | -0.076428 | 4 | False | [folds](results/advanced_6_1_catboost_weather/20261105-233407/paired_vs_base.csv) |
| 6.1 | severity | nested | 0min | catboost_classifier |  | True |  | +0.000971 | 2 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-081314/paired_vs_base.csv) |
| 6.1 | severity | nested | 1h | catboost_classifier |  | True |  | +0.000201 | 1 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-081314/paired_vs_base.csv) |
| 6.1 | severity | nested | 20min | catboost_classifier |  | True |  | -0.000119 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-081314/paired_vs_base.csv) |
| 6.1 | severity | nested | 24h | catboost_classifier |  | True |  | +0.000336 | 1 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-081314/paired_vs_base.csv) |
| 6.1 | severity | nested | 6h | catboost_classifier |  | True |  | +0.000928 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261106-081314/paired_vs_base.csv) |
| 6.2 | severity | nested | 0min | catboost_classifier | temperature | True |  | -0.000475 | 4 | False | [folds](results/advanced_6_2_catboost_classifier_weather_temperature/20261106-090637/paired_vs_base.csv) |
| 6.2 | severity | nested | 1h | catboost_classifier | temperature | True |  | -0.000429 | 4 | False | [folds](results/advanced_6_2_catboost_classifier_weather_temperature/20261106-090637/paired_vs_base.csv) |
| 6.2 | severity | nested | 20min | xgboost_classifier | temperature | False |  | -0.000414 | 3 | False | [folds](results/advanced_6_2_xgboost_classifier_temperature/20261105-232712/paired_vs_base.csv) |
| 6.2 | severity | nested | 24h | catboost_classifier | temperature | False |  | -0.000285 | 4 | False | [folds](results/advanced_6_2_catboost_classifier_temperature/20261106-083757/paired_vs_base.csv) |
| 6.2 | severity | nested | 6h | catboost_classifier | temperature | False |  | -0.000304 | 5 | False | [folds](results/advanced_6_2_catboost_classifier_temperature/20261106-083757/paired_vs_base.csv) |
| 6.3 | delay | nested | 0min | ebm |  | False |  | +0.227994 | 0 | False | [folds](results/advanced_6_3_ebm/20261105-225309/paired_vs_base.csv) |
| 6.3 | delay | nested | 1h | ebm |  | False |  | +0.101689 | 1 | False | [folds](results/advanced_6_3_ebm/20261105-225309/paired_vs_base.csv) |
| 6.3 | delay | nested | 20min | ebm |  | False |  | +0.138409 | 0 | False | [folds](results/advanced_6_3_ebm/20261105-225309/paired_vs_base.csv) |
| 6.3 | delay | nested | 24h | ebm |  | False |  | +0.087528 | 2 | False | [folds](results/advanced_6_3_ebm/20261105-225309/paired_vs_base.csv) |
| 6.3 | delay | nested | 6h | ebm |  | False |  | +0.087383 | 2 | False | [folds](results/advanced_6_3_ebm/20261105-225309/paired_vs_base.csv) |
| 6.3 | severity | nested | 0min | ebm_classifier |  | False |  | +0.011726 | 1 | False | [folds](results/advanced_6_3_ebm_classifier/20261105-230214/paired_vs_base.csv) |
| 6.3 | severity | nested | 1h | ebm_classifier |  | False |  | +0.010781 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261105-230214/paired_vs_base.csv) |
| 6.3 | severity | nested | 20min | ebm_classifier |  | False |  | +0.015184 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261105-230214/paired_vs_base.csv) |
| 6.3 | severity | nested | 24h | ebm_classifier |  | False |  | +0.006941 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261105-230214/paired_vs_base.csv) |
| 6.3 | severity | nested | 6h | ebm_classifier |  | False |  | +0.007665 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261105-230214/paired_vs_base.csv) |
| 6.5 | delay | nested | 0min | ngboost |  | True |  | +0.050357 | 2 | False | [folds](results/advanced_6_5_ngboost_weather/20261105-221901/paired_vs_base.csv) |
| 6.5 | delay | nested | 1h | ngboost |  | False |  | +0.048665 | 1 | False | [folds](results/advanced_6_5_ngboost/20261105-202458/paired_vs_base.csv) |
| 6.5 | delay | nested | 20min | ngboost |  | False |  | +0.023902 | 1 | False | [folds](results/advanced_6_5_ngboost/20261105-202458/paired_vs_base.csv) |
| 6.5 | delay | nested | 24h | ngboost |  | False |  | +0.038672 | 2 | False | [folds](results/advanced_6_5_ngboost/20261105-202458/paired_vs_base.csv) |
| 6.5 | delay | nested | 6h | ngboost |  | True |  | +0.019298 | 2 | False | [folds](results/advanced_6_5_ngboost_weather/20261105-221901/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 0min | lightgbm | log_shift | True |  | +0.115678 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261105-231453/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 1h | lightgbm | log_shift | True |  | +0.101776 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261105-231453/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 20min | lightgbm | log_shift | True |  | +0.096735 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261105-231453/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 24h | lightgbm | log_shift | True |  | +0.131838 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261105-231453/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 6h | lightgbm | log_shift | False |  | +0.218975 | 0 | False | [folds](results/advanced_6_6_lightgbm_log_shift/20261105-231414/paired_vs_base.csv) |
| 7.1 | delay | nested | 0min | mlp |  | False |  | +0.496748 | 0 | False | [folds](results/advanced_7_1_mlp/20261106-000818/paired_vs_base.csv) |
| 7.1 | delay | nested | 1h | mlp |  | False |  | +0.357092 | 0 | False | [folds](results/advanced_7_1_mlp/20261106-000818/paired_vs_base.csv) |
| 7.1 | delay | nested | 20min | mlp |  | False |  | +0.459160 | 0 | False | [folds](results/advanced_7_1_mlp/20261106-000818/paired_vs_base.csv) |
| 7.1 | delay | nested | 24h | mlp |  | False |  | +0.531205 | 0 | False | [folds](results/advanced_7_1_mlp/20261106-000818/paired_vs_base.csv) |
| 7.1 | delay | nested | 6h | mlp |  | True |  | +0.459027 | 1 | False | [folds](results/advanced_7_1_mlp_weather/20261106-003834/paired_vs_base.csv) |
| 7.1 | severity | nested | 0min | mlp_classifier |  | False |  | +0.015531 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261106-010950/paired_vs_base.csv) |
| 7.1 | severity | nested | 1h | mlp_classifier |  | True |  | +0.015151 | 0 | False | [folds](results/advanced_7_1_mlp_classifier_weather/20261106-014233/paired_vs_base.csv) |
| 7.1 | severity | nested | 20min | mlp_classifier |  | False |  | +0.020669 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261106-010950/paired_vs_base.csv) |
| 7.1 | severity | nested | 24h | mlp_classifier |  | False |  | +0.016402 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261106-010950/paired_vs_base.csv) |
| 7.1 | severity | nested | 6h | mlp_classifier |  | True |  | +0.012044 | 1 | False | [folds](results/advanced_7_1_mlp_classifier_weather/20261106-014233/paired_vs_base.csv) |
| 7.2 | delay | nested | 0min | multitask |  | False |  | -0.066440 | 3 | False | [folds](results/advanced_7_2_multitask/20261106-002552/paired_vs_base.csv) |
| 7.2 | delay | nested | 1h | multitask |  | True |  | -0.197748 | 4 | True | [folds](results/advanced_7_2_multitask_weather/20261106-005920/paired_vs_base.csv) |
| 7.2 | delay | nested | 20min | multitask |  | True |  | -0.075359 | 5 | False | [folds](results/advanced_7_2_multitask_weather/20261106-005920/paired_vs_base.csv) |
| 7.2 | delay | nested | 24h | multitask |  | False |  | -0.136672 | 4 | True | [folds](results/advanced_7_2_multitask/20261106-002552/paired_vs_base.csv) |
| 7.2 | delay | nested | 6h | multitask |  | False |  | -0.171043 | 4 | True | [folds](results/advanced_7_2_multitask/20261106-002552/paired_vs_base.csv) |
| 7.2 | severity | nested | 0min | multitask_classifier |  | False |  | +0.002395 | 2 | False | [folds](results/advanced_7_2_multitask_classifier/20261106-013358/paired_vs_base.csv) |
| 7.2 | severity | nested | 1h | multitask_classifier |  | False |  | -0.003509 | 4 | True | [folds](results/advanced_7_2_multitask_classifier/20261106-013358/paired_vs_base.csv) |
| 7.2 | severity | nested | 20min | multitask_classifier |  | True |  | -0.004996 | 2 | True | [folds](results/advanced_7_2_multitask_classifier_weather/20261106-020937/paired_vs_base.csv) |
| 7.2 | severity | nested | 24h | multitask_classifier |  | False |  | -0.001112 | 3 | False | [folds](results/advanced_7_2_multitask_classifier/20261106-013358/paired_vs_base.csv) |
| 7.2 | severity | nested | 6h | multitask_classifier |  | False |  | -0.001310 | 3 | False | [folds](results/advanced_7_2_multitask_classifier/20261106-013358/paired_vs_base.csv) |
| 7.3 | delay | nested | 0min | tabgns |  | False |  | -0.302133 | 4 | True | [folds](results/advanced_7_3_tabgns/20261106-053850/paired_vs_base.csv) |
| 7.3 | delay | nested | 1h | tabgns |  | True |  | -0.416853 | 4 | True | [folds](results/advanced_7_3_tabgns_weather/20261106-054731/paired_vs_base.csv) |
| 7.3 | delay | nested | 20min | tabgns |  | False |  | -0.362920 | 4 | True | [folds](results/advanced_7_3_tabgns/20261106-053850/paired_vs_base.csv) |
| 7.3 | delay | nested | 24h | tabgns |  | True |  | -0.298828 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261106-054731/paired_vs_base.csv) |
| 7.3 | delay | nested | 6h | tabgns |  | True |  | -0.376264 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261106-054731/paired_vs_base.csv) |
| 7.3 | severity | nested | 0min | tabgns_classifier |  | True |  | -0.007188 | 5 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261106-060200/paired_vs_base.csv) |
| 7.3 | severity | nested | 1h | tabgns_classifier |  | True |  | -0.002766 | 3 | False | [folds](results/advanced_7_3_tabgns_classifier_weather/20261106-060200/paired_vs_base.csv) |
| 7.3 | severity | nested | 20min | tabgns_classifier |  | True |  | -0.011862 | 4 | True | [folds](results/advanced_7_3_tabgns_classifier_weather/20261106-060200/paired_vs_base.csv) |
| 7.3 | severity | nested | 24h | tabgns_classifier |  | False |  | -0.002297 | 4 | False | [folds](results/advanced_7_3_tabgns_classifier/20261106-055405/paired_vs_base.csv) |
| 7.3 | severity | nested | 6h | tabgns_classifier |  | False |  | +0.002410 | 2 | False | [folds](results/advanced_7_3_tabgns_classifier/20261106-055405/paired_vs_base.csv) |
| 7.4 | delay | nested | 0min | tabm |  | True |  | -0.290664 | 3 | True | [folds](results/advanced_7_4_tabm_weather/20261106-024530/paired_vs_base.csv) |
| 7.4 | delay | nested | 1h | realmlp |  | True |  | -0.475752 | 4 | True | [folds](results/advanced_7_4_realmlp_weather/20261106-003452/paired_vs_base.csv) |
| 7.4 | delay | nested | 20min | realmlp |  | True |  | -0.261258 | 4 | True | [folds](results/advanced_7_4_realmlp_weather/20261106-003452/paired_vs_base.csv) |
| 7.4 | delay | nested | 24h | tabm |  | True |  | -0.499979 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261106-024530/paired_vs_base.csv) |
| 7.4 | delay | nested | 6h | tabm |  | True |  | -0.427632 | 5 | True | [folds](results/advanced_7_4_tabm_weather/20261106-024530/paired_vs_base.csv) |
| 7.4 | severity | nested | 0min | realmlp_classifier |  | True |  | -0.005237 | 4 | True | [folds](results/advanced_7_4_realmlp_classifier_weather/20261106-005935/paired_vs_base.csv) |
| 7.4 | severity | nested | 1h | tabm_classifier |  | True |  | -0.004401 | 4 | True | [folds](results/advanced_7_4_tabm_classifier_weather/20261106-051743/paired_vs_base.csv) |
| 7.4 | severity | nested | 20min | realmlp_classifier |  | True |  | -0.007386 | 4 | True | [folds](results/advanced_7_4_realmlp_classifier_weather/20261106-005935/paired_vs_base.csv) |
| 7.4 | severity | nested | 24h | realmlp_classifier |  | True |  | -0.007390 | 3 | True | [folds](results/advanced_7_4_realmlp_classifier_weather/20261106-005935/paired_vs_base.csv) |
| 7.4 | severity | nested | 6h | realmlp_classifier |  | False |  | +0.000358 | 4 | False | [folds](results/advanced_7_4_realmlp_classifier/20261106-004551/paired_vs_base.csv) |
| 7.5 | delay | nested | 0min | sequence_gru |  | True |  | +0.272619 | 1 | False | [folds](results/advanced_7_5_sequence_gru_weather/20261106-045121/paired_vs_base.csv) |
| 7.5 | delay | nested | 1h | sequence_gru |  | True |  | -0.033394 | 2 | False | [folds](results/advanced_7_5_sequence_gru_weather/20261106-045121/paired_vs_base.csv) |
| 7.5 | delay | nested | 20min | sequence_gru |  | True |  | +0.080433 | 1 | False | [folds](results/advanced_7_5_sequence_gru_weather/20261106-045121/paired_vs_base.csv) |
| 7.5 | severity | nested | 0min | sequence_gru_classifier |  | True |  | -0.000423 | 4 | False | [folds](results/advanced_7_5_sequence_gru_classifier_weather/20261106-030445/paired_vs_base.csv) |
| 7.5 | severity | nested | 1h | sequence_gru_classifier |  | True |  | -0.000559 | 2 | False | [folds](results/advanced_7_5_sequence_gru_classifier_weather/20261106-030445/paired_vs_base.csv) |
| 7.5 | severity | nested | 20min | sequence_gru_classifier |  | True |  | -0.010013 | 3 | True | [folds](results/advanced_7_5_sequence_gru_classifier_weather/20261106-030445/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 0min | lightgbm |  | False |  | -0.031131 | 4 | False | [folds](results/advanced_8_2_lightgbm_pooled/20261106-024902/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 1h | lightgbm |  | True |  | -0.082604 | 5 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261106-024918/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 20min | lightgbm |  | True |  | -0.056482 | 5 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261106-024918/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 24h | lightgbm |  | True |  | -0.061926 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261106-024918/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 6h | lightgbm |  | True |  | -0.139875 | 5 | True | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261106-024918/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 0min | lightgbm_classifier |  | True |  | -0.003528 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261106-025238/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 1h | lightgbm_classifier |  | True |  | -0.003331 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261106-025238/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 20min | lightgbm_classifier |  | True |  | -0.005273 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261106-025238/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 24h | lightgbm_classifier |  | True |  | -0.004499 | 4 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261106-025238/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 6h | lightgbm_classifier |  | True |  | -0.006225 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261106-025238/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 0min | lightgbm | init_model | False |  | -0.009523 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261106-025507/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 1h | lightgbm | init_model | False |  | -0.000772 | 3 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261106-025507/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 20min | lightgbm | init_model | False |  | -0.020509 | 3 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261106-025507/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 24h | lightgbm | stacked | False |  | +0.031896 | 1 | False | [folds](results/advanced_8_3_lightgbm_stacked/20261106-025851/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 6h | lightgbm | stacked | False |  | +0.036389 | 2 | False | [folds](results/advanced_8_3_lightgbm_stacked/20261106-025851/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 0min | lightgbm_classifier | init_model | False |  | +0.001672 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261106-031240/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 1h | lightgbm_classifier | init_model | False |  | +0.001122 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261106-031240/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 20min | lightgbm_classifier | init_model | False |  | +0.000350 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261106-031240/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 24h | lightgbm_classifier | init_model | False |  | +0.001035 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261106-031240/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 6h | lightgbm_classifier | init_model | False |  | +0.000598 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261106-031240/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 0min | lightgbm | network_transfer | True |  | -0.025385 | 5 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261106-033742/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 1h | lightgbm | network_transfer | True |  | -0.003637 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261106-033742/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 20min | lightgbm | network_transfer | True |  | -0.006648 | 4 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261106-033742/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 0min | lightgbm_classifier | network_transfer | True |  | -0.000574 | 4 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261106-034522/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 1h | lightgbm_classifier | network_transfer | True |  | -0.000466 | 4 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261106-034522/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 20min | lightgbm_classifier | network_transfer | True |  | +0.000588 | 1 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261106-034522/paired_vs_base.csv) |

#### Step 3.2 leading permutation groups and TreeSHAP features

| task | prediction_time | group | score_increase | feature | mean_abs_shap | details |
| --- | --- | --- | --- | --- | --- | --- |
| severity | 0min | delay | 0.047813 | hour | 0.154778 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261106-132728) |
| severity | 1h | recent_reliability | 0.026492 | hour | 0.129581 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261106-132728) |
| severity | 20min | delay | 0.031633 | hour | 0.156390 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261106-132728) |
| severity | 24h | calendar | 0.032740 | hour | 0.162770 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261106-132728) |
| severity | 6h | recent_reliability | 0.037833 | hour | 0.116251 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261106-132728) |
| delay | 0min | delay | 1.543640 | current_delay | 2.162328 | [folds](results/advanced_3_2_lightgbm_weather/20261106-125359) |
| delay | 1h | delay | 0.938680 | hour | 2.106649 | [folds](results/advanced_3_2_lightgbm_weather/20261106-125359) |
| delay | 20min | delay | 1.058777 | hour | 2.038173 | [folds](results/advanced_3_2_lightgbm_weather/20261106-125359) |
| delay | 24h | calendar | 0.634834 | hour | 2.559164 | [folds](results/advanced_3_2_lightgbm_weather/20261106-125359) |
| delay | 6h | recent_reliability | 0.917508 | service_mean_delay_min_28d | 1.890278 | [folds](results/advanced_3_2_lightgbm_weather/20261106-125359) |

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
| delay | True | backward | 24h | snow | 2 |
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
| delay | True | forward | 24h | snow | 2 |
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
| severity | False | backward | 0min | calendar | 3 |
| severity | False | backward | 0min | delay | 5 |
| severity | False | backward | 0min | recent_reliability | 5 |
| severity | False | backward | 0min | timetable | 5 |
| severity | False | backward | 0min | network_state | 4 |
| severity | False | backward | 0min | route | 4 |
| severity | False | forward | 24h | recent_reliability | 5 |
| severity | False | forward | 24h | timetable | 5 |
| severity | False | forward | 24h | calendar | 2 |
| severity | False | forward | 24h | route | 3 |
| severity | False | forward | 0min | calendar | 3 |
| severity | False | forward | 0min | delay | 5 |
| severity | False | forward | 0min | recent_reliability | 5 |
| severity | False | forward | 0min | timetable | 5 |
| severity | False | forward | 0min | network_state | 4 |
| severity | False | forward | 0min | route | 4 |
| severity | True | backward | 24h | recent_reliability | 5 |
| severity | True | backward | 24h | temperature | 3 |
| severity | True | backward | 24h | timetable | 5 |
| severity | True | backward | 24h | calendar | 2 |
| severity | True | backward | 24h | route | 3 |
| severity | True | backward | 24h | snow | 1 |
| severity | True | backward | 0min | delay | 5 |
| severity | True | backward | 0min | network_state | 5 |
| severity | True | backward | 0min | recent_reliability | 5 |
| severity | True | backward | 0min | timetable | 5 |
| severity | True | backward | 0min | wind | 1 |
| severity | True | backward | 0min | route | 4 |
| severity | True | backward | 0min | temperature | 2 |
| severity | True | backward | 0min | calendar | 2 |
| severity | True | forward | 24h | recent_reliability | 5 |
| severity | True | forward | 24h | temperature | 2 |
| severity | True | forward | 24h | timetable | 5 |
| severity | True | forward | 24h | calendar | 2 |
| severity | True | forward | 24h | route | 3 |
| severity | True | forward | 0min | calendar | 3 |
| severity | True | forward | 0min | delay | 5 |
| severity | True | forward | 0min | recent_reliability | 5 |
| severity | True | forward | 0min | timetable | 5 |
| severity | True | forward | 0min | network_state | 4 |
| severity | True | forward | 0min | route | 4 |
| severity | True | forward | 0min | temperature | 2 |

<!-- END ADVANCED CV -->

## Run

```bash
.venv/bin/python -m pip install -r frankfurt_munich/modeling/requirements.txt
.venv/bin/python -m pytest frankfurt_munich/modeling/tests          # about 4 min
.venv/bin/python frankfurt_munich/modeling/run_linear_models.py     # about 30 s
.venv/bin/python frankfurt_munich/modeling/run_tuned_models.py      # about 30 h, see below
```

`.venv` is at the repository root (Python 3.13).

| File | Purpose |
|---|---|
| `data.py` | `load_rows(prediction_time, weather=False)`: the prediction rows for one prediction time, joined with journey columns, plus derived columns and, with `weather=True`, the weather columns |
| `features.py` | Feature columns per prediction time |
| `models.py` | Model factories: logistic regression, linear regression, the two references, random forest, XGBoost, LightGBM and SVR regressors, and random forest, XGBoost and LightGBM classifiers |
| `metrics.py` | Class order, log loss, macro-F1, RMSE |
| `cv.py` | `cross_validate` (one prediction time) and `run_cv` (all prediction times, writes the results) |
| `run_linear_models.py` | Runs the six linear and reference models below |
| `tuning.py` | Random search of hyperparameters on the 5 folds (`run_search`) and the search spaces |
| `run_tuned_models.py` | Runs the fourteen tuned models below; `run_tuned_models.py lightgbm svr` runs only those |
| `tests/test_pipeline.py` | Checks of the method itself |
| `tests/test_tuned_models.py` | Checks of the tuned models' preprocessing and of the search |

To add a model: write a factory `make_x(numeric, categorical)` in `models.py` and call `run_cv("x", "severity" or "delay", make_x)`; the keyword argument `weather=True` adds the weather features.

## Method

- **One model per prediction time** (24 h, 6 h, 1 h, 20 min, 0 min), fitted on the rows of that prediction time only.
- **Folds:** the 5 folds of `../folds.py`, stored in the `fold` column: groups of whole ISO weeks, assigned by a greedy stratified split on `severity_class` and committed in `../split_weeks.csv` (see `../README.md`). Fold *k* validates on the rows with `fold == k` and trains on all other CV rows, including weeks after the validation weeks. There is no gap between training and validation: keeping each week on one side already separates most journeys that share weather and disruptions.
- **Fresh model per fold.** All preprocessing is inside the model's pipeline, so it is fitted on the training rows only.
- **Targets:**
  - `severity`: `severity_class`, with probabilities for the classes in the order `normal_minor, moderate, severe, canceled`.
  - `delay`: `arrival_delay_min`, completed journeys only for both training and scoring.
- **Metrics per fold:**
  - `severity`: multi-class log loss (for choosing models) and macro-F1, which uses the most probable class.
  - `delay`: RMSE in minutes.
- **Aggregation:** only the mean and standard deviation over the 5 folds. The validation predictions are saved but not pooled into one score.
- **Compare models only at the same prediction time.** The rows differ between prediction times: journeys whose cancellation is already announced are left out, so the reference scores also change with the prediction time.
- **Results:** each run writes `results/<name>/<timestamp>/` with:
  - `config.json`: task, model, whether weather is used, features and git commit;
  - `fold_metrics.csv`;
  - `summary.csv`;
  - `oof_predictions.parquet`: the validation predictions. These files are not committed;
  - tuned models only: `search_results.csv`, every configuration's fold scores, and in `config.json` the search space and the chosen configuration per prediction time.

## Features

| Feature | Type | Source |
|---|---|---|
| `current_delay` (1 h, 20 min, 0 min only) | numeric | prediction rows |
| `planned_travel_min` | numeric | planned München arrival − planned Frankfurt departure |
| `observed_stops`, `unobserved_sequence_positions` | numeric | journey file |
| `train_type`, `starts_at_origin`, `recorded_route` | categorical | journey file |
| `hour`, `weekday`, `month` | categorical | planned Frankfurt departure |
| Weather (`*_weather` models only): `origin_temperature_c`, `destination_temperature_c`, `route_temp_max_c`, `route_temp_min_c`, `route_temp_max_24h_c`, `route_temp_min_24h_c`, `route_precip_max_mm`, `route_precip_24h_mm`, `route_wind_max_ms`, `route_wind_max_24h_ms`, `route_gust_max_ms`, `route_gust_max_24h_ms`, `route_snow_depth_cm`, `snow_on_route_24h` (0/1) | numeric | `../weather_features.py`, see the Weather section of `../README.md` |

- **Weather** is the same set at every prediction time: the observed route weather at that time. The route is a fixed station list for the train number's usual branch (via Nürnberg or via Stuttgart), not the recorded stops, so it does not depend on the outcome. The column list is `FEATURE_COLUMNS` in `../weather_features.py`. `weather_hour_utc` (the DWD hour used) is not a feature. No weather value is missing; the columns are only standardised, with no other transform.
- **Stale weather file:** `load_rows` refuses a weather file whose rows (`journey_id`, `prediction_time`, `predicted_at`, in order) differ from the prediction rows, for example after `build_dataset.py` is re-run without `weather_features.py`.

- **Left out because they describe the outcome:**
  - reported times;
  - the cancellation columns, `cancellation_announced` and `cancellation_withdrawn`;
  - `shared_departure` and `is_replacement_train`;
  - the targets.
- **Caveat:** `recorded_route`, `observed_stops` and `unobserved_sequence_positions` come from the recorded stops, not the planned ones. The weather features don't (see above).
- **Linear models:**
  - missing `current_delay` becomes 0 plus a 0/1 missing indicator;
  - numeric columns are standardised;
  - categorical columns are one-hot encoded, and categories unseen in training get all-zero columns.
- **SVR:** the same columns as the linear models; the target is standardised inside the model.
- **Tree models:** missing `current_delay` stays missing and numeric columns are not scaled. The random forests get the one-hot columns; XGBoost and LightGBM split on the categorical columns natively, with the training rows' levels (an unseen level becomes missing). The classifiers use the same preprocessing as the regressors.

## Models

| Name | Task | Model |
|---|---|---|
| `class_shares` | severity | Reference: the training class shares for every journey |
| `logistic_regression` | severity | Multinomial logistic regression, L2 penalty with C = 1 (scikit-learn's default, not tuned) |
| `logistic_regression_weather` | severity | The same, with the weather features added |
| `training_mean` | delay | Reference: the training mean delay for every journey |
| `linear_regression` | delay | Ordinary least squares |
| `linear_regression_weather` | delay | The same, with the weather features added |
| `random_forest` | delay | Random forest, 300 trees, tuned |
| `xgboost` | delay | XGBoost (hist trees, squared error), tuned |
| `lightgbm` | delay | LightGBM (squared error), tuned |
| `svr` | delay | Support vector regression, RBF kernel, tuned |
| `*_weather` | delay | Each of the four, with the weather features added |
| `random_forest_classifier` | severity | Random forest classifier (Gini splits), 300 trees, tuned |
| `xgboost_classifier` | severity | XGBoost multi-class (softmax, log loss), tuned |
| `lightgbm_classifier` | severity | LightGBM multi-class (softmax, log loss), tuned |
| `*_classifier_weather` | severity | Each of the three, with the weather features added |

The railway-only and weather models use the same rows and folds, so they can be compared fold by fold.

## Tests

`tests/test_pipeline.py` checks the method, not the models:

- the data files follow the committed week assignment `../split_weeks.csv`, so a build with another split fails;
- no week is in both the hold-out set and the CV set or in two folds; the hold-out set is 15–25% of the weeks;
- the class shares of every fold are within 2 percentage points of the CV set's;
- at every prediction time, the rows' folds equal those in `../data/journeys_cv.parquet`;
- `current_delay` is used only from 1 h on;
- loading with weather adds only the weather columns (same rows, order and values), an out-of-date weather file is refused (CV files only), the weather features are never missing, and the weather hour is never after `predicted_at` or after the planned departure minus the real lead;
- weather features are used only by the `*_weather` models;
- loading the hold-out set raises an error;
- log loss agrees with scikit-learn's, and the class-share reference scores exactly the cross-entropy computed by hand;
- the probability columns are in the fixed class order;
- with the answer given as a feature, the logistic model reaches log loss < 0.05 and macro-F1 = 1, and the linear model reaches RMSE ≈ 0, so predictions line up with their rows;
- with shuffled labels, the full logistic model, with and without weather, does not beat the class shares, so there is no hidden leak.

`tests/test_tuned_models.py` checks the tuned models and the search:

- the 50 configurations of each search space are the same on every draw and lie inside the space;
- every model predicts for a route never seen in training and for a missing `current_delay`;
- XGBoost and LightGBM get the training rows' category levels;
- with the answer given as a feature, the tree models reach RMSE < 3 min with 99% of rows within 2 min, so predictions line up with their rows (trees cannot predict beyond the training delays);
- with shuffled delays, LightGBM and the random forest do not beat the training mean;
- the search chooses the configuration with the lowest mean RMSE and returns that configuration's fold scores and predictions;
- every classifier gives finite probabilities that sum to 1 over all four classes, also for an unseen route and a missing `current_delay`;
- the XGBoost classifier, which needs integer labels, keeps the original class labels;
- with the answer given as a feature, every classifier reaches log loss < 0.05 and macro-F1 = 1, so predictions line up with their rows;
- with shuffled classes, the LightGBM and random forest classifiers do not beat the class shares;
- the `severity` search chooses the configuration with the lowest mean log loss.

## Tuning

`tuning.py` runs a random search for each tuned model, prediction time and feature set. The classifiers use the same search spaces as the regressors and are chosen by log loss instead of RMSE:

- **50 configurations** drawn once per model by scikit-learn's `ParameterSampler` (seed 42). The railway-only and weather runs and all prediction times try the same 50.
- **Scored by the shared CV:** each configuration gets the same 5 week folds as every other model; the one with the lowest mean RMSE is chosen, and its fold scores and validation predictions are the run's result.
- **Flat search, not nested:** the same folds choose and score the configuration, so the chosen score is slightly optimistic. Choosing on 4 folds and scoring the 5th gave on average 0.04 min (at most 0.16 min) more RMSE, and 0.001 (at most 0.002) more log loss, than the chosen score.
- **Search spaces** (`SPACES` in `tuning.py`; U uniform, LU log-uniform):
  - random forest: `max_features` U[0.1, 1], `min_samples_leaf` LU[1, 50], `max_depth` one of None, 8, 16, 32, `max_samples` U[0.5, 1];
  - XGBoost: `n_estimators` LU[100, 2000], `learning_rate` LU[0.005, 0.3], `max_depth` 2–10, `min_child_weight` LU[1, 100], `subsample` U[0.5, 1], `colsample_bytree` U[0.3, 1], `reg_lambda` LU[0.001, 100], `reg_alpha` LU[0.001, 10];
  - LightGBM: as XGBoost, but `num_leaves` LU[4, 256] and `min_child_samples` LU[5, 200] instead of `max_depth` and `min_child_weight`, plus `cat_smooth` LU[1, 100];
  - SVR: `C` LU[0.1, 100], `epsilon` LU[0.01, 1] (in standard deviations of the delay), `gamma` LU[0.0001, 1].
- **Run time:** every model is single-threaded and the search is serial: about 17 CPU hours for the eight delay runs (SVR and the random forest take most of it) and 13 for the six severity runs. The committed runs evaluated configurations in parallel; because each model is single-threaded with a fixed seed, a serial run gives the same scores, only the `seconds` column differs.

## Results, 19 October 2026

Mean ± standard deviation over the 5 week folds, with the fixed weather route. The railway-only scores are identical to those of 15 October, so only the `*_weather` runs of 19 October are in `results/`.

**Severity**

| Prediction time | Log loss: class shares | Log loss: logistic | Log loss: logistic + weather | Macro-F1: class shares | Macro-F1: logistic | Macro-F1: logistic + weather |
|---|---|---|---|---|---|---|
| 24 h | 0.968 ± 0.009 | 0.927 ± 0.017 | 0.926 ± 0.021 | 0.193 ± 0.001 | 0.223 ± 0.003 | 0.233 ± 0.005 |
| 6 h | 0.958 ± 0.007 | 0.918 ± 0.016 | 0.919 ± 0.017 | 0.194 ± 0.001 | 0.222 ± 0.005 | 0.232 ± 0.005 |
| 1 h | 0.923 ± 0.009 | 0.824 ± 0.014 | 0.825 ± 0.016 | 0.196 ± 0.002 | 0.329 ± 0.009 | 0.336 ± 0.020 |
| 20 min | 0.905 ± 0.006 | 0.794 ± 0.016 | 0.796 ± 0.019 | 0.196 ± 0.001 | 0.350 ± 0.015 | 0.355 ± 0.012 |
| 0 min | 0.897 ± 0.005 | 0.778 ± 0.015 | 0.779 ± 0.017 | 0.197 ± 0.001 | 0.361 ± 0.014 | 0.368 ± 0.017 |

**Delay** (completed journeys)

| Prediction time | RMSE: training mean | RMSE: linear | RMSE: linear + weather |
|---|---|---|---|
| 24 h | 25.43 ± 0.91 | 24.56 ± 0.88 | 24.54 ± 0.87 |
| 6 h | 25.43 ± 0.91 | 24.56 ± 0.88 | 24.57 ± 0.86 |
| 1 h | 25.43 ± 0.91 | 22.92 ± 0.82 | 22.92 ± 0.76 |
| 20 min | 25.43 ± 0.91 | 22.37 ± 0.87 | 22.37 ± 0.81 |
| 0 min | 25.43 ± 0.91 | 22.03 ± 1.00 | 22.04 ± 0.95 |

- Both models beat their reference at every prediction time.
- The largest gain comes from 1 h on, when `current_delay` is available.
- **Weather adds almost nothing to these linear models.** Per fold (weather minus railway-only), log loss is 0.0013–0.0015 higher on average from 6 h on and 0.0003 lower at 24 h, macro-F1 0.005–0.011 higher, and RMSE changes by less than 0.03 min on average. All are far smaller than the spread between folds. Macro-F1 counts only the most probable class; log loss, which judges the probabilities and is used for choosing models, does not improve beyond noise. A linear model can only use each weather value as a straight line, so this does not yet show that weather is useless; non-linear models may use it better.
- The class-share reference always predicts `normal_minor`, so its macro-F1 stays near 0.2.
- The `delay` rows are the same at 24 h and 6 h (completed journeys, no `current_delay`). The railway-only scores are therefore identical; the weather scores differ slightly because the weather differs.
- The standard deviations are much smaller than with the earlier chronological folds, because every fold mixes weeks from the whole period. The scores are not comparable with the chronological results from 12 October.

## Results: tuned delay models, 26 October 2026

RMSE in minutes, mean ± standard deviation over the 5 week folds, completed journeys; railway only / railway + weather. Linear regression from the table above for comparison.

| Prediction time | Linear | Random forest | XGBoost | LightGBM | SVR |
|---|---|---|---|---|---|
| 24 h | 24.56 / 24.54 | 24.43 ± 0.92 / 24.54 ± 0.84 | 24.43 ± 0.85 / 24.38 ± 0.81 | 24.41 ± 0.91 / 24.39 ± 0.80 | 24.66 ± 0.97 / 24.52 ± 0.90 |
| 6 h | 24.56 / 24.57 | 24.43 ± 0.92 / 24.56 ± 0.85 | 24.43 ± 0.85 / 24.40 ± 0.83 | 24.41 ± 0.91 / 24.42 ± 0.81 | 24.66 ± 0.97 / 24.60 ± 0.96 |
| 1 h | 22.92 / 22.92 | 22.63 ± 0.85 / 22.84 ± 0.80 | 22.65 ± 0.76 / 22.72 ± 0.89 | 22.60 ± 0.79 / 22.69 ± 0.83 | 22.74 ± 0.91 / 22.82 ± 0.92 |
| 20 min | 22.37 / 22.37 | 22.00 ± 0.83 / 22.11 ± 0.84 | 21.97 ± 0.83 / 22.01 ± 0.96 | 21.92 ± 0.80 / 21.98 ± 0.89 | 22.05 ± 0.87 / 22.15 ± 0.87 |
| 0 min | 22.03 / 22.04 | 21.60 ± 0.87 / 21.74 ± 1.02 | 21.62 ± 0.83 / 21.68 ± 1.04 | 21.55 ± 0.88 / 21.61 ± 0.97 | 21.68 ± 0.97 / 21.77 ± 0.99 |

- **The tree models beat linear regression** by 0.1–0.5 min, most from 1 h on, when `current_delay` is known. Fold by fold the gain from 20 min on is consistent (t = 3.5–7.5 over the 5 paired folds).
- **The three tree models tie:** without weather they are within 0.08 min of each other, far less than the spread between folds. SVR is the weakest, and at 24 h and 6 h without weather it is worse than linear regression.
- **Weather does not help:** no gain larger than the fold noise (|t| ≤ 2.1 for every model and prediction time), and from 1 h on it makes every tuned model slightly worse. The same holds when the same configuration is compared with and without weather, so it is not an accident of the search.
- Journeys with 60+ min delay cause about 60% of the squared error; all models predict them far too low.

## Results: tuned severity classifiers, 31 October 2026

Mean ± standard deviation over the 5 week folds; railway only / railway + weather. Logistic regression from the 19 October table for comparison.

**Log loss** (lower is better; used for choosing models)

| Prediction time | Logistic | Random forest | XGBoost | LightGBM |
|---|---|---|---|---|
| 24 h | 0.927 / 0.926 | 0.923 ± 0.014 / 0.927 ± 0.014 | 0.919 ± 0.013 / 0.917 ± 0.014 | 0.918 ± 0.011 / 0.920 ± 0.015 |
| 6 h | 0.918 / 0.919 | 0.914 ± 0.014 / 0.920 ± 0.014 | 0.910 ± 0.013 / 0.910 ± 0.012 | 0.910 ± 0.012 / 0.913 ± 0.013 |
| 1 h | 0.824 / 0.825 | 0.816 ± 0.013 / 0.822 ± 0.013 | 0.811 ± 0.012 / 0.814 ± 0.013 | 0.810 ± 0.012 / 0.814 ± 0.013 |
| 20 min | 0.794 / 0.796 | 0.782 ± 0.012 / 0.786 ± 0.013 | 0.777 ± 0.012 / 0.780 ± 0.012 | 0.777 ± 0.011 / 0.778 ± 0.010 |
| 0 min | 0.778 / 0.779 | 0.762 ± 0.013 / 0.768 ± 0.012 | 0.759 ± 0.012 / 0.763 ± 0.011 | 0.759 ± 0.011 / 0.762 ± 0.010 |

**Macro-F1** (higher is better; the most probable class)

| Prediction time | Logistic | Random forest | XGBoost | LightGBM |
|---|---|---|---|---|
| 24 h | 0.223 / 0.233 | 0.211 ± 0.006 / 0.213 ± 0.016 | 0.229 ± 0.005 / 0.219 ± 0.012 | 0.222 ± 0.007 / 0.247 ± 0.013 |
| 6 h | 0.222 / 0.232 | 0.210 ± 0.005 / 0.203 ± 0.010 | 0.233 ± 0.005 / 0.214 ± 0.013 | 0.222 ± 0.007 / 0.210 ± 0.013 |
| 1 h | 0.329 / 0.336 | 0.336 ± 0.019 / 0.323 ± 0.017 | 0.336 ± 0.018 / 0.334 ± 0.018 | 0.350 ± 0.019 / 0.327 ± 0.017 |
| 20 min | 0.350 / 0.355 | 0.358 ± 0.017 / 0.359 ± 0.017 | 0.381 ± 0.023 / 0.362 ± 0.015 | 0.385 ± 0.019 / 0.375 ± 0.022 |
| 0 min | 0.361 / 0.368 | 0.371 ± 0.014 / 0.373 ± 0.015 | 0.388 ± 0.023 / 0.377 ± 0.020 | 0.390 ± 0.015 / 0.381 ± 0.016 |

- **XGBoost and LightGBM beat logistic regression** at every prediction time, on every fold without weather: log loss 0.007–0.009 lower at 24 h and 6 h and 0.018–0.019 lower at 0 min (t = 3–7 over the 5 paired folds). As for delay, the gain grows once `current_delay` is known.
- **The two boosters tie:** within 0.0011 log loss without weather, 0.003 with. The random forest is weaker: at 24 h and 6 h it beats logistic regression on only 4 of 5 folds (t ≈ 1.6), and with weather it is no better than logistic regression there. Random forest probabilities tend to be too cautious (see `literature_review/classification_models.md`), which log loss penalises; this was not checked here.
- **Weather does not help:** it lowers log loss only for XGBoost at 24 h and 6 h, by at most 0.002 (|t| ≤ 1); otherwise it raises it by up to 0.007, most for the random forest (t up to 2.7). Macro-F1 also falls with weather for both boosters from 6 h on (XGBoost also at 24 h); LightGBM at 24 h (+0.025) is the exception.
- **Macro-F1 stays low before departure:** about 0.22 at 24 h and 6 h, as for logistic regression: the most probable class is `normal_minor` for 94–98% of journeys (62.8% are). It rises to 0.39 at 0 min, where `normal_minor` is still the most probable class for 83–84%.
