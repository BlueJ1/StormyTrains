# Modelling: shared cross-validation

Every model for Frankfurt (Main) Hbf → München Hbf is trained and scored by the same procedure, so that models differ only in their features and estimator. A model is a function that returns a fresh, unfitted scikit-learn estimator; `cv.py` does the rest.

Only the CV set is used (`../data/prediction_rows_cv.parquet`). The weather models merge on the route weather from `../data/prediction_rows_weather_cv.parquet`; the railway-only models never read that file. Loading the hold-out set needs `allow_holdout=True`.

## Advanced modelling

The earlier scores below use recorded-route inputs. `run_advanced.py` runs the CV experiments in [PLAN.md](PLAN.md) with the new cutoff-specific inputs. It keeps the same prediction rows and week folds. The recorded route fields are available only in the step 0.7 leakage comparison and the historical reproduction commands.

The advanced runner reads generated CV inputs in `../data/advanced/`. It requires the timestamped timetable and live data and checks that they cover the same prediction keys. It does not fall back to the monthly file's latest delay. Railway-only runs do not read the advanced weather file.

Download the 27 `monthly_processed_data_plan` files for July 2024 through September 2026 from the [pinned plan archive](https://huggingface.co/datasets/piebro/deutsche-bahn-data/tree/d27b9c413536bb5fe393954371f572050e55176b/monthly_processed_data_plan). The default input directory is `data/monthly_processed_data_plan/`. The [legacy fetcher](https://github.com/piebro/deutsche-bahn-data/blob/f19888b770a68d5f9010adac33af70e400e2395d/fetch_data.py) used the requested timetable hour in plan filenames. The primary build leaves those values missing and accepts exact request stamps from 3 November 2025. The 45-minute bound in the plan applies to the separate change archive.

The prediction cohort supplies the planned Frankfurt departure and München arrival. The additional timetable fields come from the scheduled train's plan available at the cutoff. If the branch is unavailable, the weather pipeline retains the two endpoint temperatures and leaves the route aggregates missing. The live delay requires both a visible change report and a visible plan. For retained replacement trains it also requires a visible link to the scheduled service.

| Input coverage | 24 h | 6 h | 1 h | 20 min | 0 min |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV prediction rows | 17,466 | 17,380 | 17,114 | 16,994 | 16,947 |
| Scheduled branch and route available | 0 | 0 | 728 | 1,151 | 1,368 |
| Snapshot-based `current_delay` available for modelling | — | — | 2,301 | 2,875 | 3,142 |

```bash
.venv/bin/python frankfurt_munich/scheduled_timetable.py
.venv/bin/python frankfurt_munich/history_features.py --feature all
.venv/bin/python frankfurt_munich/network_state.py
.venv/bin/python frankfurt_munich/modeling/build_advanced_weather.py
.venv/bin/python frankfurt_munich/modeling/run_advanced.py 0.7 --jobs 4
.venv/bin/python frankfurt_munich/modeling/run_advanced.py 1.1 --jobs 4
.venv/bin/python frankfurt_munich/modeling/report_cv.py --write-readme
```

`run_advanced.py --help` lists the CV steps and filters. Fixed configurations come from the committed flat searches. Step 0.8 uses 50 configurations inside each outer training set. The `auto_groups` and `auto_representations` options to `--nested-variant` also select features inside each outer training set. A variant fixed after seeing the exploratory results retains that label even when its parameters are tuned again.

Steps 6.5 and 4.2 use a separate Python 3.13 environment with [requirements-distributions.txt](requirements-distributions.txt). NGBoost and LightGBMLSS constrain dependencies to older releases than the main environment uses.

Step 8.3 uses railway inputs from the three auxiliary corridors. The builder drops ambiguous shared departures, replacements and additional endpoint stops. Each fit admits only source runs whose origin, run start, planned destination and reported outcome weeks are eligible. The inner and outer fits also exclude validation journey IDs. Step 8.4 applies the same run exclusion and checks the planned, reported and cutoff weeks of its network pairs.

```bash
.venv/bin/python frankfurt_munich/modeling/build_corridor_inputs.py
.venv/bin/python frankfurt_munich/modeling/network_transfer.py
.venv/bin/python frankfurt_munich/modeling/sequence_inputs.py
```

<!-- BEGIN ADVANCED CV -->

### Advanced CV results

Scores use the five committed CV week folds. Positive paired differences favor the reference.

#### Step 0.7 reference scores

| Model | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- |
| [class_shares](results/advanced_0_7_class_shares/20261104-221643) | log_loss | 0.968037 | 0.958148 | 0.923149 | 0.904638 | 0.896917 |
| [lightgbm](results/advanced_0_7_lightgbm/20261104-224118) | rmse | 24.675104 | 24.675104 | 24.328226 | 24.308444 | 24.258947 |
| [lightgbm_classifier](results/advanced_0_7_lightgbm_classifier/20261104-224242) | log_loss | 0.945820 | 0.936088 | 0.875056 | 0.855717 | 0.846472 |
| [lightgbm_classifier + weather](results/advanced_0_7_lightgbm_classifier_weather/20261104-224300) | log_loss | 0.925683 | 0.921648 | 0.875382 | 0.856994 | 0.847314 |
| [lightgbm + weather](results/advanced_0_7_lightgbm_weather/20261104-224120) | rmse | 24.568290 | 24.614369 | 24.319954 | 24.264246 | 24.204176 |
| [linear_regression](results/advanced_0_7_linear_regression/20261104-224102) | rmse | 24.754142 | 24.754142 | 24.492036 | 24.469500 | 24.413992 |
| [linear_regression + weather](results/advanced_0_7_linear_regression_weather/20261104-224103) | rmse | 24.721331 | 24.716158 | 24.461115 | 24.426417 | 24.386380 |
| [logistic_regression](results/advanced_0_7_logistic_regression/20261104-224106) | log_loss | 0.938133 | 0.928956 | 0.888334 | 0.869152 | 0.859676 |
| [logistic_regression + weather](results/advanced_0_7_logistic_regression_weather/20261104-224110) | log_loss | 0.936987 | 0.927120 | 0.888229 | 0.867654 | 0.858925 |
| [training_mean](results/advanced_0_7_training_mean/20261104-221643) | rmse | 25.427529 | 25.427529 | 25.427529 | 25.427529 | 25.427529 |
| [xgboost](results/advanced_0_7_xgboost/20261104-224147) | rmse | 24.579947 | 24.579947 | 24.356854 | 24.321870 | 24.270758 |
| [xgboost_classifier](results/advanced_0_7_xgboost_classifier/20261104-224328) | log_loss | 0.925556 | 0.916071 | 0.875978 | 0.857372 | 0.848460 |
| [xgboost_classifier + weather](results/advanced_0_7_xgboost_classifier_weather/20261104-224441) | log_loss | 0.924406 | 0.916988 | 0.875484 | 0.855575 | 0.847770 |
| [xgboost + weather](results/advanced_0_7_xgboost_weather/20261104-224238) | rmse | 24.509518 | 24.463382 | 24.319560 | 24.264176 | 24.201848 |

#### Step 0.8 nested reference scores

| Model | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- |
| [lightgbm](results/advanced_0_8_lightgbm/20261104-234257) | rmse | 24.549264 | 24.589544 | 24.335040 | 24.312546 | 24.250788 |
| [lightgbm + weather](results/advanced_0_8_lightgbm_weather/20261105-001256) | rmse | 24.557616 | 24.467488 | 24.327460 | 24.274753 | 24.220357 |
| [xgboost](results/advanced_0_8_xgboost/20261105-000345) | rmse | 24.602398 | 24.629562 | 24.366489 | 24.325133 | 24.280402 |

#### Smallest paired differences by step, task and horizon

| step | task | evaluation | prediction_time | method | variant | weather | contrast | mean | folds_improved | reaches_minimum_gain | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | delay | exploratory | 0min | lightgbm | recorded_route | True |  | -0.139061 | 4 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261104-224122/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 1h | linear_regression | recorded_route | True |  | -0.149100 | 4 | True | [folds](results/advanced_0_7_linear_regression_weather_recorded_route/20261104-224105/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 20min | lightgbm | recorded_route | True |  | -0.143934 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261104-224122/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 24h | lightgbm | recorded_route | True |  | -0.207831 | 5 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261104-224122/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 6h | lightgbm | recorded_route | False |  | -0.196247 | 5 | True | [folds](results/advanced_0_7_lightgbm_recorded_route/20261104-224123/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 0min | logistic_regression | recorded_route | True |  | -0.008216 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261104-224126/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 1h | logistic_regression | recorded_route | False |  | -0.008637 | 5 | True | [folds](results/advanced_0_7_logistic_regression_recorded_route/20261104-224110/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 20min | logistic_regression | recorded_route | True |  | -0.008558 | 5 | True | [folds](results/advanced_0_7_logistic_regression_weather_recorded_route/20261104-224126/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 24h | lightgbm_classifier | recorded_route | False |  | -0.025361 | 5 | True | [folds](results/advanced_0_7_lightgbm_classifier_recorded_route/20261104-224256/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 6h | lightgbm_classifier | recorded_route | False |  | -0.024343 | 5 | True | [folds](results/advanced_0_7_lightgbm_classifier_recorded_route/20261104-224256/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 0min | lightgbm | calendar_combined | False |  | -0.011592 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_combined/20261104-224805/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 1h | lightgbm | calendar_cyclic | False |  | -0.042352 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261104-224558/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 20min | lightgbm | calendar_combined | False |  | -0.008622 | 3 | False | [folds](results/advanced_1_1_lightgbm_calendar_combined/20261104-224805/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 24h | lightgbm | calendar_cyclic | False |  | -0.092723 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261104-224558/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 6h | lightgbm | calendar_cyclic | False |  | -0.092723 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261104-224558/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 0min | lightgbm_classifier | calendar_cyclic | False |  | -0.000972 | 3 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261104-224734/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 1h | lightgbm_classifier | calendar_cyclic | False |  | -0.000541 | 2 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261104-224734/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 20min | lightgbm_classifier | calendar_cyclic | False |  | -0.000571 | 3 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261104-224734/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 24h | lightgbm_classifier | calendar_cyclic | False |  | -0.018779 | 5 | True | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261104-224734/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 6h | lightgbm_classifier | calendar_cyclic | False |  | -0.017964 | 5 | True | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261104-224734/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 0min | linear_regression | day_of_year | False |  | -0.042086 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261104-225013/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 1h | linear_regression | day_of_year | False |  | -0.043506 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261104-225013/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 20min | linear_regression | day_of_year | False |  | -0.043904 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261104-225013/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 24h | lightgbm | minute_of_day | False |  | -0.076194 | 5 | False | [folds](results/advanced_1_2_lightgbm_minute_of_day/20261104-225012/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 6h | lightgbm | minute_of_day | False |  | -0.076194 | 5 | False | [folds](results/advanced_1_2_lightgbm_minute_of_day/20261104-225012/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 0min | logistic_regression | day_of_year | False |  | -0.002424 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261104-225020/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 1h | logistic_regression | day_of_year | False |  | -0.002480 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261104-225020/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 20min | logistic_regression | day_of_year | False |  | -0.002607 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261104-225020/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 24h | lightgbm_classifier | day_of_year | False |  | -0.005019 | 4 | True | [folds](results/advanced_1_2_lightgbm_classifier_day_of_year/20261104-225232/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 6h | lightgbm_classifier | day_of_year | False |  | -0.005258 | 4 | True | [folds](results/advanced_1_2_lightgbm_classifier_day_of_year/20261104-225232/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 0min | linear_regression | route_target | False |  | -0.028286 | 2 | False | [folds](results/advanced_1_3_linear_regression_route_target/20261104-231348/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 1h | linear_regression | route_target | False |  | -0.033610 | 5 | False | [folds](results/advanced_1_3_linear_regression_route_target/20261104-231348/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 20min | linear_regression | route_target | False |  | -0.030930 | 3 | False | [folds](results/advanced_1_3_linear_regression_route_target/20261104-231348/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 24h | linear_regression | route_frequency | False |  | -0.000000 | 1 | False | [folds](results/advanced_1_3_linear_regression_route_frequency/20261104-230655/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 6h | linear_regression | route_frequency | False |  | -0.000000 | 1 | False | [folds](results/advanced_1_3_linear_regression_route_frequency/20261104-230655/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 0min | logistic_regression | route_frequency | False |  | -0.000310 | 4 | False | [folds](results/advanced_1_3_logistic_regression_route_frequency/20261104-230706/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 1h | logistic_regression | route_frequency | False |  | -0.000261 | 4 | False | [folds](results/advanced_1_3_logistic_regression_route_frequency/20261104-230706/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 20min | logistic_regression | route_frequency | False |  | -0.000372 | 4 | False | [folds](results/advanced_1_3_logistic_regression_route_frequency/20261104-230706/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 24h | lightgbm_classifier | route_frequency | False |  | +0.000000 | 0 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_frequency/20261104-231245/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 6h | lightgbm_classifier | route_frequency | False |  | +0.000000 | 0 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_frequency/20261104-231245/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 0min | lightgbm | delay_missing | False |  | -0.003999 | 3 | False | [folds](results/advanced_1_4_lightgbm_delay_missing/20261104-225429/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 1h | lightgbm | delay_missing | False |  | -0.003775 | 3 | False | [folds](results/advanced_1_4_lightgbm_delay_missing/20261104-225429/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 20min | lightgbm | delay_missing | False |  | -0.002284 | 4 | False | [folds](results/advanced_1_4_lightgbm_delay_missing/20261104-225429/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 0min | lightgbm_classifier | delay_missing | False |  | -0.000370 | 4 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_missing/20261104-225545/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 1h | lightgbm_classifier | delay_clipped | False |  | +0.000000 | 0 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_clipped/20261104-225359/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 20min | lightgbm_classifier | delay_log | False |  | -0.000026 | 2 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_log/20261104-225417/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 0min | lightgbm | weather_extremes | True |  | +0.057086 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261104-225608/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 1h | lightgbm | weather_extremes | True |  | +0.027150 | 1 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261104-225608/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 20min | lightgbm | weather_extremes | True |  | +0.047409 | 1 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261104-225608/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 24h | lightgbm | weather_extremes | True |  | +0.035919 | 2 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_extremes/20261104-225608/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 6h | lightgbm | weather_change | True |  | +0.099358 | 1 | False | [folds](results/advanced_1_5_lightgbm_weather_weather_change/20261104-225624/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 0min | lightgbm_classifier | weather_extremes | True |  | -0.000809 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261104-225741/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 1h | lightgbm_classifier | weather_extremes | True |  | -0.001270 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261104-225741/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 20min | lightgbm_classifier | weather_extremes | True |  | -0.000601 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261104-225741/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 24h | lightgbm_classifier | weather_extremes | True |  | -0.000250 | 2 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_extremes/20261104-225741/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 6h | lightgbm_classifier | weather_change | True |  | +0.002440 | 1 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_weather_change/20261104-225836/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 0min | linear_regression | recent_reliability | False |  | -0.130678 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261104-233656/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 1h | linear_regression | recent_reliability | False |  | -0.124041 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261104-233656/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 20min | linear_regression | recent_reliability | False |  | -0.124756 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261104-233656/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 24h | lightgbm | recent_reliability | False |  | -0.060234 | 3 | False | [folds](results/advanced_2_1_lightgbm_recent_reliability/20261104-233727/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 6h | lightgbm | recent_reliability | False |  | -0.134408 | 4 | True | [folds](results/advanced_2_1_lightgbm_recent_reliability/20261104-233727/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 0min | logistic_regression | recent_reliability | False |  | -0.004938 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261104-233702/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 1h | logistic_regression | recent_reliability | False |  | -0.004150 | 3 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261104-233702/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 20min | logistic_regression | recent_reliability | False |  | -0.004545 | 4 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261104-233702/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 24h | lightgbm_classifier | recent_reliability | False |  | -0.018437 | 5 | True | [folds](results/advanced_2_1_lightgbm_classifier_recent_reliability/20261104-233917/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 6h | lightgbm_classifier | recent_reliability | False |  | -0.019732 | 5 | True | [folds](results/advanced_2_1_lightgbm_classifier_recent_reliability/20261104-233917/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 0min | lightgbm | planned_timetable | False |  | -0.001580 | 3 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261104-225910/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 1h | lightgbm | planned_timetable | False |  | -0.006947 | 4 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261104-225910/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 20min | lightgbm | planned_timetable | False |  | +0.006877 | 1 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261104-225910/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 24h | lightgbm | planned_timetable | False |  | +0.000000 | 0 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261104-225910/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 6h | lightgbm | planned_timetable | False |  | +0.000000 | 0 | False | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261104-225910/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 0min | lightgbm_classifier | planned_timetable | False |  | +0.000203 | 2 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261104-230027/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 1h | logistic_regression | planned_timetable | False |  | +0.000577 | 1 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261104-225859/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 20min | lightgbm_classifier | planned_timetable | False |  | +0.000188 | 1 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261104-230027/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 24h | logistic_regression | planned_timetable | False |  | -0.000028 | 2 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261104-225859/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 6h | logistic_regression | planned_timetable | False |  | -0.000126 | 4 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261104-225859/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 0min | linear_regression | holidays | False |  | -0.008641 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261104-230039/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 1h | linear_regression | holidays | False |  | -0.012231 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261104-230039/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 20min | linear_regression | holidays | False |  | -0.010184 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261104-230039/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 24h | lightgbm | holidays | False |  | -0.121562 | 5 | True | [folds](results/advanced_2_3_lightgbm_holidays/20261104-230101/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 6h | lightgbm | holidays | False |  | -0.121562 | 5 | True | [folds](results/advanced_2_3_lightgbm_holidays/20261104-230101/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 0min | lightgbm_classifier | holidays | False |  | +0.000801 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261104-230254/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 1h | logistic_regression | holidays | False |  | +0.001301 | 2 | False | [folds](results/advanced_2_3_logistic_regression_holidays/20261104-230047/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 20min | lightgbm_classifier | holidays | False |  | +0.001371 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261104-230254/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 24h | lightgbm_classifier | holidays | False |  | -0.022389 | 5 | True | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261104-230254/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 6h | lightgbm_classifier | holidays | False |  | -0.021148 | 5 | True | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261104-230254/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 0min | lightgbm | delay_dynamics | False |  | -0.104514 | 5 | True | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261104-230317/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 1h | lightgbm | delay_dynamics | False |  | -0.066877 | 5 | False | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261104-230317/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 20min | linear_regression | delay_dynamics | False |  | -0.055166 | 5 | False | [folds](results/advanced_2_4_linear_regression_delay_dynamics/20261104-230300/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 0min | lightgbm_classifier | delay_dynamics | False |  | -0.005252 | 5 | True | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261104-230445/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 1h | lightgbm_classifier | delay_dynamics | False |  | -0.003155 | 5 | True | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261104-230445/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 20min | logistic_regression | delay_dynamics | False |  | -0.002017 | 5 | False | [folds](results/advanced_2_4_logistic_regression_delay_dynamics/20261104-230305/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 0min | linear_regression | network_state | False |  | +0.000810 | 2 | False | [folds](results/advanced_2_5_linear_regression_network_state/20261104-231324/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 1h | lightgbm | network_state | False |  | -0.012655 | 3 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261104-231347/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 20min | lightgbm | network_state | False |  | -0.019906 | 5 | False | [folds](results/advanced_2_5_lightgbm_network_state/20261104-231347/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 24h | lightgbm | network_state | False |  | -0.146078 | 4 | True | [folds](results/advanced_2_5_lightgbm_network_state/20261104-231347/paired_vs_base.csv) |
| 2.5 | delay | exploratory | 6h | lightgbm | network_state | False |  | -0.135668 | 4 | True | [folds](results/advanced_2_5_lightgbm_network_state/20261104-231347/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 0min | lightgbm_classifier | network_state | False |  | -0.000763 | 4 | False | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261104-231521/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 1h | lightgbm_classifier | network_state | False |  | -0.000014 | 3 | False | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261104-231521/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 20min | lightgbm_classifier | network_state | False |  | -0.000617 | 3 | False | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261104-231521/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 24h | lightgbm_classifier | network_state | False |  | -0.022407 | 5 | True | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261104-231521/paired_vs_base.csv) |
| 2.5 | severity | exploratory | 6h | lightgbm_classifier | network_state | False |  | -0.020794 | 5 | True | [folds](results/advanced_2_5_lightgbm_classifier_network_state/20261104-231521/paired_vs_base.csv) |
| 2.all | delay | exploratory | 0min | lightgbm | all_groups | False |  | -0.209551 | 5 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261104-234002/paired_vs_base.csv) |
| 2.all | delay | exploratory | 1h | lightgbm | all_groups | False |  | -0.124440 | 4 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261104-234002/paired_vs_base.csv) |
| 2.all | delay | exploratory | 20min | lightgbm | all_groups | False |  | -0.156433 | 5 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261104-234002/paired_vs_base.csv) |
| 2.all | delay | exploratory | 24h | lightgbm | all_groups | False |  | -0.066662 | 3 | False | [folds](results/advanced_2_all_lightgbm_all_groups/20261104-234002/paired_vs_base.csv) |
| 2.all | delay | exploratory | 6h | lightgbm | all_groups | False |  | -0.145888 | 3 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261104-234002/paired_vs_base.csv) |
| 2.all | severity | exploratory | 0min | lightgbm_classifier | all_groups | False |  | -0.004388 | 3 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261104-234219/paired_vs_base.csv) |
| 2.all | severity | exploratory | 1h | logistic_regression | all_groups | False |  | -0.001972 | 4 | False | [folds](results/advanced_2_all_logistic_regression_all_groups/20261104-233938/paired_vs_base.csv) |
| 2.all | severity | exploratory | 20min | lightgbm_classifier | all_groups | False |  | -0.002591 | 4 | False | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261104-234219/paired_vs_base.csv) |
| 2.all | severity | exploratory | 24h | lightgbm_classifier | all_groups | False |  | -0.018851 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261104-234219/paired_vs_base.csv) |
| 2.all | severity | exploratory | 6h | lightgbm_classifier | all_groups | False |  | -0.020496 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261104-234219/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | linear_regression | drop_route | True | full | -0.049654 | 4 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_route/20261104-234238/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | linear_regression | add_calendar | True | timetable-only | -0.630523 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_calendar/20261104-234232/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | linear_regression | drop_route | True | full | -0.052617 | 5 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_route/20261104-234238/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | linear_regression | add_calendar | True | timetable-only | -0.627818 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_calendar/20261104-234232/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | linear_regression | drop_route | True | full | -0.046112 | 3 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_route/20261104-234238/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | linear_regression | add_calendar | True | timetable-only | -0.630232 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_calendar/20261104-234232/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | lightgbm | drop_recent_reliability | True | full | -0.041049 | 3 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_recent_reliability/20261104-234646/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | linear_regression | add_calendar | True | timetable-only | -0.629407 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_calendar/20261104-234232/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | drop_network_state | True | full | -0.017142 | 4 | False | [folds](results/advanced_3_1_linear_regression_weather_drop_network_state/20261104-234244/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | add_calendar | True | timetable-only | -0.629407 | 5 | True | [folds](results/advanced_3_1_linear_regression_weather_add_calendar/20261104-234232/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | drop_wind | True | full | -0.000875 | 5 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_wind/20261104-234517/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | lightgbm_classifier | add_calendar | True | timetable-only | -0.023106 | 5 | True | [folds](results/advanced_3_1_lightgbm_classifier_weather_add_calendar/20261104-235233/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | drop_route | True | full | -0.001342 | 5 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_route/20261104-234359/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | lightgbm_classifier | add_calendar | True | timetable-only | -0.023259 | 5 | True | [folds](results/advanced_3_1_lightgbm_classifier_weather_add_calendar/20261104-235233/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | drop_route | True | full | -0.001080 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_route/20261104-234359/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | lightgbm_classifier | add_calendar | True | timetable-only | -0.023940 | 5 | True | [folds](results/advanced_3_1_lightgbm_classifier_weather_add_calendar/20261104-235233/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | lightgbm_classifier | drop_recent_reliability | True | full | -0.001253 | 4 | False | [folds](results/advanced_3_1_lightgbm_classifier_weather_drop_recent_reliability/20261104-235932/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | lightgbm_classifier | add_calendar | True | timetable-only | -0.024978 | 5 | True | [folds](results/advanced_3_1_lightgbm_classifier_weather_add_calendar/20261104-235233/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | drop_network_state | True | full | -0.000558 | 3 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_network_state/20261104-234426/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | add_calendar | True | timetable-only | -0.024782 | 5 | True | [folds](results/advanced_3_1_logistic_regression_weather_add_calendar/20261104-234332/paired_vs_base.csv) |

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
