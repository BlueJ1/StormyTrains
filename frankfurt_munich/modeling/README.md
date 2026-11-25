# Modelling: shared cross-validation

Every model for Frankfurt (Main) Hbf → München Hbf is trained and scored by the same procedure, so that models differ only in their features and estimator. A model is a function that returns a fresh, unfitted scikit-learn estimator; `cv.py` does the rest.

Only the CV set is used (`../data/prediction_rows_cv.parquet`). The weather models merge on the route weather from `../data/prediction_rows_weather_cv.parquet`; the railway-only models never read that file. Loading the hold-out set needs `allow_holdout=True`.

The cutoff-safe experiments of PR #12 (new input features, nested tuning, other model families, ensembles, neural networks, transfer learning and the hold-out evaluation) are in [`../advanced_modeling/`](../advanced_modeling/README.md). They reuse this folder's cross-validation and model code. For them, `features.py` names feature sets: its default `base` set leaves out `recorded_route`, `observed_stops` and `unobserved_sequence_positions`, which come from the recorded stops and can depend on the outcome. `run_linear_models.py` and `run_tuned_models.py` pass `feature_set="recorded_route_diagnostic"`, the original columns. Their results (15–31 October 2026) were deleted on 16 November 2026 (finishing plan B12) and stay in the git history. The eight flat searches of LightGBM and XGBoost that the advanced models reuse for their fixed configurations were moved to [`../advanced_modeling/flat_searches/`](../advanced_modeling/flat_searches/).

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
| `models.py` | Model factories: logistic regression, linear regression, the two baselines (class shares and the training mean), random forest, XGBoost, LightGBM and SVR regressors, and random forest, XGBoost and LightGBM classifiers |
| `metrics.py` | Class order, log loss, macro-F1, RMSE |
| `cv.py` | `cross_validate` (one prediction time) and `run_cv` (all prediction times, writes the results) |
| `run_linear_models.py` | Runs the six linear and baseline models |
| `tuning.py` | Random search of hyperparameters on the 5 folds (`run_search`) and the search spaces |
| `run_tuned_models.py` | Runs the fourteen tuned models; `run_tuned_models.py lightgbm svr` runs only those |
| `tests/test_pipeline.py` | Checks of the method itself |
| `tests/test_tuned_models.py` | Checks of the tuned models' preprocessing and of the search |

To add a model: write a factory `make_x(numeric, categorical)` in `models.py` and call `run_cv("x", "severity" or "delay", make_x)`; the keyword argument `weather=True` adds the weather features. `run_cv` uses the `base` feature set by default; the first models used `feature_set="recorded_route_diagnostic"`, so a model added with the default is not comparable with them.

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
- **Run time:** every model is single-threaded and the search is serial: about 17 CPU hours for the eight delay runs (SVR and the random forest take most of it) and 13 for the six severity runs. The searches evaluated configurations in parallel; because each model is single-threaded with a fixed seed, a serial run gives the same scores, only the `seconds` column differs.
