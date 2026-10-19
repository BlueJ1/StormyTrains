# Modelling: shared cross-validation

Every model for Frankfurt (Main) Hbf → München Hbf is trained and scored by the same procedure, so that models differ only in their features and estimator. A model is a function that returns a fresh, unfitted scikit-learn estimator; `cv.py` does the rest.

Only the CV set is used (`../data/prediction_rows_cv.parquet`). The weather models merge on the route weather from `../data/prediction_rows_weather_cv.parquet`; the railway-only models never read that file. Loading the hold-out set needs `allow_holdout=True`.

## Run

```bash
.venv/bin/python -m pip install -r frankfurt_munich/modeling/requirements.txt
.venv/bin/python -m pytest frankfurt_munich/modeling/tests          # about 5 s
.venv/bin/python frankfurt_munich/modeling/run_linear_models.py     # about 30 s
```

`.venv` is at the repository root (Python 3.13).

| File | Purpose |
|---|---|
| `data.py` | `load_rows(prediction_time, weather=False)`: the prediction rows for one prediction time, joined with journey columns, plus derived columns and, with `weather=True`, the weather columns |
| `features.py` | Feature columns per prediction time |
| `models.py` | Model factories: logistic regression, linear regression, and the two references |
| `metrics.py` | Class order, log loss, macro-F1, RMSE |
| `cv.py` | `cross_validate` (one prediction time) and `run_cv` (all prediction times, writes the results) |
| `run_linear_models.py` | Runs the six models below |
| `tests/test_pipeline.py` | Checks of the method itself |

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
  - `oof_predictions.parquet`: the validation predictions. These files are not committed.

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

## Models

| Name | Task | Model |
|---|---|---|
| `class_shares` | severity | Reference: the training class shares for every journey |
| `logistic_regression` | severity | Multinomial logistic regression, L2 penalty with C = 1 (scikit-learn's default, not tuned) |
| `logistic_regression_weather` | severity | The same, with the weather features added |
| `training_mean` | delay | Reference: the training mean delay for every journey |
| `linear_regression` | delay | Ordinary least squares |
| `linear_regression_weather` | delay | The same, with the weather features added |

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
