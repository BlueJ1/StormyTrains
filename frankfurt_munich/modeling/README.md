# Modelling: shared cross-validation

Every model for Frankfurt (Main) Hbf → München Hbf is trained and scored by the same procedure, so that models differ only in their features and estimator. A model is a function that returns a fresh, unfitted scikit-learn estimator; `cv.py` does the rest.

Only the CV set is used (`../data/prediction_rows_cv.parquet`). Loading the hold-out set needs `allow_holdout=True`.

## Run

```bash
.venv/bin/python -m pip install -r frankfurt_munich/modeling/requirements.txt
.venv/bin/python -m pytest frankfurt_munich/modeling/tests          # about 5 s
.venv/bin/python frankfurt_munich/modeling/run_linear_models.py     # about 10 s
```

`.venv` is at the repository root (Python 3.12).

| File | Purpose |
|---|---|
| `data.py` | `load_rows(prediction_time)`: the prediction rows for one prediction time, joined with journey columns, plus derived columns |
| `features.py` | Feature columns per prediction time |
| `models.py` | Model factories: logistic regression, linear regression, and the two references |
| `metrics.py` | Class order, log loss, macro-F1, RMSE |
| `cv.py` | `cross_validate` (one prediction time) and `run_cv` (all prediction times, writes the results) |
| `run_linear_models.py` | Runs the four models below |
| `tests/test_pipeline.py` | Checks of the method itself |

To add a model: write a factory `make_x(numeric, categorical)` in `models.py` and call `run_cv("x", "severity" or "delay", make_x)`.

## Method

- **One model per prediction time** (24 h, 6 h, 1 h, 20 min, 0 min), fitted on the rows of that prediction time only.
- **Folds:** the 5 folds of `../folds.py`, stored in the `fold` column: random groups of whole days, stratified by `severity_class` (see `../README.md`). Fold *k* validates on the rows with `fold == k` and trains on all other CV rows, including days after the validation days. There is no gap between training and validation: keeping each day on one side already separates journeys that share weather and disruptions.
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
  - `config.json`: task, model, features and git commit;
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

- **Left out because they describe the outcome:**
  - reported times;
  - the cancellation columns, `cancellation_announced` and `cancellation_withdrawn`;
  - `shared_departure` and `is_replacement_train`;
  - the targets.
- **Caveat:** `recorded_route`, `observed_stops` and `unobserved_sequence_positions` come from the recorded stops, not the planned ones.
- **Linear models:**
  - missing `current_delay` becomes 0 plus a 0/1 missing indicator;
  - numeric columns are standardised;
  - categorical columns are one-hot encoded, and categories unseen in training get all-zero columns.

## Models

| Name | Task | Model |
|---|---|---|
| `class_shares` | severity | Reference: the training class shares for every journey |
| `logistic_regression` | severity | Multinomial logistic regression, L2 penalty with C = 1 (scikit-learn's default, not tuned) |
| `training_mean` | delay | Reference: the training mean delay for every journey |
| `linear_regression` | delay | Ordinary least squares |

## Tests

`tests/test_pipeline.py` checks the method, not the models:

- the rows' folds equal those in `../data/journeys_cv.parquet`;
- at every prediction time, the 5 folds split the rows into training and validation with no day on both sides, and every row is validated exactly once;
- `current_delay` is used only from 1 h on;
- loading the hold-out set raises an error;
- log loss agrees with scikit-learn's, and the class-share reference scores exactly the cross-entropy computed by hand;
- the probability columns are in the fixed class order;
- with the answer given as a feature, the logistic model reaches log loss < 0.05 and macro-F1 = 1, and the linear model reaches RMSE ≈ 0, so predictions line up with their rows;
- with shuffled labels, the full logistic model does not beat the class shares, so there is no hidden leak.

## Results, 12 October 2026

Mean ± standard deviation over the 5 folds.

| Prediction time | Log loss: class shares | Log loss: logistic | Macro-F1: class shares | Macro-F1: logistic | RMSE: training mean | RMSE: linear |
|---|---|---|---|---|---|---|
| 24 h | 0.956 ± 0.065 | 0.926 ± 0.057 | 0.195 | 0.208 | 25.50 ± 1.46 | 25.03 ± 1.16 |
| 6 h | 0.951 ± 0.066 | 0.921 ± 0.058 | 0.195 | 0.209 | 25.50 ± 1.46 | 25.02 ± 1.15 |
| 1 h | 0.915 ± 0.065 | 0.846 ± 0.039 | 0.197 | 0.290 | 25.50 ± 1.46 | 24.13 ± 0.82 |
| 20 min | 0.896 ± 0.064 | 0.810 ± 0.044 | 0.198 | 0.307 | 25.50 ± 1.46 | 23.06 ± 0.85 |
| 0 min | 0.888 ± 0.060 | 0.796 ± 0.042 | 0.198 | 0.325 | 25.50 ± 1.46 | 22.84 ± 0.90 |

- Both models beat their reference at every prediction time.
- The largest gain comes from 1 h on, when `current_delay` is available.
- The class-share reference always predicts `normal_minor`, so its macro-F1 stays near 0.2.
