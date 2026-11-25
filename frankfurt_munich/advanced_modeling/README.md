# Advanced modelling: Frankfurt (Main) Hbf → München Hbf

The experiments of the [advanced modelling plan](PLAN.md), added in PR #12: inputs known at the cutoff, new feature groups, nested tuning, other model families, ensembles, neural networks, transfer learning, and the one-time hold-out evaluation of railway-only against weather models. The plan is also published at https://claude.ai/artifact/KAso5U9ubq5rFxMy7juUna (same content as `PLAN.md` and `plan.html`).

**Outcome.** Most of this work is exploratory screening, scored on the same five CV week folds as [`../modeling/`](../modeling/README.md). New input features gave the clearest gains: recent reliability (step 2.1) reached the minimum gain we agreed on in advance (D12: 0.1 min RMSE, 0.003 log loss) in 17 of 20 comparisons, and all new groups together in 19 of 20. Changes to the learner rarely helped. Seed and configuration bagging reached it in 8 of 120 comparisons and pooling the horizons in 6 of 20. The two-stage models, CatBoost, the explainable boosting machine, the distributional models, calibration, stacking, weighted ensembles and corridor or network transfer never did, and the plain MLP never did against LightGBM. In the second hold-out evaluation (24 November 2026), the finalists are LightGBM with nested tuning and automatically chosen feature groups for all ten tasks and horizons, each with four feature sets (railway only, observed weather, forecasts, both). On the 23 hold-out weeks, neither observed weather nor forecasts improved either task. 33 of the 50 comparisons are exactly zero because both models are the same, and the only interval that excludes zero shows observed weather raising delay RMSE at 24 h by 0.143 min (95% interval 0.039 to 0.246). The first hold-out evaluation (6 November 2026), with LightGBM and linear regression as finalists, found no gain from observed weather either: all ten 95% intervals included zero.

## Layout

| Files | Purpose |
|---|---|
| `scheduled_timetable.py`, `history_features.py`, `recent_reliability.py`, `holiday_features.py` | Inputs known at the cutoff, built from the plan and change snapshots: planned timetable, live delay and its dynamics, recent reliability, network state, holidays |
| `build_advanced_weather.py`, `build_corridor_inputs.py`, `network_transfer.py`, `sequence_inputs.py` | Weather on the planned route, other corridors (8.3), network-wide delay pairs (8.4), upstream sequences (7.5) |
| `build_forecast_weather.py` | GFS forecast weather for the hours of the planned journey (step 10.8, decision D17), alongside the observed weather |
| `holdout_gate.py` | Refuses to build hold-out inputs unless the code still matches the checksums stored in the committed finalist list (`finalists.json`) |
| `run_advanced.py`, `experiment_specs.py`, `experiment_models.py`, `advanced_data.py` | The CV steps of the plan: which runs exist (model, inputs and feature set), their models and inputs |
| `representations.py`, `group_selection.py`, `decomposed.py`, `ensembles.py`, `extra_models.py`, `distributions.py`, `neural.py`, `gated_neural.py`, `sequence_model.py`, `corridor_transfer.py` | Phase 1 and 3 to 8 methods |
| `compare.py`, `report_cv.py`, `audit_cv_results.py` | Paired fold comparisons, the CV tables below, and the output audit of the CV runs |
| `run_final.py`, `final_analysis.py`, `result_plots.py`, `finalists.json` | Registration of finalists and the hold-out evaluation |
| `flat_searches/` | The eight flat searches of the first models (LightGBM and XGBoost, with and without weather, both tasks), moved from `../modeling/results/` on 16 November 2026 |
| `results/` | One folder per CV experiment, `results/advanced_holdout/` (second hold-out evaluation) and `results/advanced_holdout_first/` (first hold-out evaluation) |
| `tests/` | `.venv/bin/python -m pytest frankfurt_munich/advanced_modeling/tests` |
| `_paths.py` | Puts `../modeling` and `..` on the import path |

The cross-validation, tuning, feature sets and model factories are shared with the earlier models in `../modeling/` (`cv.py`, `tuning.py`, `features.py`, `models.py`, `data.py`). Generated inputs are written to `../data/advanced/`, which is not committed. The distributional learners need the separate environment in [requirements-distributions.txt](requirements-distributions.txt).

## Inputs and runs

The earlier scores in [`../modeling/`](../modeling/README.md) use recorded-route inputs. `run_advanced.py` runs the CV experiments in [PLAN.md](PLAN.md) with the new cutoff-specific inputs. It keeps the same prediction rows and week folds. The recorded route fields are available only in the step 0.7 leakage comparison and the historical reproduction commands.

The advanced runner reads generated CV inputs in `../data/advanced/`. It requires the timestamped timetable and reconstructed live-delay data and checks that they cover the same prediction keys. It reads advanced weather only for runs with weather.

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

The GFS forecast Frankfurt temperature is known for every CV row and the München temperature for all but 10 (two journeys on 30 September 2026 arrive after the end of the extract); the route features need the weather branch, like the observed weather. Each row uses the latest GFS run that started by its cutoff (a run counts as available from its start, decision D17) and the hours from the planned departure to the planned arrival. The forecast runs of step 10.9 use them in CV (results below), and the second hold-out evaluation (step 10.6, finishing plan B6) used them on the hold-out weeks.

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

`run_advanced.py --help` lists the CV steps and filters. Each weather run has two forecast versions (decision D17, step 10.9): `_forecast` (GFS forecasts in place of the observations) and `_weather_forecast` (observations plus forecasts), except XGBoost with nested tuning, CatBoost and the runs built on them. `--weather-arm` picks the feature sets to run (`railway`, `weather`, `forecast`, `weather_forecast`), and `audit_cv_results.py --arm` audits only those. Both forecast versions reuse the observed-weather flat configuration. Fixed configurations come from the eight flat searches of the first models in `flat_searches/`: `config.json` gives the configuration chosen per prediction time, and `search_results.csv` supplies the top-five averages of the configuration bagging (step 5.2). The runs record these sources relative to `flat_searches/`. Following the user decision on 5 November, remaining booster and CatBoost searches use 20 configurations inside each outer training set. Six completed 50-configuration baseline searches also supply budget-matched 20-configuration controls by reusing their fixed first 20 inner-score rows and refitting the outer models. The saved configurations link to the original searches. The starting-score tables below and each run configuration record the actual count. Final registration uses at most 20 configurations. Neural searches use eight. The `auto_groups` and `auto_representations` options to `--nested-variant` also select features inside each outer training set. For runs with weather, `auto_groups` (automatically chosen feature groups) has a second step, the weather selection step (decided 23 November 2026, review of PR #18): starting from the railway groups that the first step chose, it screens all of the run's weather groups, none, and each group dropped once (observed and forecast groups separately), over the same three configurations; ties keep all weather. The fold's choice is saved as `weather_variant`. The eight weather runs of step 0.8 with `auto_groups` were rerun with it on 23–24 November 2026 from commit `432a86e`. Over their 200 outer folds it kept all weather in 19, dropped all weather in 71 and dropped one group in 110 (most often precipitation: observed 26, forecast 24). A variant fixed after seeing the exploratory results retains that label even when its parameters are tuned again.

When running a subset of horizons with `--prediction-time`, set a separate `--results-dir`. Reports and registration select the latest timestamp in each run directory, so a partial run in the main results directory can replace the complete run in those reports.

The model-family screens use the cutoff-safe base columns for their learner comparisons. The boosters with nested tuning and automatically chosen feature groups (step 0.8) evaluate the expanded Phase 2 inputs separately.
The corrected policy ranks each feature variant by its best inner-fold score across the first three configurations in the same search. It then tunes the selected variant over the full 20 configurations. The extra screening fits reuse configurations from that budget.

All 354 active CV runs were repeated on the rebuilt inputs at commit `7e0e96a` (finishing plan B4, 13–15 November 2026): the 366 runs of the rebuild decision (D14) less the four network-state runs (2.5) and the eight network-state ablations in 3.1, dropped on 13 November. The four observed-weather `auto_groups` runs of step 0.8 come from the rerun at `432a86e` (above), so the audit takes both commits; all 354 passed (`audit_cv_results.py --commit 7e0e96a2569f4722c545856aeae96c0170d7ee80 432a86e53e5a9bda366207f7c1a8c562c56751bb`). It checks that each run has its files, was made from a clean commit, has the expected horizons with five folds each, and has summaries equal to its fold means. It also checks that the saved predictions cover every CV row once in its committed fold, are valid, and reproduce the fold scores. The automatic feature-group selection reached the minimum gain we agreed on (D12: 0.1 min RMSE, 0.003 log loss) in 19 of 20 LightGBM comparisons and 13 of 20 XGBoost comparisons. Before the weather selection step it was 20 of 20; LightGBM severity with weather at 24 h now improves by 0.0023, below the minimum. None of the 12 network-transfer comparisons reached it.

The 364 forecast runs (decision D17, step 10.9, finishing plan B17) were run on 19–20 November 2026 from commit `7300d61`; their four `auto_groups` runs of step 0.8 come from the rerun at `432a86e`. All 364 passed the output audit (`audit_cv_results.py --commit 7300d61bf69e7c504b15a327c48092b6a8532481 432a86e53e5a9bda366207f7c1a8c562c56751bb --arm forecast weather_forecast --accept-b17-dirty-flag`). 356 of them recorded "(uncommitted changes)" although nothing outside `results/` had changed: `cv.git_commit` read its `results/` exclusion relative to `modeling/`, so new advanced results counted as changes. The exclusion is now anchored at the repository top, and the audit option accepts the flag only for forecast runs from `7300d61` (group decision, 20 November 2026). `forecast_comparisons.py` pairs each forecast run with the run of the same step, model and variant in the other feature sets. It writes the differences fold by fold (same folds for both runs), and whether their mean reaches the minimum gain, to `results/forecast_arm_comparisons.csv`. They are exploratory unless both runs use nested tuning. The summary leaves out the recorded-route leakage diagnostic (0.7) and pairs whose two runs have the same inputs at every horizon (group-ablation variants of 3.1 that contain neither run's weather) and keeps ties. For the linear models and boosters:

| Comparison (first against second) | Comparisons | First ahead by the minimum gain | Second ahead by the minimum gain | First has the lower mean |
| --- | --- | --- | --- | --- |
| Forecasts against railway only | 234 | 3 | 29 | 119 |
| Forecasts against observations | 426 | 43 | 8 | 343 |
| Observations plus forecasts against observations | 506 | 5 | 4 | 186 |
| Observations plus forecasts against forecasts | 506 | 0 | 76 | 72 |

The `configuration` column of the CSV matters for the first row. Fixed-configuration boosters take the flat search of their own feature set (D2), so in 149 of the 234 comparisons the forecast run used the observed-weather configuration and the railway run the railway one; railway only reached the minimum gain in 23 of these and forecasts in none. These pairs compare two pipelines, not inputs alone. Of the other 85, 65 share a configuration (forecasts ahead by the minimum gain in 2, railway only in 6) and 20 are tuned separately for each feature set (1 and 0).

Forecasts did better than observations but not better than railway inputs alone, and neither source added anything to the other. The EBM and the neural networks did worse with either source than with railway inputs alone (railway only ahead by the minimum gain in 36 of 66 comparisons with forecasts), less so with forecasts than with observations (forecasts ahead by the minimum gain in 45 of 66). LightGBMLSS (4.2, 6.5) is unstable in every run, with fold RMSE up to 24,340 min, and its pairs are not interpreted. Forecast extremes (step 1.5) reached the minimum gain against the same run's raw inputs in 5 of 10 comparisons with forecasts alone and 7 of 10 with both sources; the 24-hour forecast change reached it in none and was worse by the minimum gain in 2 and 4. The weather subsets (step 3.4) reached the minimum gain against all weather of their run in 1 of 40 (forecasts) and 8 of 40 (both). In the group ablation (step 3.1), adding forecast temperature to timetable-only inputs reached the minimum gain in all 20 comparisons of each feature set, the other forecast groups in none, and dropping a forecast group from the full inputs in none of 80 per feature set.

The review of 10 November asked for the neural networks and the pooled model to be compared with LightGBM with nested tuning (step 0.8), not only with the MLP or the fixed-configuration booster. These comparisons use the same fold-by-fold differences (`compare.py`), against the step 0.8 LightGBM run of the same task and feature set on the same horizons:

| Step | Comparisons | Lower mean error than LightGBM with nested tuning | Reached minimum gain |
| --- | ---: | ---: | ---: |
| 7.1 MLP | 20 | 0 | 0 |
| 7.2 multi-task MLP | 20 | 0 | 0 |
| 7.3 TabGNS | 20 | 0 | 0 |
| 7.4 RealMLP and TabM | 40 | 1 | 0 |
| 7.5 GRU (1 h on) | 12 | 0 | 0 |
| 8.2 pooled prediction times (exploratory) | 20 | 19 | 4 |

The one neural network with a lower mean error is TabM with weather for delay at 24 h (0.010 min, three folds improved). The pooled model reached the minimum gain for severity at 0 min, for severity with weather at 1 h and 20 min, and for delay with weather at 6 h. The pooled models reuse flat-search configurations, so these comparisons are exploratory.

The distributional models (steps 6.5 and 4.2) use a separate Python 3.13 environment with [requirements-distributions.txt](requirements-distributions.txt). NGBoost and LightGBMLSS constrain dependencies to older releases than the main environment uses. In the CV rerun on the corrected inputs (finishing plan B4) they ran in that environment with `OMP_NUM_THREADS=1`, for example `OMP_NUM_THREADS=1 <distribution-python> frankfurt_munich/advanced_modeling/run_advanced.py 6.5 --jobs 3`. Set it as well when `run_final.py evaluate` is given `--distribution-python`. Without that setting LightGBMLSS crashes on macOS, because scikit-learn, LightGBM and torch each load their own OpenMP runtime. The models are single-threaded, so the setting does not change their results. The run of 4.2 and 6.5 before the rebuild had recorded the main environment's numpy, pandas and scikit-learn versions, not these pins.

The LightGBMLSS adapter standardizes delay within each training fold and returns the Gaussian mixture in minutes. Fitting uses the package's Gumbel-softmax relaxation. Prediction uses softmax weights from the fitted logits, giving a mean and CDF independent of prediction batch order. [LightGBMLSS API reference](https://statmixedml.github.io/LightGBMLSS/api/)

The transfer from other corridors (step 8.3) uses railway inputs from the three auxiliary corridors. The builder drops ambiguous shared departures, replacements and additional endpoint stops. Each fit admits only source runs whose origin, run start, planned destination and reported outcome weeks are eligible. The inner and outer fits also exclude validation journey IDs. The network-wide delay propagation (step 8.4) applies the same run exclusion and checks the planned, reported and cutoff weeks of its network pairs.

```bash
.venv/bin/python frankfurt_munich/advanced_modeling/build_corridor_inputs.py
.venv/bin/python frankfurt_munich/advanced_modeling/network_transfer.py
.venv/bin/python frankfurt_munich/advanced_modeling/sequence_inputs.py
```

### Holdout evaluation

`results/advanced_holdout_first/` keeps the finalist list and results of the first hold-out evaluation, run at commit `065b077`; they moved there on 24 November 2026 to make room for the second evaluation. The source has changed since registration (the review fixes, then the move into this folder on 8 November 2026), so the hold-out builders, which check the code against the checksums in the finalist list, refuse to run with this list. The PR review changed the development pipeline after that evaluation. The scores below belong to the original inputs and source, and the review reruns use CV only. The original hold-out inputs predate the review fixes. Two of those fixes could change the scores: the recent-reliability features (2.1) of the hold-out rows drew past journeys from both splits, while the CV rows used CV journeys only; and before a replacement was announced, `current_delay` and the delay dynamics (2.4) of a train that was later replaced were set to missing. For the original protocol and the checksums of its code, use commit `065b077` and its recorded inputs. The original feature inputs are archived locally in `../data/advanced_original_065b077/`, and the corrected CV inputs are in `../data/advanced/`. The evaluator refuses to overwrite the completed results.

The registration can re-tune the feature sets in parallel after the shared feature selection. It records the tuning environment and stores checksums of the code and of the selected CV inputs. The hold-out builders check the code against these checksums. The evaluation checks the registered CV inputs before scoring and records each model’s interpreter and package versions. On a retry, the scorer reuses completed models when the committed finalist list and the checksums of both splits match.

After a forced stop, resume with the same finalist list once the scorer and its worker processes have exited and the stale `.evaluation.lock` has been cleared.

After the CV steps are complete, `run_final.py register` ranks the pipelines that have CV results for all four feature sets (railway only, observed weather, forecasts, observations plus forecasts) by their railway-only CV score. It writes the top-ranked pipeline for each task and horizon, with all four feature sets, to `finalists.json` (finishing plan B6, group decisions of 24 November 2026). The railway feature groups are chosen on the railway-only version and reused in every feature set. In the pipelines that choose feature groups automatically (step 0.8), each weather version then chooses on its own CV weeks whether to keep all, none or all but one of its weather groups; the weather versions of nested group selection (step 3.3) keep all their weather groups. With the CV results of 24 November, LightGBM with nested tuning and automatically chosen feature groups is chosen for all ten tasks and horizons. `register` will not overwrite an existing `finalists.json`; the first evaluation's list is in `results/advanced_holdout_first/finalists.json`. Commit the new file, together with the plan entry that records the registration, before generating hold-out inputs.

```bash
.venv/bin/python frankfurt_munich/advanced_modeling/run_final.py register --jobs 4
# After committing finalists.json and the plan entry that records it
.venv/bin/python frankfurt_munich/advanced_modeling/scheduled_timetable.py --split holdout
.venv/bin/python frankfurt_munich/advanced_modeling/history_features.py --split holdout
.venv/bin/python frankfurt_munich/advanced_modeling/build_advanced_weather.py --split holdout
.venv/bin/python frankfurt_munich/advanced_modeling/build_forecast_weather.py --split holdout
```

Build any additional inputs named by the finalists with `--split holdout`. Use `history_features.py --feature reliability` for recent reliability and `sequence_inputs.py` for upstream sequences. Then run `run_final.py evaluate` once. If a distributional learner is registered, supply its pinned interpreter via `--distribution-python`. The evaluation saves scores, bootstrap intervals over whole ISO weeks (the same weeks for both models) and subgroup errors for five comparisons (observed weather against railway only in `{task}_weather.csv` and `{task}_errors.csv`; forecasts against railway only and against observations, and observations plus forecasts against observations and against forecasts in `{task}_<candidate>_vs_<reference>.csv` and `..._errors.csv`), calibration, risk trajectories, and SHAP summaries in `results/advanced_holdout/`.

#### Second hold-out evaluation (24 November 2026)

The finalist list (`finalists.json`; finishing plan B6, modelling plan 10.5) fixed 40 models before any hold-out input was built: for both tasks at all five horizons, LightGBM with nested tuning and automatically chosen feature groups, each with four feature sets (railway only, observed weather, forecasts, both). It uses all railway feature groups, except for severity at 6 h, where recent reliability is the only added group. A second step chose, on all 95 CV weeks, whether each weather version keeps all, none or all but one of its weather groups. It kept weather in only 7 of the 30 weather versions. The other 23 have the same columns and settings as the railway-only model, so they are the same model and score identically. The group chose to score the finalists as registered. Each model was refitted on the 95 CV weeks and scored once on the 23 hold-out weeks. The 95% intervals come from a bootstrap with 2,000 resamples that draws whole ISO weeks, the same weeks for both models of a comparison; each interval is computed on its own (pointwise). [Finalist list](finalists.json), [scores](results/advanced_holdout/scores.csv)

Delay RMSE (min). The training mean is the baseline. "= railway" marks a model that is identical to the railway-only model.

| Horizon | Rows | Railway | Observed weather | Forecasts | Both | Training mean |
|---|---|---|---|---|---|---|
| 24 h | 4,089 | 25.106 | 25.249 | = railway | = railway | 26.409 |
| 6 h | 4,086 | 24.995 | = railway | 25.025 | = railway | 26.398 |
| 1 h | 4,084 | 24.562 | = railway | 24.557 | = railway | 26.390 |
| 20 min | 4,084 | 24.553 | = railway | = railway | = railway | 26.390 |
| 0 min | 4,084 | 24.468 | = railway | = railway | = railway | 26.390 |

Severity log loss, with macro-F1 in parentheses. Class shares are the baseline.

| Horizon | Rows | Railway | Observed weather | Forecasts | Both | Class shares |
|---|---|---|---|---|---|---|
| 24 h | 4,313 | 0.9137 (0.245) | 0.9170 (0.209) | 0.9120 (0.241) | 0.9156 (0.208) | 0.9708 (0.194) |
| 6 h | 4,294 | 0.9049 (0.222) | = railway | 0.9041 (0.215) | = railway | 0.9633 (0.195) |
| 1 h | 4,237 | 0.8616 (0.259) | = railway | = railway | = railway | 0.9347 (0.196) |
| 20 min | 4,204 | 0.8412 (0.271) | = railway | = railway | = railway | 0.9150 (0.197) |
| 0 min | 4,188 | 0.8221 (0.276) | = railway | = railway | = railway | 0.9046 (0.198) |

There are 50 comparisons: five pairs of feature sets for each of the ten task–horizon combinations. 33 of them are exactly zero, because both models are the same. Only one comparison has an interval that excludes zero: at 24 h, observed weather raised delay RMSE by 0.143 min compared with railway only (95% interval 0.039 to 0.246). Forecasts against observations and both against observations show the same difference with the opposite sign, because at 24 h the delay models with forecasts and with both equal the railway-only model. The other nonzero differences are smaller than the minimum gain we agreed on (D12: 0.1 min RMSE, 0.003 log loss) or have intervals that include zero. The largest is forecasts against observations for severity at 24 h: 0.0050 lower log loss (−0.0110 to 0.0004). Every finalist beats its baseline at every horizon. [Comparisons](results/advanced_holdout/), [subgroup errors](results/advanced_holdout/severity_errors.csv)

`run_final.py` did not include the two baselines, so they were scored after the main run, on the same training and hold-out rows, with [score_references.py](results/advanced_holdout/score_references.py) ([baseline scores](results/advanced_holdout/references.csv)). The group decided to report them on the hold-out set (finishing plan Q7); nothing is chosen with them. The hold-out inputs were rebuilt with the corrected code on 24 November 2026 in `../data/advanced/`; the first evaluation's inputs stay archived in `../data/advanced_original_065b077/`.

#### First hold-out evaluation (6 November 2026)

The finalist list fixed 20 models (ten pairs of railway-only and weather models) before the hold-out labels were read. Each model was refitted on the 95 CV weeks and scored on the same rows as the other model of its pair across 23 held-out ISO weeks. LightGBM was selected for all severity horizons and for delay at 24 h and departure. Full-group linear regression was selected for delay at 6 h, 1 h and 20 min. The LightGBM pairs share railway features, and each model was tuned separately over the same 20 configurations. [Finalist list](results/advanced_holdout_first/finalists.json)

Weather minus railway is reported below, with negative deltas favoring weather. The 95% intervals come from a bootstrap with 2,000 resamples of whole ISO weeks, the same weeks for both models; each interval is computed on its own (pointwise). All ten intervals include zero. The weather model had higher delay RMSE at all five horizons, by 0.017 to 0.193 min. Severity log loss changed most at 24 h, increasing by 0.00486. These results cover the held-out weeks on this corridor under the stratified week split. Route weather aggregates were present for 57–65% of severity rows across horizons. This retrospective comparison uses DWD observation hours ending by the cutoff and assumes publication at the labeled hour. Actual publication latency was unmeasured.

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

At departure, weather lowered log loss for normal or minor delays by 0.00866 and raised it for severe delays by 0.02732 and cancellations by 0.01909. It raised RMSE for severe delays by 0.490 min. Departure severity log loss moved by less than 0.0006 in either period and in opposite directions on the Nürnberg and Stuttgart branches. The heavy-rain groups contain two to four rows per horizon, and 1,486 departure severity rows lack a branch assignment. These subgroup differences are descriptive. [Severity subgroup scores](results/advanced_holdout_first/severity_errors.csv), [delay subgroup scores](results/advanced_holdout_first/delay_errors.csv)

For severe delays and cancellations, the calibration bins at or above 0.5 have at most five observations per nonempty bin. Departure hour led the SHAP rankings for all ten severity models. Planned travel time and service reliability also ranked highly. The runner uses TreeSHAP for LightGBM and sampled permutation SHAP for linear regression. [Reliability plots](results/advanced_holdout_first/calibration.svg), [SHAP summaries](results/advanced_holdout_first/shap.csv)

The risk trajectories show the earliest journey with all five horizons in each realized class. The railway-only model's severe-or-cancelled probability for the severe example rose from 10.4% at 24 h to 18.5% at 6 h and was 10.1% at departure. The cancelled example stayed between 5.9% and 8.3% with the railway-only model. [Risk trajectories](results/advanced_holdout_first/risk.svg)

<!-- BEGIN ADVANCED CV -->

### Advanced CV results

Scores use the five committed CV week folds. A difference is a run's score minus the score of the run it is compared with, on the same folds, so positive differences favor the run it is compared with. The column `arm` is the feature set (railway only, plus observed weather, plus forecasts, plus both); `contrast` says whether a group-ablation variant of step 3.1 is compared with timetable-only or with full inputs.

#### Step 0.7 starting scores (inputs known at the cutoff)

| Model | Search configs | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [class_shares](results/advanced_0_7_class_shares/20261113-212614) | fixed | log_loss | 0.967992 | 0.958166 | 0.922901 | 0.904186 | 0.896396 |
| [lightgbm](results/advanced_0_7_lightgbm/20261113-212635) | fixed | rmse | 24.495471 | 24.505243 | 24.080572 | 24.020343 | 23.721266 |
| [lightgbm_classifier](results/advanced_0_7_lightgbm_classifier/20261113-212808) | fixed | log_loss | 0.920320 | 0.911833 | 0.865819 | 0.845798 | 0.828542 |
| [lightgbm_classifier + forecast](results/advanced_0_7_lightgbm_classifier_forecast/20261119-162945) | fixed | log_loss | 0.928039 | 0.912640 | 0.869074 | 0.850085 | 0.833054 |
| [lightgbm_classifier + weather](results/advanced_0_7_lightgbm_classifier_weather/20261113-212856) | fixed | log_loss | 0.922384 | 0.916049 | 0.869455 | 0.850772 | 0.834340 |
| [lightgbm_classifier + weather + forecast](results/advanced_0_7_lightgbm_classifier_weather_forecast/20261119-163049) | fixed | log_loss | 0.925491 | 0.914796 | 0.871908 | 0.852550 | 0.836367 |
| [lightgbm + forecast](results/advanced_0_7_lightgbm_forecast/20261119-162654) | fixed | rmse | 24.575030 | 24.444375 | 24.046055 | 24.014528 | 23.733145 |
| [lightgbm + weather](results/advanced_0_7_lightgbm_weather/20261113-212638) | fixed | rmse | 24.513819 | 24.501714 | 24.163462 | 24.090464 | 23.787463 |
| [lightgbm + weather + forecast](results/advanced_0_7_lightgbm_weather_forecast/20261119-162701) | fixed | rmse | 24.534664 | 24.472006 | 24.166255 | 24.101728 | 23.822051 |
| [linear_regression](results/advanced_0_7_linear_regression/20261113-212614) | fixed | rmse | 24.628921 | 24.634678 | 24.191326 | 24.148835 | 23.906206 |
| [linear_regression + forecast](results/advanced_0_7_linear_regression_forecast/20261119-162633) | fixed | rmse | 24.568709 | 24.566650 | 24.156443 | 24.111695 | 23.874428 |
| [linear_regression + weather](results/advanced_0_7_linear_regression_weather/20261113-212615) | fixed | rmse | 24.605495 | 24.644969 | 24.208634 | 24.161579 | 23.918108 |
| [linear_regression + weather + forecast](results/advanced_0_7_linear_regression_weather_forecast/20261119-162635) | fixed | rmse | 24.588301 | 24.613677 | 24.212568 | 24.164154 | 23.923117 |
| [logistic_regression](results/advanced_0_7_logistic_regression/20261113-212619) | fixed | log_loss | 0.930236 | 0.920970 | 0.876390 | 0.856842 | 0.841432 |
| [logistic_regression + forecast](results/advanced_0_7_logistic_regression_forecast/20261119-162644) | fixed | log_loss | 0.929791 | 0.920103 | 0.875706 | 0.856061 | 0.840557 |
| [logistic_regression + weather](results/advanced_0_7_logistic_regression_weather/20261113-212624) | fixed | log_loss | 0.931270 | 0.924145 | 0.877911 | 0.857848 | 0.842388 |
| [logistic_regression + weather + forecast](results/advanced_0_7_logistic_regression_weather_forecast/20261119-162651) | fixed | log_loss | 0.931855 | 0.924428 | 0.878428 | 0.858195 | 0.842820 |
| [training_mean](results/advanced_0_7_training_mean/20261113-212613) | fixed | rmse | 25.423325 | 25.431851 | 25.403242 | 25.404544 | 25.404926 |
| [xgboost](results/advanced_0_7_xgboost/20261113-212830) | fixed | rmse | 24.478576 | 24.484618 | 24.085617 | 23.999789 | 23.697279 |
| [xgboost_classifier](results/advanced_0_7_xgboost_classifier/20261113-212944) | fixed | log_loss | 0.920912 | 0.911975 | 0.865809 | 0.848441 | 0.831868 |
| [xgboost_classifier + forecast](results/advanced_0_7_xgboost_classifier_forecast/20261119-163020) | fixed | log_loss | 0.920114 | 0.910251 | 0.868163 | 0.849405 | 0.831858 |
| [xgboost_classifier + weather](results/advanced_0_7_xgboost_classifier_weather/20261113-213111) | fixed | log_loss | 0.920159 | 0.913658 | 0.869298 | 0.850174 | 0.833422 |
| [xgboost_classifier + weather + forecast](results/advanced_0_7_xgboost_classifier_weather_forecast/20261119-163115) | fixed | log_loss | 0.920630 | 0.912798 | 0.870900 | 0.851810 | 0.834791 |
| [xgboost + forecast](results/advanced_0_7_xgboost_forecast/20261119-162741) | fixed | rmse | 24.455409 | 24.423039 | 24.119006 | 23.972568 | 23.705033 |
| [xgboost + weather](results/advanced_0_7_xgboost_weather/20261113-212859) | fixed | rmse | 24.445143 | 24.478797 | 24.209807 | 24.065709 | 23.766262 |
| [xgboost + weather + forecast](results/advanced_0_7_xgboost_weather_forecast/20261119-162757) | fixed | rmse | 24.449038 | 24.471733 | 24.221998 | 24.057736 | 23.787437 |

#### Step 0.8 starting scores with nested tuning

| Model | Search configs | Score | 24h | 6h | 1h | 20min | 0min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [lightgbm](results/advanced_0_8_lightgbm/20261113-221438) | 20 | rmse | 24.521828 | 24.555725 | 24.118380 | 24.022716 | 23.776330 |
| [lightgbm_classifier](results/advanced_0_8_lightgbm_classifier/20261113-230146) | 20 | log_loss | 0.924414 | 0.915752 | 0.868344 | 0.845636 | 0.833885 |
| [lightgbm_classifier + forecast](results/advanced_0_8_lightgbm_classifier_forecast/20261119-183411) | 20 | log_loss | 0.923411 | 0.912694 | 0.867925 | 0.848411 | 0.832955 |
| [lightgbm_classifier + weather](results/advanced_0_8_lightgbm_classifier_weather/20261114-000751) | 20 | log_loss | 0.922686 | 0.915433 | 0.870172 | 0.849489 | 0.833584 |
| [lightgbm_classifier + weather + forecast](results/advanced_0_8_lightgbm_classifier_weather_forecast/20261119-191602) | 20 | log_loss | 0.923066 | 0.914795 | 0.869869 | 0.850357 | 0.834324 |
| [lightgbm + forecast](results/advanced_0_8_lightgbm_forecast/20261119-171241) | 20 | rmse | 24.546744 | 24.558150 | 24.085562 | 24.037148 | 23.753036 |
| [lightgbm + weather](results/advanced_0_8_lightgbm_weather/20261113-223957) | 20 | rmse | 24.545003 | 24.574248 | 24.131414 | 24.071326 | 23.822866 |
| [lightgbm + weather + forecast](results/advanced_0_8_lightgbm_weather_forecast/20261119-173139) | 20 | rmse | 24.527007 | 24.589642 | 24.173748 | 24.130543 | 23.852086 |
| [xgboost](results/advanced_0_8_xgboost/20261113-224228) | 20 | rmse | 24.508199 | 24.520966 | 24.081305 | 24.000247 | 23.692860 |
| [xgboost_classifier](results/advanced_0_8_xgboost_classifier/20261114-001244) | 20 | log_loss | 0.920901 | 0.912369 | 0.867531 | 0.847626 | 0.832320 |
| [xgboost_classifier + weather](results/advanced_0_8_xgboost_classifier_weather/20261114-013919) | 20 | log_loss | 0.920159 | 0.913658 | 0.866848 | 0.847307 | 0.831830 |
| [xgboost + weather](results/advanced_0_8_xgboost_weather/20261113-233729) | 20 | rmse | 24.469813 | 24.486058 | 24.077068 | 24.065709 | 23.766262 |

#### Smallest differences by step, task and horizon (same folds for both runs)

| step | task | evaluation | prediction_time | method | variant | arm | contrast | mean | folds_improved | reaches_minimum_gain | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | delay | exploratory | 0min | lightgbm | recorded_route | forecast |  | -0.103568 | 4 | True | [folds](results/advanced_0_7_lightgbm_forecast_recorded_route/20261119-162657/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 1h | xgboost | recorded_route | weather |  | -0.107646 | 4 | True | [folds](results/advanced_0_7_xgboost_weather_recorded_route/20261113-212921/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 20min | lightgbm | recorded_route | forecast |  | -0.104656 | 4 | True | [folds](results/advanced_0_7_lightgbm_forecast_recorded_route/20261119-162657/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 24h | lightgbm | recorded_route | weather |  | -0.129485 | 4 | True | [folds](results/advanced_0_7_lightgbm_weather_recorded_route/20261113-212651/paired_vs_base.csv) |
| 0.7 | delay | exploratory | 6h | lightgbm | recorded_route | railway |  | -0.107571 | 5 | True | [folds](results/advanced_0_7_lightgbm_recorded_route/20261113-212642/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 0min | xgboost_classifier | recorded_route | weather |  | -0.004221 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_recorded_route/20261113-213135/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 1h | xgboost_classifier | recorded_route | weather_forecast |  | -0.003349 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_forecast_recorded_route/20261119-163150/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 20min | xgboost_classifier | recorded_route | weather |  | -0.003986 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_recorded_route/20261113-213135/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 24h | lightgbm_classifier | recorded_route | forecast |  | -0.004121 | 5 | True | [folds](results/advanced_0_7_lightgbm_classifier_forecast_recorded_route/20261119-162958/paired_vs_base.csv) |
| 0.7 | severity | exploratory | 6h | xgboost_classifier | recorded_route | weather_forecast |  | -0.004022 | 5 | True | [folds](results/advanced_0_7_xgboost_classifier_weather_forecast_recorded_route/20261119-163150/paired_vs_base.csv) |
| 0.8 | delay | nested | 0min | lightgbm | auto_groups | weather |  | -0.429845 | 4 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261123-235912/paired_vs_base.csv) |
| 0.8 | delay | nested | 1h | lightgbm | auto_groups | weather |  | -0.385901 | 5 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261123-235912/paired_vs_base.csv) |
| 0.8 | delay | nested | 20min | lightgbm | auto_groups | weather |  | -0.422638 | 4 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261123-235912/paired_vs_base.csv) |
| 0.8 | delay | nested | 24h | lightgbm | auto_groups | weather |  | -0.246184 | 5 | True | [folds](results/advanced_0_8_lightgbm_weather_auto_groups/20261123-235912/paired_vs_base.csv) |
| 0.8 | delay | nested | 6h | lightgbm | auto_groups | forecast |  | -0.393689 | 5 | True | [folds](results/advanced_0_8_lightgbm_forecast_auto_groups/20261124-000146/paired_vs_base.csv) |
| 0.8 | severity | nested | 0min | xgboost_classifier | auto_groups | weather |  | -0.014120 | 5 | True | [folds](results/advanced_0_8_xgboost_classifier_weather_auto_groups/20261124-054102/paired_vs_base.csv) |
| 0.8 | severity | nested | 1h | lightgbm_classifier | auto_groups | weather |  | -0.012686 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_auto_groups/20261124-022127/paired_vs_base.csv) |
| 0.8 | severity | nested | 20min | lightgbm_classifier | auto_groups | weather_forecast |  | -0.015494 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_forecast_auto_groups/20261124-031730/paired_vs_base.csv) |
| 0.8 | severity | nested | 24h | lightgbm_classifier | auto_groups | weather_forecast |  | -0.007470 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_weather_forecast_auto_groups/20261124-031730/paired_vs_base.csv) |
| 0.8 | severity | nested | 6h | lightgbm_classifier | auto_groups | railway |  | -0.014266 | 5 | True | [folds](results/advanced_0_8_lightgbm_classifier_auto_groups/20261114-040128/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 0min | lightgbm | calendar_cyclic | railway |  | -0.037526 | 4 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261114-081158/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 1h | lightgbm | calendar_cyclic | railway |  | -0.060085 | 5 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261114-081158/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 20min | lightgbm | calendar_cyclic | railway |  | -0.037351 | 5 | False | [folds](results/advanced_1_1_lightgbm_calendar_cyclic/20261114-081158/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 24h | linear_regression | calendar_combined | railway |  | -0.000006 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261114-081157/paired_vs_base.csv) |
| 1.1 | delay | exploratory | 6h | linear_regression | calendar_combined | railway |  | -0.000003 | 4 | False | [folds](results/advanced_1_1_linear_regression_calendar_combined/20261114-081157/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 0min | lightgbm_classifier | calendar_cyclic | railway |  | -0.001720 | 4 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261114-081302/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 1h | lightgbm_classifier | calendar_cyclic | railway |  | -0.002585 | 5 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261114-081302/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 20min | lightgbm_classifier | calendar_cyclic | railway |  | -0.001793 | 5 | False | [folds](results/advanced_1_1_lightgbm_classifier_calendar_cyclic/20261114-081302/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 24h | logistic_regression | calendar_combined | railway |  | +0.000078 | 2 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261114-081202/paired_vs_base.csv) |
| 1.1 | severity | exploratory | 6h | logistic_regression | calendar_combined | railway |  | +0.000043 | 2 | False | [folds](results/advanced_1_1_logistic_regression_calendar_combined/20261114-081202/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 0min | lightgbm | minute_of_day | railway |  | -0.049968 | 4 | False | [folds](results/advanced_1_2_lightgbm_minute_of_day/20261114-081354/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 1h | lightgbm | minute_of_day | railway |  | -0.041076 | 5 | False | [folds](results/advanced_1_2_lightgbm_minute_of_day/20261114-081354/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 20min | linear_regression | day_of_year | railway |  | -0.036690 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261114-081338/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 24h | linear_regression | day_of_year | railway |  | -0.043554 | 3 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261114-081338/paired_vs_base.csv) |
| 1.2 | delay | exploratory | 6h | linear_regression | day_of_year | railway |  | -0.043075 | 2 | False | [folds](results/advanced_1_2_linear_regression_day_of_year/20261114-081338/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 0min | lightgbm_classifier | minute_of_day | railway |  | -0.002075 | 4 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261114-081503/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 1h | logistic_regression | day_of_year | railway |  | -0.001733 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261114-081342/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 20min | logistic_regression | day_of_year | railway |  | -0.001853 | 3 | False | [folds](results/advanced_1_2_logistic_regression_day_of_year/20261114-081342/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 24h | lightgbm_classifier | minute_of_day | railway |  | -0.001301 | 3 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261114-081503/paired_vs_base.csv) |
| 1.2 | severity | exploratory | 6h | lightgbm_classifier | minute_of_day | railway |  | -0.001252 | 3 | False | [folds](results/advanced_1_2_lightgbm_classifier_minute_of_day/20261114-081503/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 0min | lightgbm | route_onehot | railway |  | -0.013491 | 4 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 1h | lightgbm | route_onehot | railway |  | -0.015656 | 3 | False | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 20min | linear_regression | route_target | railway |  | -0.019671 | 2 | False | [folds](results/advanced_1_3_linear_regression_route_target/20261114-081531/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 24h | lightgbm | route_onehot | railway |  | -0.109568 | 5 | True | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | delay | exploratory | 6h | lightgbm | route_onehot | railway |  | -0.112356 | 5 | True | [folds](results/advanced_1_3_lightgbm_route_onehot/20261114-081530/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 0min | lightgbm_classifier | route_onehot | railway |  | -0.001761 | 5 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 1h | lightgbm_classifier | route_onehot | railway |  | -0.001457 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 20min | lightgbm_classifier | route_onehot | railway |  | -0.001523 | 4 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 24h | lightgbm_classifier | route_onehot | railway |  | -0.001387 | 3 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.3 | severity | exploratory | 6h | lightgbm_classifier | route_onehot | railway |  | -0.001393 | 3 | False | [folds](results/advanced_1_3_lightgbm_classifier_route_onehot/20261114-081650/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 0min | lightgbm | delay_missing | railway |  | -0.005123 | 3 | False | [folds](results/advanced_1_4_lightgbm_delay_missing/20261114-081741/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 1h | lightgbm | delay_clipped | railway |  | -0.004195 | 5 | False | [folds](results/advanced_1_4_lightgbm_delay_clipped/20261114-081732/paired_vs_base.csv) |
| 1.4 | delay | exploratory | 20min | linear_regression | delay_clipped | railway |  | -0.002789 | 3 | False | [folds](results/advanced_1_4_linear_regression_delay_clipped/20261114-081723/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 0min | logistic_regression | delay_bins | railway |  | -0.000137 | 2 | False | [folds](results/advanced_1_4_logistic_regression_delay_bins/20261114-081728/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 1h | lightgbm_classifier | delay_missing | railway |  | -0.000599 | 5 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_missing/20261114-081834/paired_vs_base.csv) |
| 1.4 | severity | exploratory | 20min | lightgbm_classifier | delay_missing | railway |  | -0.000106 | 3 | False | [folds](results/advanced_1_4_lightgbm_classifier_delay_missing/20261114-081834/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 0min | lightgbm | weather_extremes | weather_forecast |  | -0.140177 | 4 | True | [folds](results/advanced_1_5_lightgbm_weather_forecast_weather_extremes/20261119-230148/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 1h | lightgbm | weather_extremes | weather_forecast |  | -0.164510 | 5 | True | [folds](results/advanced_1_5_lightgbm_weather_forecast_weather_extremes/20261119-230148/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 20min | lightgbm | weather_extremes | weather_forecast |  | -0.142909 | 3 | True | [folds](results/advanced_1_5_lightgbm_weather_forecast_weather_extremes/20261119-230148/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 24h | lightgbm | weather_extremes | forecast |  | -0.152433 | 4 | True | [folds](results/advanced_1_5_lightgbm_forecast_weather_extremes/20261119-230148/paired_vs_base.csv) |
| 1.5 | delay | exploratory | 6h | lightgbm | weather_extremes | weather_forecast |  | +0.020806 | 1 | False | [folds](results/advanced_1_5_lightgbm_weather_forecast_weather_extremes/20261119-230148/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 0min | lightgbm_classifier | weather_extremes | weather_forecast |  | -0.008511 | 4 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_forecast_weather_extremes/20261119-230300/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 1h | lightgbm_classifier | weather_extremes | weather_forecast |  | -0.009688 | 5 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_forecast_weather_extremes/20261119-230300/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 20min | lightgbm_classifier | weather_extremes | weather_forecast |  | -0.008230 | 4 | True | [folds](results/advanced_1_5_lightgbm_classifier_weather_forecast_weather_extremes/20261119-230300/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 24h | lightgbm_classifier | weather_extremes | forecast |  | -0.008232 | 4 | True | [folds](results/advanced_1_5_lightgbm_classifier_forecast_weather_extremes/20261119-230258/paired_vs_base.csv) |
| 1.5 | severity | exploratory | 6h | lightgbm_classifier | weather_extremes | weather_forecast |  | -0.000732 | 3 | False | [folds](results/advanced_1_5_lightgbm_classifier_weather_forecast_weather_extremes/20261119-230300/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 0min | linear_regression | recent_reliability | railway |  | -0.383250 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 1h | linear_regression | recent_reliability | railway |  | -0.392831 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 20min | linear_regression | recent_reliability | railway |  | -0.395945 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 24h | linear_regression | recent_reliability | railway |  | -0.249050 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | delay | exploratory | 6h | linear_regression | recent_reliability | railway |  | -0.402704 | 5 | True | [folds](results/advanced_2_1_linear_regression_recent_reliability/20261114-082052/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 0min | logistic_regression | recent_reliability | railway |  | -0.014766 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 1h | logistic_regression | recent_reliability | railway |  | -0.014679 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 20min | logistic_regression | recent_reliability | railway |  | -0.014624 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 24h | logistic_regression | recent_reliability | railway |  | -0.008698 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.1 | severity | exploratory | 6h | logistic_regression | recent_reliability | railway |  | -0.014694 | 5 | True | [folds](results/advanced_2_1_logistic_regression_recent_reliability/20261114-082058/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 0min | lightgbm | planned_timetable | railway |  | -0.117382 | 4 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 1h | lightgbm | planned_timetable | railway |  | -0.118120 | 4 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 20min | lightgbm | planned_timetable | railway |  | -0.126810 | 5 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 24h | lightgbm | planned_timetable | railway |  | -0.207707 | 5 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | delay | exploratory | 6h | lightgbm | planned_timetable | railway |  | -0.223698 | 5 | True | [folds](results/advanced_2_2_lightgbm_planned_timetable/20261114-082258/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 0min | logistic_regression | planned_timetable | railway |  | -0.002254 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261114-082248/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 1h | lightgbm_classifier | planned_timetable | railway |  | -0.002090 | 4 | False | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261114-082412/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 20min | logistic_regression | planned_timetable | railway |  | -0.001681 | 3 | False | [folds](results/advanced_2_2_logistic_regression_planned_timetable/20261114-082248/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 24h | lightgbm_classifier | planned_timetable | railway |  | -0.004379 | 4 | True | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261114-082412/paired_vs_base.csv) |
| 2.2 | severity | exploratory | 6h | lightgbm_classifier | planned_timetable | railway |  | -0.004231 | 4 | True | [folds](results/advanced_2_2_lightgbm_classifier_planned_timetable/20261114-082412/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 0min | lightgbm | holidays | railway |  | -0.002987 | 3 | False | [folds](results/advanced_2_3_lightgbm_holidays/20261114-082433/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 1h | linear_regression | holidays | railway |  | -0.004998 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261114-082417/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 20min | lightgbm | holidays | railway |  | -0.007948 | 2 | False | [folds](results/advanced_2_3_lightgbm_holidays/20261114-082433/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 24h | linear_regression | holidays | railway |  | -0.010071 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261114-082417/paired_vs_base.csv) |
| 2.3 | delay | exploratory | 6h | linear_regression | holidays | railway |  | -0.010398 | 3 | False | [folds](results/advanced_2_3_linear_regression_holidays/20261114-082417/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 0min | lightgbm_classifier | holidays | railway |  | +0.001639 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 1h | lightgbm_classifier | holidays | railway |  | +0.001081 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 20min | lightgbm_classifier | holidays | railway |  | +0.001629 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 24h | lightgbm_classifier | holidays | railway |  | +0.000274 | 3 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.3 | severity | exploratory | 6h | lightgbm_classifier | holidays | railway |  | +0.000178 | 2 | False | [folds](results/advanced_2_3_lightgbm_classifier_holidays/20261114-082547/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 0min | lightgbm | delay_dynamics | railway |  | -0.043798 | 5 | False | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261114-082600/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 1h | lightgbm | delay_dynamics | railway |  | -0.036659 | 4 | False | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261114-082600/paired_vs_base.csv) |
| 2.4 | delay | exploratory | 20min | lightgbm | delay_dynamics | railway |  | -0.060889 | 5 | False | [folds](results/advanced_2_4_lightgbm_delay_dynamics/20261114-082600/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 0min | lightgbm_classifier | delay_dynamics | railway |  | -0.002985 | 5 | False | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261114-082653/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 1h | lightgbm_classifier | delay_dynamics | railway |  | -0.003230 | 5 | True | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261114-082653/paired_vs_base.csv) |
| 2.4 | severity | exploratory | 20min | lightgbm_classifier | delay_dynamics | railway |  | -0.002352 | 5 | False | [folds](results/advanced_2_4_lightgbm_classifier_delay_dynamics/20261114-082653/paired_vs_base.csv) |
| 2.all | delay | exploratory | 0min | lightgbm | all_groups | railway |  | -0.386688 | 5 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261114-082725/paired_vs_base.csv) |
| 2.all | delay | exploratory | 1h | linear_regression | all_groups | railway |  | -0.333216 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261114-082701/paired_vs_base.csv) |
| 2.all | delay | exploratory | 20min | lightgbm | all_groups | railway |  | -0.378661 | 5 | True | [folds](results/advanced_2_all_lightgbm_all_groups/20261114-082725/paired_vs_base.csv) |
| 2.all | delay | exploratory | 24h | linear_regression | all_groups | railway |  | -0.244677 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261114-082701/paired_vs_base.csv) |
| 2.all | delay | exploratory | 6h | linear_regression | all_groups | railway |  | -0.382828 | 5 | True | [folds](results/advanced_2_all_linear_regression_all_groups/20261114-082701/paired_vs_base.csv) |
| 2.all | severity | exploratory | 0min | lightgbm_classifier | all_groups | railway |  | -0.011990 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 1h | lightgbm_classifier | all_groups | railway |  | -0.010606 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 20min | lightgbm_classifier | all_groups | railway |  | -0.010677 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 24h | lightgbm_classifier | all_groups | railway |  | -0.005692 | 5 | True | [folds](results/advanced_2_all_lightgbm_classifier_all_groups/20261114-082910/paired_vs_base.csv) |
| 2.all | severity | exploratory | 6h | logistic_regression | all_groups | railway |  | -0.011190 | 5 | True | [folds](results/advanced_2_all_logistic_regression_all_groups/20261114-082717/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | linear_regression | drop_route | forecast | full | -0.037511 | 3 | False | [folds](results/advanced_3_1_linear_regression_forecast_drop_route/20261119-230448/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 0min | lightgbm | add_delay | forecast | timetable-only | -0.965828 | 5 | True | [folds](results/advanced_3_1_lightgbm_forecast_add_delay/20261119-230606/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | lightgbm | drop_wind | weather_forecast | full | -0.035293 | 4 | False | [folds](results/advanced_3_1_lightgbm_weather_forecast_drop_wind/20261119-231606/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 1h | linear_regression | add_recent_reliability | forecast | timetable-only | -0.819301 | 5 | True | [folds](results/advanced_3_1_linear_regression_forecast_add_recent_reliability/20261119-230446/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | lightgbm | drop_temperature | weather | full | -0.046254 | 3 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_temperature/20261114-083124/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 20min | linear_regression | add_recent_reliability | forecast | timetable-only | -0.812501 | 5 | True | [folds](results/advanced_3_1_linear_regression_forecast_add_recent_reliability/20261119-230446/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | lightgbm | drop_precipitation | weather | full | -0.048063 | 5 | False | [folds](results/advanced_3_1_lightgbm_weather_drop_precipitation/20261114-083133/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 24h | linear_regression | add_calendar | forecast | timetable-only | -0.560817 | 5 | True | [folds](results/advanced_3_1_linear_regression_forecast_add_calendar/20261119-230447/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | drop_temperature | weather_forecast | full | -0.024329 | 5 | False | [folds](results/advanced_3_1_linear_regression_weather_forecast_drop_temperature/20261119-231114/paired_vs_base.csv) |
| 3.1 | delay | exploratory | 6h | linear_regression | add_recent_reliability | forecast | timetable-only | -0.767714 | 5 | True | [folds](results/advanced_3_1_linear_regression_forecast_add_recent_reliability/20261119-230446/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | lightgbm_classifier | drop_forecast_wind | forecast | full | -0.002161 | 5 | False | [folds](results/advanced_3_1_lightgbm_classifier_forecast_drop_forecast_wind/20261119-231316/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 0min | logistic_regression | add_recent_reliability | forecast | timetable-only | -0.032062 | 5 | True | [folds](results/advanced_3_1_logistic_regression_forecast_add_recent_reliability/20261119-230507/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | drop_route | weather_forecast | full | -0.002114 | 4 | False | [folds](results/advanced_3_1_logistic_regression_weather_forecast_drop_route/20261119-231237/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 1h | logistic_regression | add_recent_reliability | forecast | timetable-only | -0.031938 | 5 | True | [folds](results/advanced_3_1_logistic_regression_forecast_add_recent_reliability/20261119-230507/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | drop_route | weather | full | -0.002094 | 5 | False | [folds](results/advanced_3_1_logistic_regression_weather_drop_route/20261114-082946/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 20min | logistic_regression | add_recent_reliability | forecast | timetable-only | -0.031606 | 5 | True | [folds](results/advanced_3_1_logistic_regression_forecast_add_recent_reliability/20261119-230507/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | drop_temperature | weather_forecast | full | -0.001727 | 5 | False | [folds](results/advanced_3_1_logistic_regression_weather_forecast_drop_temperature/20261119-231401/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 24h | logistic_regression | add_recent_reliability | forecast | timetable-only | -0.022182 | 5 | True | [folds](results/advanced_3_1_logistic_regression_forecast_add_recent_reliability/20261119-230507/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | drop_temperature | weather_forecast | full | -0.002252 | 5 | False | [folds](results/advanced_3_1_logistic_regression_weather_forecast_drop_temperature/20261119-231401/paired_vs_base.csv) |
| 3.1 | severity | exploratory | 6h | logistic_regression | add_recent_reliability | forecast | timetable-only | -0.030013 | 5 | True | [folds](results/advanced_3_1_logistic_regression_forecast_add_recent_reliability/20261119-230507/paired_vs_base.csv) |
| 3.3 | delay | nested | 0min | linear_regression | backward | railway |  | -0.042898 | 3 | False | [folds](results/advanced_3_3_linear_regression_backward/20261114-090529/paired_vs_base.csv) |
| 3.3 | delay | nested | 1h | linear_regression | forward | forecast |  | -0.049022 | 3 | False | [folds](results/advanced_3_3_linear_regression_forecast_forward/20261120-000533/paired_vs_base.csv) |
| 3.3 | delay | nested | 20min | linear_regression | backward | weather_forecast |  | -0.066757 | 3 | False | [folds](results/advanced_3_3_linear_regression_weather_forecast_backward/20261120-001546/paired_vs_base.csv) |
| 3.3 | delay | nested | 24h | linear_regression | forward | weather |  | +0.017868 | 3 | False | [folds](results/advanced_3_3_linear_regression_weather_forward/20261114-090701/paired_vs_base.csv) |
| 3.3 | delay | nested | 6h | linear_regression | backward | weather |  | +0.018640 | 1 | False | [folds](results/advanced_3_3_linear_regression_weather_backward/20261114-090837/paired_vs_base.csv) |
| 3.3 | severity | nested | 0min | logistic_regression | backward | weather_forecast |  | -0.002717 | 4 | False | [folds](results/advanced_3_3_logistic_regression_weather_forecast_backward/20261120-005732/paired_vs_base.csv) |
| 3.3 | severity | nested | 1h | logistic_regression | forward | forecast |  | -0.003559 | 5 | True | [folds](results/advanced_3_3_logistic_regression_forecast_forward/20261120-001521/paired_vs_base.csv) |
| 3.3 | severity | nested | 20min | logistic_regression | backward | weather_forecast |  | -0.002916 | 4 | False | [folds](results/advanced_3_3_logistic_regression_weather_forecast_backward/20261120-005732/paired_vs_base.csv) |
| 3.3 | severity | nested | 24h | logistic_regression | backward | weather_forecast |  | -0.002581 | 4 | False | [folds](results/advanced_3_3_logistic_regression_weather_forecast_backward/20261120-005732/paired_vs_base.csv) |
| 3.3 | severity | nested | 6h | logistic_regression | backward | weather_forecast |  | -0.003516 | 4 | True | [folds](results/advanced_3_3_logistic_regression_weather_forecast_backward/20261120-005732/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 0min | logistic_regression | l1 | weather_forecast |  | -0.004607 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_forecast_l1/20261120-003435/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 1h | logistic_regression | l1 | weather_forecast |  | -0.005002 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_forecast_l1/20261120-003435/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 20min | logistic_regression | l1 | weather_forecast |  | -0.005063 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_forecast_l1/20261120-003435/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 24h | logistic_regression | l1 | weather_forecast |  | -0.004538 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_forecast_l1/20261120-003435/paired_vs_base.csv) |
| 3.3 | severity | prespecified | 6h | logistic_regression | l1 | weather_forecast |  | -0.005303 | 5 | True | [folds](results/advanced_3_3_logistic_regression_weather_forecast_l1/20261120-003435/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 0min | lightgbm | weather_extremes | weather_forecast |  | -0.086510 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_forecast_weather_extremes/20261120-005816/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 1h | lightgbm | weather_extremes | weather_forecast |  | -0.070507 | 3 | False | [folds](results/advanced_3_4_lightgbm_weather_forecast_weather_extremes/20261120-005816/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 20min | lightgbm | weather_extremes | weather_forecast |  | -0.109324 | 4 | True | [folds](results/advanced_3_4_lightgbm_weather_forecast_weather_extremes/20261120-005816/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 24h | lightgbm | weather_extremes | weather_forecast |  | -0.110163 | 2 | True | [folds](results/advanced_3_4_lightgbm_weather_forecast_weather_extremes/20261120-005816/paired_vs_base.csv) |
| 3.4 | delay | exploratory | 6h | lightgbm | temperature | weather |  | -0.016882 | 4 | False | [folds](results/advanced_3_4_lightgbm_weather_temperature/20261114-092933/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 0min | lightgbm_classifier | weather_extremes | weather_forecast |  | -0.004705 | 5 | True | [folds](results/advanced_3_4_lightgbm_classifier_weather_forecast_weather_extremes/20261120-010259/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 1h | lightgbm_classifier | weather_extremes | weather_forecast |  | -0.003031 | 4 | True | [folds](results/advanced_3_4_lightgbm_classifier_weather_forecast_weather_extremes/20261120-010259/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 20min | lightgbm_classifier | weather_extremes | weather_forecast |  | -0.003736 | 5 | True | [folds](results/advanced_3_4_lightgbm_classifier_weather_forecast_weather_extremes/20261120-010259/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 24h | lightgbm_classifier | temperature | weather_forecast |  | -0.002603 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_forecast_temperature/20261120-010128/paired_vs_base.csv) |
| 3.4 | severity | exploratory | 6h | lightgbm_classifier | temperature | weather_forecast |  | -0.001155 | 4 | False | [folds](results/advanced_3_4_lightgbm_classifier_weather_forecast_temperature/20261120-010128/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_class | weather |  | +0.000105 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261114-093418/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_class | weather_forecast |  | -0.000701 | 4 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_forecast_cancellation_then_class/20261120-010548/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_class | weather |  | -0.000312 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_weather_cancellation_then_class/20261114-093418/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_class | forecast |  | -0.000758 | 3 | False | [folds](results/advanced_4_1_lightgbm_classifier_forecast_cancellation_then_class/20261120-010459/paired_vs_base.csv) |
| 4.1 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_class | forecast |  | +0.000054 | 2 | False | [folds](results/advanced_4_1_lightgbm_classifier_forecast_cancellation_then_class/20261120-010459/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 0min | quantile_forest_severity |  | weather_forecast |  | +0.025696 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather_forecast/20261120-013315/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 1h | quantile_forest_severity |  | weather_forecast |  | +0.021421 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather_forecast/20261120-013315/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 20min | quantile_forest_severity |  | weather_forecast |  | +0.022344 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather_forecast/20261120-013315/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 24h | quantile_forest_severity |  | weather |  | +0.021106 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261114-222012/paired_vs_base.csv) |
| 4.2 | severity | exploratory | 6h | quantile_forest_severity |  | weather |  | +0.027903 | 0 | False | [folds](results/advanced_4_2_quantile_forest_severity_weather/20261114-222012/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 0min | lightgbm_classifier | cancellation_then_ordinal | weather |  | +0.001280 | 1 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261114-093557/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 1h | lightgbm_classifier | cancellation_then_ordinal | weather_forecast |  | +0.000853 | 1 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_forecast_cancellation_then_ordinal/20261120-010802/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 20min | lightgbm_classifier | cancellation_then_ordinal | weather |  | +0.000391 | 2 | False | [folds](results/advanced_4_3_lightgbm_classifier_weather_cancellation_then_ordinal/20261114-093557/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 24h | lightgbm_classifier | cancellation_then_ordinal | forecast |  | +0.000484 | 2 | False | [folds](results/advanced_4_3_lightgbm_classifier_forecast_cancellation_then_ordinal/20261120-010724/paired_vs_base.csv) |
| 4.3 | severity | exploratory | 6h | lightgbm_classifier | cancellation_then_ordinal | forecast |  | +0.001407 | 0 | False | [folds](results/advanced_4_3_lightgbm_classifier_forecast_cancellation_then_ordinal/20261120-010724/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 0min | lightgbm | mixture_of_experts | weather_forecast |  | -0.054439 | 4 | False | [folds](results/advanced_4_4_lightgbm_weather_forecast_mixture_of_experts/20261120-010933/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 1h | lightgbm | mixture_of_experts | railway |  | -0.081268 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261114-093712/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 20min | lightgbm | mixture_of_experts | weather_forecast |  | -0.051201 | 4 | False | [folds](results/advanced_4_4_lightgbm_weather_forecast_mixture_of_experts/20261120-010933/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 24h | lightgbm | mixture_of_experts | weather_forecast |  | -0.050354 | 3 | False | [folds](results/advanced_4_4_lightgbm_weather_forecast_mixture_of_experts/20261120-010933/paired_vs_base.csv) |
| 4.4 | delay | exploratory | 6h | lightgbm | mixture_of_experts | railway |  | -0.036355 | 5 | False | [folds](results/advanced_4_4_lightgbm_mixture_of_experts/20261114-093712/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 0min | lightgbm | change_then_level | weather_forecast |  | +0.003545 | 3 | False | [folds](results/advanced_4_5_lightgbm_weather_forecast_change_then_level/20261120-010956/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 1h | lightgbm | change_then_level | railway |  | -0.009455 | 3 | False | [folds](results/advanced_4_5_lightgbm_change_then_level/20261114-093727/paired_vs_base.csv) |
| 4.5 | delay | exploratory | 20min | lightgbm | change_then_level | weather_forecast |  | +0.005027 | 2 | False | [folds](results/advanced_4_5_lightgbm_weather_forecast_change_then_level/20261120-010956/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 0min | lightgbm | plus_catboost | weather |  | -0.063649 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_catboost/20261115-055728/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 1h | lightgbm | plus_linear | weather |  | -0.104592 | 3 | True | [folds](results/advanced_5_1_lightgbm_weather_plus_linear/20261115-055728/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 20min | lightgbm | plus_linear | weather |  | -0.076194 | 3 | False | [folds](results/advanced_5_1_lightgbm_weather_plus_linear/20261115-055728/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 24h | lightgbm | plus_linear | weather_forecast |  | -0.027705 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_forecast_plus_linear/20261120-015920/paired_vs_base.csv) |
| 5.1 | delay | exploratory | 6h | lightgbm | plus_linear | weather_forecast |  | -0.034312 | 4 | False | [folds](results/advanced_5_1_lightgbm_weather_forecast_plus_linear/20261120-015920/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 0min | lightgbm_classifier | plus_catboost | weather |  | -0.003296 | 4 | True | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 1h | lightgbm_classifier | plus_catboost | weather |  | -0.003655 | 5 | True | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 20min | lightgbm_classifier | plus_catboost | weather |  | -0.004283 | 5 | True | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 24h | lightgbm_classifier | plus_catboost | weather |  | -0.001245 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_weather_plus_catboost/20261115-055729/paired_vs_base.csv) |
| 5.1 | severity | exploratory | 6h | lightgbm_classifier | geometric | railway |  | -0.000596 | 4 | False | [folds](results/advanced_5_1_lightgbm_classifier_geometric/20261115-055728/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 0min | xgboost | top5_flat | railway |  | -0.044814 | 5 | False | [folds](results/advanced_5_2_xgboost_top5_flat/20261115-055915/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 1h | xgboost | top5_flat | weather_forecast |  | -0.121145 | 5 | True | [folds](results/advanced_5_2_xgboost_weather_forecast_top5_flat/20261120-020725/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 20min | lightgbm | top5_flat | forecast |  | -0.057763 | 5 | False | [folds](results/advanced_5_2_lightgbm_forecast_top5_flat/20261120-020115/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 24h | lightgbm | top5_flat | forecast |  | -0.135763 | 5 | True | [folds](results/advanced_5_2_lightgbm_forecast_top5_flat/20261120-020115/paired_vs_base.csv) |
| 5.2 | delay | exploratory | 6h | lightgbm | seeds | railway |  | -0.037025 | 4 | False | [folds](results/advanced_5_2_lightgbm_seeds/20261115-055922/paired_vs_base.csv) |
| 5.2 | delay | nested | 0min | lightgbm | top5_nested | weather |  | -0.097550 | 4 | False | [folds](results/advanced_5_2_lightgbm_weather_top5_nested/20261115-060105/paired_vs_base.csv) |
| 5.2 | delay | nested | 1h | lightgbm | top5_nested | weather_forecast |  | -0.063314 | 5 | False | [folds](results/advanced_5_2_lightgbm_weather_forecast_top5_nested/20261120-020049/paired_vs_base.csv) |
| 5.2 | delay | nested | 20min | lightgbm | top5_nested | forecast |  | -0.090016 | 5 | False | [folds](results/advanced_5_2_lightgbm_forecast_top5_nested/20261120-020024/paired_vs_base.csv) |
| 5.2 | delay | nested | 24h | lightgbm | top5_nested | forecast |  | -0.082676 | 5 | False | [folds](results/advanced_5_2_lightgbm_forecast_top5_nested/20261120-020024/paired_vs_base.csv) |
| 5.2 | delay | nested | 6h | lightgbm | top5_nested | forecast |  | -0.111321 | 5 | True | [folds](results/advanced_5_2_lightgbm_forecast_top5_nested/20261120-020024/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 0min | xgboost_classifier | top5_flat | railway |  | -0.002645 | 5 | False | [folds](results/advanced_5_2_xgboost_classifier_top5_flat/20261115-060637/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 1h | lightgbm_classifier | top5_flat | railway |  | -0.002322 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_flat/20261115-060209/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 20min | xgboost_classifier | top5_flat | weather_forecast |  | -0.003514 | 5 | True | [folds](results/advanced_5_2_xgboost_classifier_weather_forecast_top5_flat/20261120-022017/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 24h | lightgbm_classifier | top5_flat | forecast |  | -0.005712 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_forecast_top5_flat/20261120-020724/paired_vs_base.csv) |
| 5.2 | severity | exploratory | 6h | lightgbm_classifier | top5_flat | weather |  | -0.001888 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_weather_top5_flat/20261115-060817/paired_vs_base.csv) |
| 5.2 | severity | nested | 0min | lightgbm_classifier | top5_nested | railway |  | -0.004957 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.2 | severity | nested | 1h | lightgbm_classifier | top5_nested | railway |  | -0.002698 | 5 | False | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.2 | severity | nested | 20min | xgboost_classifier | top5_nested | railway |  | -0.001477 | 5 | False | [folds](results/advanced_5_2_xgboost_classifier_top5_nested/20261115-060637/paired_vs_base.csv) |
| 5.2 | severity | nested | 24h | lightgbm_classifier | top5_nested | railway |  | -0.003319 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.2 | severity | nested | 6h | lightgbm_classifier | top5_nested | railway |  | -0.003799 | 5 | True | [folds](results/advanced_5_2_lightgbm_classifier_top5_nested/20261115-055934/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 0min | lightgbm | greedy | weather |  | -0.036564 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 1h | lightgbm | greedy | weather |  | -0.100464 | 4 | True | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 20min | lightgbm | greedy | weather |  | -0.071008 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 24h | lightgbm | greedy | weather_forecast |  | -0.034836 | 4 | False | [folds](results/advanced_5_3_lightgbm_weather_forecast_greedy/20261120-022421/paired_vs_base.csv) |
| 5.3 | delay | exploratory | 6h | lightgbm | greedy | weather |  | -0.015177 | 2 | False | [folds](results/advanced_5_3_lightgbm_weather_greedy/20261115-061906/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 0min | lightgbm_classifier | greedy | weather |  | -0.002335 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_greedy/20261115-063353/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 1h | lightgbm_classifier | greedy | weather_forecast |  | -0.003104 | 5 | True | [folds](results/advanced_5_3_lightgbm_classifier_weather_forecast_greedy/20261120-024405/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 20min | lightgbm_classifier | greedy | weather_forecast |  | -0.003592 | 5 | True | [folds](results/advanced_5_3_lightgbm_classifier_weather_forecast_greedy/20261120-024405/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 24h | lightgbm_classifier | greedy | weather_forecast |  | -0.000881 | 3 | False | [folds](results/advanced_5_3_lightgbm_classifier_weather_forecast_greedy/20261120-024405/paired_vs_base.csv) |
| 5.3 | severity | exploratory | 6h | lightgbm_classifier | greedy | railway |  | -0.000368 | 4 | False | [folds](results/advanced_5_3_lightgbm_classifier_greedy/20261115-062543/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 0min | lightgbm | stack | weather |  | -0.037648 | 4 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 1h | lightgbm | stack | weather |  | -0.118885 | 4 | True | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 20min | lightgbm | stack | weather |  | -0.082396 | 4 | False | [folds](results/advanced_5_4_lightgbm_weather_stack/20261115-063702/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 24h | lightgbm | stack | weather_forecast |  | -0.043530 | 3 | False | [folds](results/advanced_5_4_lightgbm_weather_forecast_stack/20261120-024808/paired_vs_base.csv) |
| 5.4 | delay | exploratory | 6h | lightgbm | stack | weather_forecast |  | -0.010000 | 3 | False | [folds](results/advanced_5_4_lightgbm_weather_forecast_stack/20261120-024808/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 0min | lightgbm_classifier | stack | weather |  | +0.000477 | 3 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 1h | lightgbm_classifier | stack | weather_forecast |  | -0.000122 | 3 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_forecast_stack/20261120-030753/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 20min | lightgbm_classifier | stack | weather_forecast |  | -0.001453 | 3 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_forecast_stack/20261120-030753/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 24h | lightgbm_classifier | stack | weather |  | +0.002419 | 1 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 5.4 | severity | exploratory | 6h | lightgbm_classifier | stack | weather |  | +0.002416 | 2 | False | [folds](results/advanced_5_4_lightgbm_classifier_weather_stack/20261115-065142/paired_vs_base.csv) |
| 6.1 | delay | nested | 0min | catboost |  | weather |  | -0.016223 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | delay | nested | 1h | catboost |  | weather |  | -0.111373 | 5 | True | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | delay | nested | 20min | catboost |  | railway |  | +0.012945 | 2 | False | [folds](results/advanced_6_1_catboost/20261114-113030/paired_vs_base.csv) |
| 6.1 | delay | nested | 24h | catboost |  | weather |  | -0.034199 | 3 | False | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | delay | nested | 6h | catboost |  | weather |  | -0.029309 | 4 | False | [folds](results/advanced_6_1_catboost_weather/20261114-122923/paired_vs_base.csv) |
| 6.1 | severity | nested | 0min | catboost_classifier |  | weather |  | -0.000027 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 1h | catboost_classifier |  | weather |  | +0.000227 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 20min | catboost_classifier |  | weather |  | +0.001187 | 2 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 24h | catboost_classifier |  | weather |  | +0.000487 | 3 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.1 | severity | nested | 6h | catboost_classifier |  | weather |  | +0.000960 | 1 | False | [folds](results/advanced_6_1_catboost_classifier_weather/20261115-055718/paired_vs_base.csv) |
| 6.2 | severity | nested | 0min | lightgbm_classifier | temperature | railway |  | -0.000581 | 4 | False | [folds](results/advanced_6_2_lightgbm_classifier_temperature/20261115-065325/paired_vs_base.csv) |
| 6.2 | severity | nested | 1h | lightgbm_classifier | temperature | weather |  | -0.000378 | 4 | False | [folds](results/advanced_6_2_lightgbm_classifier_weather_temperature/20261115-065502/paired_vs_base.csv) |
| 6.2 | severity | nested | 20min | lightgbm_classifier | temperature | railway |  | -0.000366 | 3 | False | [folds](results/advanced_6_2_lightgbm_classifier_temperature/20261115-065325/paired_vs_base.csv) |
| 6.2 | severity | nested | 24h | lightgbm_classifier | temperature | railway |  | -0.000196 | 4 | False | [folds](results/advanced_6_2_lightgbm_classifier_temperature/20261115-065325/paired_vs_base.csv) |
| 6.2 | severity | nested | 6h | catboost_classifier | temperature | railway |  | -0.000459 | 4 | False | [folds](results/advanced_6_2_catboost_classifier_temperature/20261115-072723/paired_vs_base.csv) |
| 6.3 | delay | nested | 0min | ebm |  | railway |  | +0.154187 | 0 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 1h | ebm |  | railway |  | +0.214240 | 1 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 20min | ebm |  | railway |  | +0.257095 | 1 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 24h | ebm |  | railway |  | +0.112140 | 0 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | delay | nested | 6h | ebm |  | railway |  | +0.086063 | 1 | False | [folds](results/advanced_6_3_ebm/20261114-162523/paired_vs_base.csv) |
| 6.3 | severity | nested | 0min | ebm_classifier |  | railway |  | +0.009253 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 1h | ebm_classifier |  | railway |  | +0.011103 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 20min | ebm_classifier |  | railway |  | +0.015673 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 24h | ebm_classifier |  | railway |  | +0.010166 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.3 | severity | nested | 6h | ebm_classifier |  | railway |  | +0.009921 | 0 | False | [folds](results/advanced_6_3_ebm_classifier/20261114-162949/paired_vs_base.csv) |
| 6.5 | delay | nested | 0min | ngboost |  | railway |  | -0.016746 | 2 | False | [folds](results/advanced_6_5_ngboost/20261114-220700/paired_vs_base.csv) |
| 6.5 | delay | nested | 1h | ngboost |  | forecast |  | -0.020427 | 3 | False | [folds](results/advanced_6_5_ngboost_forecast/20261120-015138/paired_vs_base.csv) |
| 6.5 | delay | nested | 20min | ngboost |  | railway |  | +0.032981 | 3 | False | [folds](results/advanced_6_5_ngboost/20261114-220700/paired_vs_base.csv) |
| 6.5 | delay | nested | 24h | ngboost |  | forecast |  | +0.024547 | 3 | False | [folds](results/advanced_6_5_ngboost_forecast/20261120-015138/paired_vs_base.csv) |
| 6.5 | delay | nested | 6h | ngboost |  | forecast |  | -0.037456 | 4 | False | [folds](results/advanced_6_5_ngboost_forecast/20261120-015138/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 0min | lightgbm | log_shift | weather_forecast |  | +0.072627 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_forecast_log_shift/20261120-032330/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 1h | lightgbm | log_shift | weather |  | +0.047583 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261114-163707/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 20min | lightgbm | log_shift | weather |  | +0.056928 | 2 | False | [folds](results/advanced_6_6_lightgbm_weather_log_shift/20261114-163707/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 24h | lightgbm | log_shift | forecast |  | +0.059750 | 2 | False | [folds](results/advanced_6_6_lightgbm_forecast_log_shift/20261120-032326/paired_vs_base.csv) |
| 6.6 | delay | exploratory | 6h | lightgbm | log_shift | railway |  | +0.204537 | 0 | False | [folds](results/advanced_6_6_lightgbm_log_shift/20261114-163652/paired_vs_base.csv) |
| 7.1 | delay | nested | 0min | mlp |  | railway |  | +0.425626 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 1h | mlp |  | railway |  | +0.349914 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 20min | mlp |  | railway |  | +0.437380 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 24h | mlp |  | railway |  | +0.461818 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | delay | nested | 6h | mlp |  | railway |  | +0.380262 | 0 | False | [folds](results/advanced_7_1_mlp/20261114-165121/paired_vs_base.csv) |
| 7.1 | severity | nested | 0min | mlp_classifier |  | railway |  | +0.012711 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 1h | mlp_classifier |  | railway |  | +0.015032 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 20min | mlp_classifier |  | railway |  | +0.017340 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 24h | mlp_classifier |  | railway |  | +0.016576 | 1 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.1 | severity | nested | 6h | mlp_classifier |  | railway |  | +0.014627 | 0 | False | [folds](results/advanced_7_1_mlp_classifier/20261114-165118/paired_vs_base.csv) |
| 7.2 | delay | nested | 0min | multitask |  | weather_forecast |  | -0.391624 | 4 | True | [folds](results/advanced_7_2_multitask_weather_forecast/20261120-041417/paired_vs_base.csv) |
| 7.2 | delay | nested | 1h | multitask |  | forecast |  | -0.270089 | 4 | True | [folds](results/advanced_7_2_multitask_forecast/20261120-041442/paired_vs_base.csv) |
| 7.2 | delay | nested | 20min | multitask |  | weather_forecast |  | -0.532935 | 5 | True | [folds](results/advanced_7_2_multitask_weather_forecast/20261120-041417/paired_vs_base.csv) |
| 7.2 | delay | nested | 24h | multitask |  | forecast |  | -0.219541 | 4 | True | [folds](results/advanced_7_2_multitask_forecast/20261120-041442/paired_vs_base.csv) |
| 7.2 | delay | nested | 6h | multitask |  | weather_forecast |  | -0.272310 | 4 | True | [folds](results/advanced_7_2_multitask_weather_forecast/20261120-041417/paired_vs_base.csv) |
| 7.2 | severity | nested | 0min | multitask_classifier |  | weather_forecast |  | -0.003675 | 4 | True | [folds](results/advanced_7_2_multitask_classifier_weather_forecast/20261120-041451/paired_vs_base.csv) |
| 7.2 | severity | nested | 1h | multitask_classifier |  | railway |  | -0.001589 | 2 | False | [folds](results/advanced_7_2_multitask_classifier/20261114-173937/paired_vs_base.csv) |
| 7.2 | severity | nested | 20min | multitask_classifier |  | weather_forecast |  | -0.004564 | 4 | True | [folds](results/advanced_7_2_multitask_classifier_weather_forecast/20261120-041451/paired_vs_base.csv) |
| 7.2 | severity | nested | 24h | multitask_classifier |  | railway |  | -0.001436 | 2 | False | [folds](results/advanced_7_2_multitask_classifier/20261114-173937/paired_vs_base.csv) |
| 7.2 | severity | nested | 6h | multitask_classifier |  | weather_forecast |  | -0.001080 | 4 | False | [folds](results/advanced_7_2_multitask_classifier_weather_forecast/20261120-041451/paired_vs_base.csv) |
| 7.3 | delay | nested | 0min | tabgns |  | weather_forecast |  | -0.845147 | 5 | True | [folds](results/advanced_7_3_tabgns_weather_forecast/20261120-042653/paired_vs_base.csv) |
| 7.3 | delay | nested | 1h | tabgns |  | weather |  | -0.602559 | 5 | True | [folds](results/advanced_7_3_tabgns_weather/20261114-181426/paired_vs_base.csv) |
| 7.3 | delay | nested | 20min | tabgns |  | weather_forecast |  | -0.949654 | 5 | True | [folds](results/advanced_7_3_tabgns_weather_forecast/20261120-042653/paired_vs_base.csv) |
| 7.3 | delay | nested | 24h | tabgns |  | weather_forecast |  | -0.801969 | 5 | True | [folds](results/advanced_7_3_tabgns_weather_forecast/20261120-042653/paired_vs_base.csv) |
| 7.3 | delay | nested | 6h | tabgns |  | weather_forecast |  | -0.793875 | 5 | True | [folds](results/advanced_7_3_tabgns_weather_forecast/20261120-042653/paired_vs_base.csv) |
| 7.3 | severity | nested | 0min | tabgns_classifier |  | weather_forecast |  | -0.011705 | 5 | True | [folds](results/advanced_7_3_tabgns_classifier_weather_forecast/20261120-042534/paired_vs_base.csv) |
| 7.3 | severity | nested | 1h | tabgns_classifier |  | weather_forecast |  | -0.008042 | 5 | True | [folds](results/advanced_7_3_tabgns_classifier_weather_forecast/20261120-042534/paired_vs_base.csv) |
| 7.3 | severity | nested | 20min | tabgns_classifier |  | weather_forecast |  | -0.013248 | 4 | True | [folds](results/advanced_7_3_tabgns_classifier_weather_forecast/20261120-042534/paired_vs_base.csv) |
| 7.3 | severity | nested | 24h | tabgns_classifier |  | weather_forecast |  | -0.017823 | 5 | True | [folds](results/advanced_7_3_tabgns_classifier_weather_forecast/20261120-042534/paired_vs_base.csv) |
| 7.3 | severity | nested | 6h | tabgns_classifier |  | weather_forecast |  | -0.018349 | 5 | True | [folds](results/advanced_7_3_tabgns_classifier_weather_forecast/20261120-042534/paired_vs_base.csv) |
| 7.4 | delay | nested | 0min | tabm |  | weather_forecast |  | -1.029356 | 5 | True | [folds](results/advanced_7_4_tabm_weather_forecast/20261120-052432/paired_vs_base.csv) |
| 7.4 | delay | nested | 1h | tabm |  | weather_forecast |  | -0.782614 | 5 | True | [folds](results/advanced_7_4_tabm_weather_forecast/20261120-052432/paired_vs_base.csv) |
| 7.4 | delay | nested | 20min | tabm |  | weather_forecast |  | -1.187849 | 5 | True | [folds](results/advanced_7_4_tabm_weather_forecast/20261120-052432/paired_vs_base.csv) |
| 7.4 | delay | nested | 24h | tabm |  | weather_forecast |  | -1.102878 | 5 | True | [folds](results/advanced_7_4_tabm_weather_forecast/20261120-052432/paired_vs_base.csv) |
| 7.4 | delay | nested | 6h | tabm |  | weather_forecast |  | -1.074188 | 5 | True | [folds](results/advanced_7_4_tabm_weather_forecast/20261120-052432/paired_vs_base.csv) |
| 7.4 | severity | nested | 0min | realmlp_classifier |  | weather_forecast |  | -0.021886 | 5 | True | [folds](results/advanced_7_4_realmlp_classifier_weather_forecast/20261120-044626/paired_vs_base.csv) |
| 7.4 | severity | nested | 1h | tabm_classifier |  | weather_forecast |  | -0.015534 | 4 | True | [folds](results/advanced_7_4_tabm_classifier_weather_forecast/20261120-054423/paired_vs_base.csv) |
| 7.4 | severity | nested | 20min | tabm_classifier |  | weather_forecast |  | -0.018865 | 5 | True | [folds](results/advanced_7_4_tabm_classifier_weather_forecast/20261120-054423/paired_vs_base.csv) |
| 7.4 | severity | nested | 24h | realmlp_classifier |  | weather_forecast |  | -0.021248 | 4 | True | [folds](results/advanced_7_4_realmlp_classifier_weather_forecast/20261120-044626/paired_vs_base.csv) |
| 7.4 | severity | nested | 6h | realmlp_classifier |  | weather_forecast |  | -0.025985 | 5 | True | [folds](results/advanced_7_4_realmlp_classifier_weather_forecast/20261120-044626/paired_vs_base.csv) |
| 7.5 | delay | nested | 0min | sequence_gru |  | railway |  | +0.051973 | 2 | False | [folds](results/advanced_7_5_sequence_gru/20261114-210059/paired_vs_base.csv) |
| 7.5 | delay | nested | 1h | sequence_gru |  | weather_forecast |  | +0.081956 | 2 | False | [folds](results/advanced_7_5_sequence_gru_weather_forecast/20261120-062723/paired_vs_base.csv) |
| 7.5 | delay | nested | 20min | sequence_gru |  | forecast |  | +0.005143 | 2 | False | [folds](results/advanced_7_5_sequence_gru_forecast/20261120-062652/paired_vs_base.csv) |
| 7.5 | severity | nested | 0min | sequence_gru_classifier |  | forecast |  | -0.002751 | 3 | False | [folds](results/advanced_7_5_sequence_gru_classifier_forecast/20261120-062734/paired_vs_base.csv) |
| 7.5 | severity | nested | 1h | sequence_gru_classifier |  | forecast |  | -0.001493 | 2 | False | [folds](results/advanced_7_5_sequence_gru_classifier_forecast/20261120-062734/paired_vs_base.csv) |
| 7.5 | severity | nested | 20min | sequence_gru_classifier |  | weather_forecast |  | -0.001841 | 3 | False | [folds](results/advanced_7_5_sequence_gru_classifier_weather_forecast/20261120-062756/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 0min | lightgbm |  | weather |  | -0.014614 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 1h | lightgbm |  | weather |  | -0.088032 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 20min | lightgbm |  | weather |  | -0.075066 | 5 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 24h | lightgbm |  | forecast |  | -0.107083 | 4 | True | [folds](results/advanced_8_2_lightgbm_forecast_pooled/20261120-062810/paired_vs_base.csv) |
| 8.2 | delay | exploratory | 6h | lightgbm |  | weather |  | -0.034767 | 4 | False | [folds](results/advanced_8_2_lightgbm_weather_pooled/20261114-163727/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 0min | lightgbm_classifier |  | weather |  | -0.003529 | 4 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 1h | lightgbm_classifier |  | weather_forecast |  | -0.003246 | 4 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_forecast_pooled/20261120-062921/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 20min | lightgbm_classifier |  | weather |  | -0.005264 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 24h | lightgbm_classifier |  | forecast |  | -0.004237 | 5 | True | [folds](results/advanced_8_2_lightgbm_classifier_forecast_pooled/20261120-062857/paired_vs_base.csv) |
| 8.2 | severity | exploratory | 6h | lightgbm_classifier |  | weather |  | -0.002808 | 5 | False | [folds](results/advanced_8_2_lightgbm_classifier_weather_pooled/20261114-163847/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 0min | lightgbm | stacked | railway |  | +0.023619 | 1 | False | [folds](results/advanced_8_3_lightgbm_stacked/20261114-164223/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 1h | lightgbm | init_model | railway |  | +0.018429 | 1 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 20min | lightgbm | init_model | railway |  | +0.016414 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 24h | lightgbm | init_model | railway |  | +0.023085 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | delay | exploratory | 6h | lightgbm | init_model | railway |  | +0.022009 | 2 | False | [folds](results/advanced_8_3_lightgbm_init_model/20261114-164021/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 0min | lightgbm_classifier | stacked | railway |  | +0.001886 | 0 | False | [folds](results/advanced_8_3_lightgbm_classifier_stacked/20261114-165804/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 1h | lightgbm_classifier | init_model | railway |  | +0.001445 | 1 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 20min | lightgbm_classifier | init_model | railway |  | +0.001088 | 2 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 24h | lightgbm_classifier | init_model | railway |  | +0.000088 | 3 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.3 | severity | exploratory | 6h | lightgbm_classifier | init_model | railway |  | -0.000295 | 4 | False | [folds](results/advanced_8_3_lightgbm_classifier_init_model/20261114-164755/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 0min | lightgbm | network_transfer | weather |  | -0.005895 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261114-170025/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 1h | lightgbm | network_transfer | weather_forecast |  | -0.008742 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_forecast_network_transfer/20261120-063136/paired_vs_base.csv) |
| 8.4 | delay | exploratory | 20min | lightgbm | network_transfer | weather |  | -0.009739 | 3 | False | [folds](results/advanced_8_4_lightgbm_weather_network_transfer/20261114-170025/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 0min | lightgbm_classifier | network_transfer | weather |  | +0.000019 | 3 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261114-170353/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 1h | lightgbm_classifier | network_transfer | forecast |  | -0.000916 | 4 | False | [folds](results/advanced_8_4_lightgbm_classifier_forecast_network_transfer/20261120-063244/paired_vs_base.csv) |
| 8.4 | severity | exploratory | 20min | lightgbm_classifier | network_transfer | weather |  | -0.000319 | 3 | False | [folds](results/advanced_8_4_lightgbm_classifier_weather_network_transfer/20261114-170353/paired_vs_base.csv) |

#### Step 3.2 importance: leading feature groups (permutation) and features (TreeSHAP)

| task | prediction_time | group | score_increase | feature | mean_abs_shap | details |
| --- | --- | --- | --- | --- | --- | --- |
| severity | 0min | delay | 0.048037 | route_id | 0.183159 | [folds](results/advanced_3_2_lightgbm_classifier_forecast/20261119-235451) |
| severity | 1h | route | 0.027767 | route_id | 0.153901 | [folds](results/advanced_3_2_lightgbm_classifier_forecast/20261119-235451) |
| severity | 20min | route | 0.026987 | route_id | 0.171192 | [folds](results/advanced_3_2_lightgbm_classifier_forecast/20261119-235451) |
| severity | 24h | route | 0.033212 | route_id | 0.166818 | [folds](results/advanced_3_2_lightgbm_classifier_forecast/20261119-235451) |
| severity | 6h | recent_reliability | 0.029232 | route_id | 0.109410 | [folds](results/advanced_3_2_lightgbm_classifier_forecast/20261119-235451) |
| severity | 0min | delay | 0.047600 | route_id | 0.179296 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 1h | route | 0.026975 | route_id | 0.146786 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 20min | route | 0.027343 | route_id | 0.167452 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 24h | route | 0.032388 | route_id | 0.162560 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 6h | recent_reliability | 0.030979 | route_id | 0.108209 | [folds](results/advanced_3_2_lightgbm_classifier_weather/20261114-090429) |
| severity | 0min | delay | 0.046673 | route_id | 0.178764 | [folds](results/advanced_3_2_lightgbm_classifier_weather_forecast/20261120-000149) |
| severity | 1h | route | 0.027197 | route_id | 0.149278 | [folds](results/advanced_3_2_lightgbm_classifier_weather_forecast/20261120-000149) |
| severity | 20min | route | 0.026538 | route_id | 0.166734 | [folds](results/advanced_3_2_lightgbm_classifier_weather_forecast/20261120-000149) |
| severity | 24h | route | 0.032297 | route_id | 0.165412 | [folds](results/advanced_3_2_lightgbm_classifier_weather_forecast/20261120-000149) |
| severity | 6h | recent_reliability | 0.028388 | route_id | 0.106926 | [folds](results/advanced_3_2_lightgbm_classifier_weather_forecast/20261120-000149) |
| delay | 0min | delay | 1.542232 | hour | 1.965113 | [folds](results/advanced_3_2_lightgbm_forecast/20261119-233114) |
| delay | 1h | delay | 0.785223 | hour | 2.119972 | [folds](results/advanced_3_2_lightgbm_forecast/20261119-233114) |
| delay | 20min | delay | 0.951932 | hour | 2.087963 | [folds](results/advanced_3_2_lightgbm_forecast/20261119-233114) |
| delay | 24h | calendar | 0.612717 | hour | 2.546202 | [folds](results/advanced_3_2_lightgbm_forecast/20261119-233114) |
| delay | 6h | recent_reliability | 0.724633 | hour | 1.665971 | [folds](results/advanced_3_2_lightgbm_forecast/20261119-233114) |
| delay | 0min | delay | 1.520474 | hour | 1.878026 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 1h | delay | 0.772937 | hour | 1.956902 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 20min | delay | 0.930739 | hour | 1.970728 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 24h | calendar | 0.628347 | hour | 2.583592 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 6h | recent_reliability | 0.748895 | service_mean_delay_min_28d | 1.701739 | [folds](results/advanced_3_2_lightgbm_weather/20261114-084127) |
| delay | 0min | delay | 1.511747 | hour | 1.869780 | [folds](results/advanced_3_2_lightgbm_weather_forecast/20261119-233157) |
| delay | 1h | delay | 0.777178 | hour | 1.988849 | [folds](results/advanced_3_2_lightgbm_weather_forecast/20261119-233157) |
| delay | 20min | delay | 0.926486 | hour | 2.001383 | [folds](results/advanced_3_2_lightgbm_weather_forecast/20261119-233157) |
| delay | 24h | route | 0.559674 | hour | 2.515821 | [folds](results/advanced_3_2_lightgbm_weather_forecast/20261119-233157) |
| delay | 6h | recent_reliability | 0.695765 | hour | 1.589368 | [folds](results/advanced_3_2_lightgbm_weather_forecast/20261119-233157) |

#### Step 3.5 feature groups chosen at 24 h and 0 min

| task | arm | direction | prediction_time | group | folds_selected |
| --- | --- | --- | --- | --- | --- |
| delay | railway | backward | 24h | calendar | 5 |
| delay | railway | backward | 24h | recent_reliability | 5 |
| delay | railway | backward | 24h | timetable | 5 |
| delay | railway | backward | 24h | route | 1 |
| delay | railway | backward | 0min | calendar | 5 |
| delay | railway | backward | 0min | delay | 5 |
| delay | railway | backward | 0min | recent_reliability | 5 |
| delay | railway | backward | 0min | timetable | 5 |
| delay | forecast | backward | 24h | calendar | 4 |
| delay | forecast | backward | 24h | forecast_snow | 3 |
| delay | forecast | backward | 24h | forecast_temperature | 5 |
| delay | forecast | backward | 24h | forecast_wind | 3 |
| delay | forecast | backward | 24h | recent_reliability | 5 |
| delay | forecast | backward | 24h | route | 3 |
| delay | forecast | backward | 24h | timetable | 5 |
| delay | forecast | backward | 0min | delay | 5 |
| delay | forecast | backward | 0min | forecast_snow | 4 |
| delay | forecast | backward | 0min | forecast_temperature | 4 |
| delay | forecast | backward | 0min | recent_reliability | 5 |
| delay | forecast | backward | 0min | timetable | 5 |
| delay | forecast | backward | 0min | route | 2 |
| delay | forecast | backward | 0min | calendar | 2 |
| delay | forecast | backward | 0min | forecast_wind | 1 |
| delay | forecast | forward | 24h | calendar | 4 |
| delay | forecast | forward | 24h | forecast_snow | 3 |
| delay | forecast | forward | 24h | forecast_temperature | 5 |
| delay | forecast | forward | 24h | forecast_wind | 3 |
| delay | forecast | forward | 24h | recent_reliability | 5 |
| delay | forecast | forward | 24h | route | 3 |
| delay | forecast | forward | 24h | timetable | 5 |
| delay | forecast | forward | 0min | delay | 5 |
| delay | forecast | forward | 0min | forecast_snow | 4 |
| delay | forecast | forward | 0min | forecast_temperature | 3 |
| delay | forecast | forward | 0min | recent_reliability | 5 |
| delay | forecast | forward | 0min | timetable | 5 |
| delay | forecast | forward | 0min | route | 1 |
| delay | forecast | forward | 0min | calendar | 3 |
| delay | forecast | forward | 0min | forecast_wind | 1 |
| delay | railway | forward | 24h | calendar | 5 |
| delay | railway | forward | 24h | recent_reliability | 5 |
| delay | railway | forward | 24h | timetable | 5 |
| delay | railway | forward | 24h | route | 1 |
| delay | railway | forward | 0min | calendar | 5 |
| delay | railway | forward | 0min | delay | 5 |
| delay | railway | forward | 0min | recent_reliability | 5 |
| delay | railway | forward | 0min | timetable | 5 |
| delay | weather | backward | 24h | recent_reliability | 5 |
| delay | weather | backward | 24h | route | 3 |
| delay | weather | backward | 24h | temperature | 3 |
| delay | weather | backward | 24h | timetable | 5 |
| delay | weather | backward | 24h | calendar | 3 |
| delay | weather | backward | 24h | wind | 1 |
| delay | weather | backward | 0min | delay | 5 |
| delay | weather | backward | 0min | recent_reliability | 5 |
| delay | weather | backward | 0min | temperature | 4 |
| delay | weather | backward | 0min | timetable | 5 |
| delay | weather | backward | 0min | calendar | 4 |
| delay | weather | backward | 0min | wind | 1 |
| delay | weather_forecast | backward | 24h | calendar | 4 |
| delay | weather_forecast | backward | 24h | forecast_snow | 3 |
| delay | weather_forecast | backward | 24h | forecast_temperature | 5 |
| delay | weather_forecast | backward | 24h | forecast_wind | 3 |
| delay | weather_forecast | backward | 24h | recent_reliability | 5 |
| delay | weather_forecast | backward | 24h | route | 3 |
| delay | weather_forecast | backward | 24h | timetable | 5 |
| delay | weather_forecast | backward | 0min | calendar | 4 |
| delay | weather_forecast | backward | 0min | delay | 5 |
| delay | weather_forecast | backward | 0min | forecast_snow | 3 |
| delay | weather_forecast | backward | 0min | recent_reliability | 5 |
| delay | weather_forecast | backward | 0min | timetable | 5 |
| delay | weather_forecast | backward | 0min | wind | 2 |
| delay | weather_forecast | backward | 0min | forecast_temperature | 2 |
| delay | weather_forecast | backward | 0min | temperature | 2 |
| delay | weather_forecast | backward | 0min | forecast_wind | 1 |
| delay | weather_forecast | forward | 24h | calendar | 4 |
| delay | weather_forecast | forward | 24h | forecast_snow | 3 |
| delay | weather_forecast | forward | 24h | forecast_temperature | 5 |
| delay | weather_forecast | forward | 24h | forecast_wind | 3 |
| delay | weather_forecast | forward | 24h | recent_reliability | 5 |
| delay | weather_forecast | forward | 24h | route | 3 |
| delay | weather_forecast | forward | 24h | timetable | 5 |
| delay | weather_forecast | forward | 0min | delay | 5 |
| delay | weather_forecast | forward | 0min | forecast_snow | 4 |
| delay | weather_forecast | forward | 0min | recent_reliability | 5 |
| delay | weather_forecast | forward | 0min | temperature | 3 |
| delay | weather_forecast | forward | 0min | timetable | 5 |
| delay | weather_forecast | forward | 0min | calendar | 4 |
| delay | weather_forecast | forward | 0min | wind | 1 |
| delay | weather_forecast | forward | 0min | forecast_wind | 1 |
| delay | weather_forecast | forward | 0min | forecast_temperature | 1 |
| delay | weather | forward | 24h | recent_reliability | 5 |
| delay | weather | forward | 24h | route | 2 |
| delay | weather | forward | 24h | temperature | 2 |
| delay | weather | forward | 24h | timetable | 5 |
| delay | weather | forward | 24h | calendar | 4 |
| delay | weather | forward | 24h | wind | 1 |
| delay | weather | forward | 0min | delay | 5 |
| delay | weather | forward | 0min | recent_reliability | 5 |
| delay | weather | forward | 0min | temperature | 4 |
| delay | weather | forward | 0min | timetable | 5 |
| delay | weather | forward | 0min | calendar | 4 |
| delay | weather | forward | 0min | wind | 1 |
| severity | railway | backward | 24h | calendar | 4 |
| severity | railway | backward | 24h | recent_reliability | 5 |
| severity | railway | backward | 24h | timetable | 5 |
| severity | railway | backward | 24h | route | 2 |
| severity | railway | backward | 0min | calendar | 1 |
| severity | railway | backward | 0min | delay | 5 |
| severity | railway | backward | 0min | recent_reliability | 5 |
| severity | railway | backward | 0min | timetable | 5 |
| severity | railway | backward | 0min | route | 3 |
| severity | forecast | backward | 24h | calendar | 3 |
| severity | forecast | backward | 24h | forecast_snow | 3 |
| severity | forecast | backward | 24h | forecast_temperature | 5 |
| severity | forecast | backward | 24h | recent_reliability | 5 |
| severity | forecast | backward | 24h | timetable | 5 |
| severity | forecast | backward | 24h | route | 2 |
| severity | forecast | backward | 0min | delay | 5 |
| severity | forecast | backward | 0min | forecast_temperature | 5 |
| severity | forecast | backward | 0min | recent_reliability | 5 |
| severity | forecast | backward | 0min | route | 5 |
| severity | forecast | backward | 0min | timetable | 5 |
| severity | forecast | backward | 0min | forecast_snow | 4 |
| severity | forecast | forward | 24h | calendar | 2 |
| severity | forecast | forward | 24h | forecast_snow | 3 |
| severity | forecast | forward | 24h | forecast_temperature | 5 |
| severity | forecast | forward | 24h | recent_reliability | 5 |
| severity | forecast | forward | 24h | timetable | 5 |
| severity | forecast | forward | 24h | route | 3 |
| severity | forecast | forward | 0min | calendar | 1 |
| severity | forecast | forward | 0min | delay | 5 |
| severity | forecast | forward | 0min | recent_reliability | 5 |
| severity | forecast | forward | 0min | timetable | 5 |
| severity | forecast | forward | 0min | forecast_snow | 4 |
| severity | forecast | forward | 0min | forecast_temperature | 4 |
| severity | forecast | forward | 0min | route | 4 |
| severity | railway | forward | 24h | calendar | 4 |
| severity | railway | forward | 24h | recent_reliability | 5 |
| severity | railway | forward | 24h | timetable | 5 |
| severity | railway | forward | 24h | route | 2 |
| severity | railway | forward | 0min | calendar | 1 |
| severity | railway | forward | 0min | delay | 5 |
| severity | railway | forward | 0min | recent_reliability | 5 |
| severity | railway | forward | 0min | timetable | 5 |
| severity | railway | forward | 0min | route | 3 |
| severity | weather | backward | 24h | calendar | 1 |
| severity | weather | backward | 24h | recent_reliability | 5 |
| severity | weather | backward | 24h | timetable | 5 |
| severity | weather | backward | 24h | route | 3 |
| severity | weather | backward | 24h | temperature | 4 |
| severity | weather | backward | 24h | wind | 1 |
| severity | weather | backward | 0min | delay | 5 |
| severity | weather | backward | 0min | precipitation | 2 |
| severity | weather | backward | 0min | recent_reliability | 5 |
| severity | weather | backward | 0min | route | 4 |
| severity | weather | backward | 0min | temperature | 4 |
| severity | weather | backward | 0min | timetable | 5 |
| severity | weather | backward | 0min | wind | 1 |
| severity | weather_forecast | backward | 24h | calendar | 2 |
| severity | weather_forecast | backward | 24h | forecast_snow | 4 |
| severity | weather_forecast | backward | 24h | forecast_temperature | 5 |
| severity | weather_forecast | backward | 24h | recent_reliability | 5 |
| severity | weather_forecast | backward | 24h | timetable | 5 |
| severity | weather_forecast | backward | 24h | route | 2 |
| severity | weather_forecast | backward | 24h | wind | 1 |
| severity | weather_forecast | backward | 0min | delay | 5 |
| severity | weather_forecast | backward | 0min | precipitation | 1 |
| severity | weather_forecast | backward | 0min | recent_reliability | 5 |
| severity | weather_forecast | backward | 0min | route | 4 |
| severity | weather_forecast | backward | 0min | temperature | 2 |
| severity | weather_forecast | backward | 0min | timetable | 5 |
| severity | weather_forecast | backward | 0min | forecast_snow | 4 |
| severity | weather_forecast | backward | 0min | forecast_temperature | 4 |
| severity | weather_forecast | forward | 24h | calendar | 2 |
| severity | weather_forecast | forward | 24h | forecast_snow | 3 |
| severity | weather_forecast | forward | 24h | forecast_temperature | 5 |
| severity | weather_forecast | forward | 24h | recent_reliability | 5 |
| severity | weather_forecast | forward | 24h | timetable | 5 |
| severity | weather_forecast | forward | 24h | route | 3 |
| severity | weather_forecast | forward | 24h | wind | 1 |
| severity | weather_forecast | forward | 0min | delay | 5 |
| severity | weather_forecast | forward | 0min | precipitation | 1 |
| severity | weather_forecast | forward | 0min | recent_reliability | 5 |
| severity | weather_forecast | forward | 0min | route | 4 |
| severity | weather_forecast | forward | 0min | temperature | 4 |
| severity | weather_forecast | forward | 0min | timetable | 5 |
| severity | weather_forecast | forward | 0min | forecast_snow | 4 |
| severity | weather_forecast | forward | 0min | forecast_temperature | 3 |
| severity | weather | forward | 24h | calendar | 3 |
| severity | weather | forward | 24h | recent_reliability | 5 |
| severity | weather | forward | 24h | timetable | 5 |
| severity | weather | forward | 24h | route | 2 |
| severity | weather | forward | 24h | temperature | 2 |
| severity | weather | forward | 24h | wind | 1 |
| severity | weather | forward | 0min | delay | 5 |
| severity | weather | forward | 0min | precipitation | 2 |
| severity | weather | forward | 0min | recent_reliability | 5 |
| severity | weather | forward | 0min | route | 4 |
| severity | weather | forward | 0min | temperature | 4 |
| severity | weather | forward | 0min | timetable | 5 |
| severity | weather | forward | 0min | wind | 1 |

<!-- END ADVANCED CV -->
