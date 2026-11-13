# Advanced modelling plan: Frankfurt (Main) Hbf → München Hbf

Drafted 4 November 2026. Decisions D1–D13 made on 4 November 2026, D14–D16 on 12 November 2026; revised the same day after the review of PR #11. Updated 12 November 2026. **On 12 November 2026 the group decided to correct the dataset and the inputs, rerun every CV arm and evaluate newly registered finalists on the hold-out set (D14–D16, Phase 10). Every result below is from before these corrections.** The PR review corrections and affected CV reruns were completed on 7 November, and the original hold-out evaluation is preserved at commit 065b077. These results were merged with PR #12 on 8 November 2026. The CV results include the fixes from its review; the original hold-out results predate them (see the log).

**Two versions.** This file and `plan.html` (published at https://claude.ai/artifact/KAso5U9ubq5rFxMy7juUna) hold the same plan. Whenever the plan changes or a step is completed, update both: the experiment's status and result line here, and the matching entry in the data lists at the bottom of `plan.html`, then republish the artifact.

Status values: `todo`, `running`, `done`, `dropped`. Cost: S (under 1 CPU hour), M (1–10 h), L (over 10 h or a new dataset). Expected gain is our guess before running, so that we can check it afterwards.

## Where we are

From `../modeling/README.md` (31 October 2026), CV set, 5 week folds:

| Task | Best so far | 24 h | 0 min | Reference |
|---|---|---|---|---|
| Severity, log loss | XGBoost / LightGBM (tie) | 0.918 | 0.759 | class shares 0.968 / 0.897 |
| Delay, RMSE (min) | LightGBM (tree models tie) | 24.41 | 21.55 | training mean 25.43 |

- These scores use `recorded_route`, `observed_stops` and `unobserved_sequence_positions`, which can depend on the outcome. Step 0.7 removes them, so the reference scores will change.
- Gains over logistic and linear regression are small before departure (log loss −0.008) and larger once `current_delay` is known (−0.019).
- Weather has not helped any model so far.
- About 60% of the squared delay error comes from journeys 60+ min late, which every model predicts far too low.
- Before departure the features are only timetable and calendar; the **recent-reliability features in `ProjectInstructions.md` are not built yet** (`../README.md`, "Not built yet").

So the plan puts new information (features) before new model families: with about ten features, a better learner has little left to find.

## Decisions

Where an experiment or phase depends on a decision, it names it (for example D2).

| # | Question | Decision |
|---|---|---|
| D1 | Order of the phases | Phase 0 → 1 → 2 → 3, then 6.1 and 5, then 4, 7, 8; 9 last. Within Phase 0, 0.7 comes first. |
| D2 | How to screen many variants cheaply | Reuse the configuration chosen per prediction time and re-tune only finalists. Because the reused configurations were chosen on all five folds, screening results are exploratory (rule 4). |
| D3 | When the CV folds choose something (configuration, feature subset, ensemble weights, stacking, early stopping) | Nested: inner week folds inside each outer training set. Flat only for a handful of pre-listed comparisons, which are then exploratory. |
| D4 | One model per prediction time, or one pooled model | Test pooling in 8.2; one model per prediction time stays the default. |
| D5 | New dependencies | All approved: catboost, torch, pytabkit (RealMLP, TabM), tabpfn, tabicl, interpret (EBM), ngboost, quantile-forest, lightgbmlss. The main learners are pinned in `requirements.txt`. The distributional learners have a separate `requirements-distributions.txt` because their dependency constraints differ. tabpfn and tabicl are no longer needed (D11). |
| D6 | Compute | No foundation model is run or fine-tuned on this laptop. Simple MLPs run on it. |
| D7 | `train_number` as a feature | Left out. 1.3 encodes only the route; 6.4 is dropped. 2.1 uses the scheduled service number only to look up past journeys, as the weather branch already does. |
| D8 | Recorded-route fields | Replaced: they leave the base set in 0.7, planned versions come in 2.2, and the old fields are used only for a leakage diagnostic. |
| D9 | Extra reported metrics | None. 0.5 is dropped; the distributional models in 6.5 are scored by the RMSE of their mean and, through 4.2, by the severity log loss. |
| D10 | Other corridors' data for transfer learning (8.3) | Run all three methods with railway inputs. The auxiliary build has cutoff-specific prediction rows and restricts source fits to target training weeks. |
| D11 | Where to run what this laptop cannot | Nothing is run off this laptop (no Colab, UPPMAX or external API). 8.1 is dropped; 7.5 is dropped if it does not fit. |
| D12 | Smallest gain that counts (rule 3) | 0.003 log loss and 0.1 min RMSE, two to three times the average optimism of the flat search measured in `../modeling/README.md` (0.001 and 0.04 min). |
| D13 | Budget for nested re-tuning of finalists (0.8) | 20 configurations for remaining booster and CatBoost searches, approved on 5 November to reduce runtime. Six completed 50-configuration baseline searches also supply budget-matched 20-configuration controls by reusing their fixed first 20 inner-score rows and refitting the outer models. The saved configurations link to the original searches. Final registration caps new searches at 20, and neural searches use eight. Railway/weather pairs use matching budgets. |
| D14 | Corrections after the project review of 10 November | Full rebuild (finishing plan Q1, Q3, Q4, Q6), decided 12 November 2026. The network state selects change rows by the IDs of ICE/IC plan rows. The timetable is treated as static after 3 November 2025 as well. The dataset build is corrected for the four selection effects R5–R8. Every input is rebuilt and all 366 CV arms are rerun. |
| D15 | Hold-out set after the corrections | Decided 12 November 2026 (finishing plan Q2, Q7). New finalists are registered from the corrected CV results only, by the rules of 9.1, and scored once on the 23 hold-out weeks, together with class shares and the training mean as references. The report gives this evaluation. The evaluation of 6 November (Phase 9) stays in the git history at commit 065b077, and its result files move to results/advanced_holdout_first/. |
| D16 | Scope | Frozen for now (finishing plan Q8, 12 November 2026): no new model families or feature groups. |

## Rules for every experiment

1. **Same evaluation as now:** CV set only, the 5 committed week folds, log loss for severity and RMSE for delay as the choosing metrics, macro-F1 reported. The hold-out set was opened once, in Phase 9. By D15 it is scored once more in Phase 10, for finalists registered from the corrected CV results; nothing is chosen on it.
2. **Inputs known at the cutoff:** every live feature must be known at `predicted_at`. By the user decision on 5 November, legacy timetable fields are accepted regardless of fetch time. The earliest legacy plan per stop is treated as its static schedule. Exact-request plan updates remain cutoff-bound. State of other journeys (delays, cancellations, outcomes) is reconstructed from the live-update snapshots as of `predicted_at`, never taken from the monthly files' final values (see Phase 2). Service keys use the scheduled train, not a replacement that was announced later. Route and stop features come from the plan, not from the recorded stops. Archived DWD observations are bounded by their labeled hour. Publication latency is unmeasured.
3. **Fold comparisons are descriptive:** report the five paired fold differences, their mean, and how many folds improve. No t-tests or p-values: the folds share training weeks, so their differences are not independent (Bengio & Grandvalet 2004). A gain counts only if its mean reaches the threshold of D12, which is fixed before screening. A smaller difference is reported as no meaningful gain, not as the models being equal. Uncertainty on the hold-out set comes from the paired week bootstrap in 9.2.
4. **Exploratory and confirmed results:** a result with configurations reused from the flat search (D2), or with anything else chosen on all five folds, is exploratory, because each outer fold helped choose the configuration it is scored with. A comparison reported as a finding is nested: baseline and candidate both choose their configuration, feature subset, weights and early stopping using only the outer fold's training weeks.
5. **Weather is the research question:** every finalist is run railway-only and with weather on the same rows. 9.1 fixes matched pairs for the final comparison.
6. **Pre-list the variants** before running, so that we do not keep trying until something wins on the same folds.
7. **Cross-fitting:** a model output used as a feature (stacking in 5.4, transferred predictions in 8.3(c) and 8.4) is produced out of fold by week within each outer training set. For the outer validation rows, the upstream model is refitted on the whole outer training set.
8. **Other journeys' data** is restricted to what is known at `predicted_at`, and data used only for training to the training weeks of the fold.
9. **Results** go to `results/<name>/<timestamp>/` as now; each experiment adds its result line to this plan and its table to `README.md`.

---

## Phase 0. Infrastructure

Changes to the shared pipeline that later phases need. 0.7 comes first: every later step starts from its base set.

| ID | Step | Cost | Status |
|---|---|---|---|
| 0.7 | **Inputs known at the cutoff** (D8). Remove `recorded_route`, `observed_stops` and `unobserved_sequence_positions` from the base set. In the 216 CV departures where `drop_shared_departures()` keeps a replacement train, check which base inputs come from the replacement (`train_type`, `starts_at_origin`, the weather branch chosen by train number) and take them from the scheduled train instead. Rerun class shares, logistic and linear regression, and LightGBM and XGBoost at their saved configurations (exploratory); these become the reference scores. The removed fields are used afterwards only to measure how much they leaked | S–M | done |
| 0.1 | **Named feature sets** in `features.py` (`feature_columns(pt, feature_set="base")`), and representation options (encodings, transforms) applied inside each model's pipeline | S | done |
| 0.2 | **`compare.py`**: the five paired fold differences between two runs, their mean, folds improved, and whether the mean reaches the D12 threshold; no t-tests (rule 3). Writes the tables for `README.md` | S | done |
| 0.3 | **Inner week folds** inside each training set (grouped by ISO week, stratified like `folds.py`) for nested selection, tuning, stacking, cross-fitting, calibration and neural-network early stopping | S | done |
| 0.4 | **Fixed-configuration runner:** refit a tuned model with the configuration chosen per prediction time in an earlier run's `config.json` (D2). Its results are marked exploratory | S | done |
| 0.8 | **Nested tuning** in `tuning.py`: each outer fold chooses its configuration on the inner folds of its training weeks only (D3). Used for finalists; the budget is D13 | S to build, M–L to run | done |
| 0.5 | Extra metrics | — | dropped (D9) |
| 0.6 | **Tests** for each of the above, in the style of `tests/` (answer-as-feature, shuffled labels, no hold-out access, no outer-fold rows in nested choices) | S | done |

- **0.7.** All 26 reference and recorded-route diagnostic runs passed audit after the review corrections. Fixed-configuration comparisons remain exploratory.
- **0.1.** Base and recorded-route diagnostic sets defined. Phase 1 transformations run inside the fitted pipeline.
- **0.2.** The comparison CLI checks paired validation rows and reports the five differences, their mean, improvement count and D12 decision.
- **0.3.** Four stratified inner week folds inside each outer training set. Tests check coverage and separation.
- **0.4.** The CV runner reads matching flat-search configurations by prediction time and marks those runs exploratory.
- **0.8.** All eight base searches and eight feature-policy arms passed audit at 20 configurations. Each feature variant was screened over the same first three configurations before tuning the selected variant. The policy reached D12 in 20 of 20 LightGBM and 13 of 20 XGBoost comparisons.

- **0.6.** The reviewed main suite passed 183 tests with five skips. Four distribution tests passed separately, leaving one intentional duplicate test skipped across the environments. All 366 CV arms passed the result audit. The 336 updated arms preserved the predictions of untouched horizons exactly.

## Phase 1. Feature representations

Question: does the way a feature is encoded change the result? Screen on the base set of 0.7 with logistic/linear regression and with LightGBM at its fixed configuration (D2). These are exploratory results (rule 4); the winners are confirmed when the finalists are re-tuned nested.

Expectation: trees split on the categories natively, so they should barely react to encodings. Linear models already get a free parameter per month through one-hot encoding; sin/cos uses 2 instead of 11, which lowers variance but can only describe one smooth yearly wave. Encodings matter most for the neural networks in Phase 7, so the winners here become their defaults.

| ID | Experiment | Variants | Models | Cost | Expected gain | Status |
|---|---|---|---|---|---|---|
| 1.1 | **Cyclic calendar** | month, hour, weekday as (a) one-hot / native categories (now), (b) integers, (c) sin + cos, (d) periodic splines (`SplineTransformer(extrapolation="periodic")`), (e) one-hot + sin/cos | LR, LightGBM | S | low for trees; low–medium for LR | done |
| 1.2 | **Finer time** | departure minute of day instead of hour; day of year (cyclic) instead of month; hour × weekday interaction for LR | LR, LightGBM | S | low | done |
| 1.3 | **Route encodings** (run after 2.2) | the planned `route_id` from 2.2: one-hot / native / frequency / target encoding fitted on inner folds. `train_number` stays out (D7) | LR, LightGBM | S | low | done |
| 1.4 | **`current_delay` transforms** (1 h, 20 min, 0 min) | raw (now) / negative values clipped to 0 (open issue 1 in `../README.md`) / `log1p` of the positive part / delay bins; explicit missing indicator for trees too | LR, LightGBM | S | low–medium for LR | done |
| 1.5 | **Weather representations** | raw (now) / threshold indicators for extremes (heat over 30 °C, gust over 20 m/s, snow, heavy rain) / change over the last 24 h | LightGBM | S | low | done |

- **1.1.** All 16 calendar runs finished under the regenerated inputs. None of 80 exploratory paired comparisons reached D12. The native calendar remains the default for the neural comparisons.

- **1.2.** All ten finer-time runs finished. None of 50 exploratory paired comparisons reached D12.

- **1.4.** All 14 delay-transform runs finished. None of 42 exploratory paired comparisons reached D12.

- **1.5.** One of 20 exploratory comparisons reached D12: weather-extreme indicators reduced LightGBM severity log loss at 24 h by 0.00311, with four folds improving. No delay comparison reached the RMSE threshold.

- **1.3.** All 14 route-encoding runs finished. None of 50 exploratory paired comparisons against native planned-route encoding reached D12. Missing routes carry no target-encoded value.

## Phase 2. New features

Question: which information that is known at the prediction time is still missing? This is where we expect the largest gains, most of all at 24 h and 6 h.

**Historical state from the snapshots.** The monthly files keep only the latest changed value per stop and no cancellation issue time, so they cannot say what was known at `predicted_at`: a late update or a withdrawn cancellation would change a feature for a prediction made earlier. 2.1, 2.4 and 2.5 are therefore built from the live-update snapshots in `data/monthly_processed_data_change`:

- an event counts only if a snapshot requested at or before `predicted_at` shows it;
- a cancellation is in force at `predicted_at` only if its `cancellation_time` is at or before `predicted_at` and no snapshot up to then shows it withdrawn;
- where a snapshot is stamped only with the start of its 6-hour fetch block, its request is taken as the latest possible time (block start plus 45 min, see `../README.md`), so nothing is counted too early.

Each feature group is added to the base set on its own and then all together, with LR and fixed-configuration LightGBM. The groups that help are re-tuned for XGBoost and LightGBM, nested (0.8).

| ID | Experiment | Content | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 2.1 | **Recent reliability** (`ProjectInstructions.md` feature group; allowed to use hold-out journeys, see `STATUS.md`) | past journeys of the same scheduled service: share severe or cancelled and mean delay over the last 7 and 28 days; for the whole corridor: the same over the last 24 h. The 3 h window was removed on 13 November 2026: shorter than the archive's six-hour fetch cycle, it measured how long ago the archive fetched. Outcomes as known at `predicted_at` from the snapshots. The key is the scheduled train number: for a retained replacement, the number of the train it replaced, switching to the replacement only from a timestamped announcement in the snapshots. The number is a lookup key, not a feature (D7) | M | **high** | done |
| 2.2 | **Planned-timetable features** (D8) | `planned_stops`, a planned `route_id` and `upstream_planned_min` (time the run has been travelling before Frankfurt), all from the plan of the scheduled train; they replace the fields removed in 0.7 | M | medium | done |
| 2.3 | **Holidays** | public holidays in Hesse and Baden-Württemberg (Bavaria's are the same, so its column was removed on 13 November 2026), school holidays in all three, bridge days, day before and after a holiday | S | low–medium | done |
| 2.4 | **Delay dynamics** (1 h on) | number of upstream stops reported, planned minutes still to run before Frankfurt; all as known at `predicted_at`. The minutes since the last report and the change of `current_delay` between prediction times were removed on 13 November 2026: both mostly follow the archive's fetch times | S | medium at 1 h–0 min | done |
| 2.5 | **Network state** | share of ICE/IC trains delayed 20+ min or cancelled in the hour before `predicted_at`, as known then from the snapshots. Counted over the stations in `data/stations_present_in_all_months.csv`, so that the coverage expansion of 3 November 2025 does not shift the feature; also at Frankfurt Hbf and München Hbf. Dropped on 13 November 2026 (review of PR #13): the shares follow the archive's fetch clock, falling from 0.18 within an hour of a fetch to 0.06 five hours later. | M | medium | dropped |

- **2.3.** All four holiday arms finished. None of 20 exploratory paired comparisons reached D12.

- **2.2.** All four timetable-feature arms passed audit. One of 20 exploratory comparisons reached D12.

- **2.4.** All four delay-dynamics arms passed audit after switching to scheduled-run history before replacement announcements. Two of 12 exploratory comparisons reached D12. After the second review, replacements without a recorded replaced train use their operated run; this changed two dynamics columns in 22 CV rows, and these arms were not rerun.

- **2.1.** Both splits now use the same full target-free stop roster. Snapshot visibility still controls whether a past outcome is known. The corrected CV cache and all four model arms passed audit. Seventeen of 20 exploratory comparisons reached D12.

- **2.5.** The input audit found a temporary archive path that had omitted all plans. The builder now requires the project archive. The corrected CV build used 5,336,631 plan rows and had network-wide shares for 61,064 of 85,901 predictions. None of the 20 corrected exploratory paired comparisons reached D12.

- **Combined groups.** All four combined-feature arms passed audit with the revised reliability and replacement history. Nineteen of 20 exploratory comparisons reached D12.

## Phase 3. Feature subsets and importance

Question: does leaving features out help, and which features carry the prediction? With only 5 folds, choosing among many subsets on the same folds overfits them, so selection works on **groups** and is nested (D3).

Groups: timetable, calendar, route, `current_delay` (+ dynamics), recent reliability, network state, and weather split into temperature, precipitation, wind and gust, and snow.

| ID | Experiment | Method | Cost | Status |
|---|---|---|---|---|
| 3.1 | **Group ablation** | drop one group at a time from the full set, and add one group at a time to a timetable-only set; LR and fixed-configuration LightGBM (exploratory) | S | done |
| 3.2 | **Importance** | permutation importance by group on the validation folds; SHAP values from LightGBM (`pred_contrib=True`, no new package) | S | done |
| 3.3 | **Nested group selection** | forward and backward selection over groups, chosen on inner folds, scored on the outer fold; compare with the full set. Also L1 logistic regression as an embedded selection | M | done |
| 3.4 | **Weather subsets** | pre-listed: temperature only, wind and gust only, snow only, extremes only (from 1.5). Weather hurt slightly as a whole; a small subset may add signal without noise | S | done |
| 3.5 | **Per prediction time** | whether the best groups differ between 24 h and 0 min (at 0 min `current_delay` may make others redundant) | S | done |

- **3.1.** All 84 ablation arms were audited after correcting the network inputs. Add-one screens reached D12 in 104 of 172 comparisons against timetable-only inputs. None of 192 drop-one comparisons reached D12 against the full groups. These comparisons are exploratory.

- **3.2.** The corrected importance fits exactly reproduced 82,755 delay and 85,901 severity predictions from the full-group ablation. Permutation importance favored calendar at 24 h, reliability at 6 h, and delay near departure. Hour led mean absolute TreeSHAP in eight of ten task and horizon panels.

- **3.3.** All 14 group-selection arms passed audit. L1 logistic regression reached D12 in seven of ten comparisons against its full-group counterpart. Neither forward nor backward selection reached D12 in its 20 comparisons.

- **3.4.** All ten corrected weather-subset arms finished. None of 40 exploratory paired comparisons reached D12 against the full-weather feature set.

- **3.5.** Across 40 outer-fold selections per horizon, timetable and recent reliability were retained in all 40 at 24 h and at departure. Delay was retained in all 40 departure selections. Network state rose from 4 to 32 selections and route from 12 to 24. Calendar was retained in 27 at each of those horizons.

## Phase 4. Two-stage and decomposed models

Question: does splitting the problem into parts that are easier to learn help? Background: `../../literature_review/classification_models.md`, "Possible extension: two-stage model".

| ID | Experiment | Method | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 4.1 | **Cancellation, then delay class** | binary model for P(cancelled) on all journeys, 3-class model on completed journeys; P(class k) = (1 − P(cancelled)) · P(k \| completed). Compare with the direct 4-class model by the same log loss | M | low–medium | done |
| 4.2 | **Cancellation, then delay distribution** | as 4.1, but stage two is a predicted delay distribution (from 6.5); the class probabilities follow from it at 20 and 60 min. The severity and delay tasks then come from one model | M | medium | done |
| 4.3 | **Ordinal stage two** | the 3 delay classes as two binary models, P(delay ≥ 20) and P(delay ≥ 60), forced to be ordered | S | low | done |
| 4.4 | **Mixture of experts for delay** | a classifier routes each journey; one regressor per delay class; predicted delay = Σ P(k) · E[delay \| k]. Aimed at the 60+ min journeys that dominate RMSE | M | medium for RMSE | done |
| 4.5 | **Predict the change, not the level** (1 h on) | target = final delay − `current_delay`; prediction = `current_delay` + predicted change. Where `current_delay` is missing (11,068 of 17,114 CV rows at 1 h), the prediction comes from the level model, fitted on the same training rows. The combined predictor is scored on the same completed journeys as the baseline, so the comparison stays paired. Trees cannot predict beyond the training range, so this helps most for large current delays | S | medium for RMSE | done |

- **4.1.** Both cancellation-then-class arms passed audit. None of ten exploratory comparisons reached D12.

- **4.3.** Both ordinal arms passed audit. None of ten exploratory comparisons reached D12.

- **4.4.** Both mixture-of-experts arms passed audit. None of ten exploratory comparisons reached D12.

- **4.5.** Both change-then-level arms passed audit. None of six exploratory comparisons reached D12.

- **4.2.** All six distribution-to-severity arms passed audit in the separate pinned environment. None of 30 exploratory comparisons reached D12.

## Phase 5. Ensembles

Question: do combinations of models beat the best single model? Boosters that tie often make different errors; neural networks and trees combine especially well (TabArena, Erickson et al. 2025).

| ID | Experiment | Method | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 5.1 | **Plain average** | average the probabilities (arithmetic and geometric mean) or delays of XGBoost + LightGBM, then adding LR, CatBoost (6.1) and the neural networks (Phase 7). No weights are fitted | S | low–medium | done |
| 5.2 | **Seed and configuration bagging** | each booster with 5–10 seeds; the average of the top 5 configurations of a search. With the existing `search_results.csv` this is exploratory (chosen on all folds); for a reported result the top 5 are chosen on inner folds | S–M | low | done |
| 5.3 | **Weighted ensemble** | weights by greedy ensemble selection (Caruana et al. 2004), chosen on inner folds | M | low–medium | done |
| 5.4 | **Stacking** | logistic regression (severity) or linear regression (delay) on cross-fitted predictions of the base models (rule 7) | M | low–medium | done |

- **5.1.** All 18 fixed averages passed audit. None of 90 exploratory comparisons reached D12 against the strongest included single model.

- **5.2.** All 24 seed and configuration averages passed audit. Four of 40 nested and four of 80 exploratory comparisons reached D12.

- **5.3.** All four greedy-weight arms finished. None of 20 exploratory comparisons reached D12 against the strongest included single model at each horizon.

- **5.4.** All four cross-fitted stacking arms finished. None of 20 exploratory comparisons reached D12 against the strongest included single model at each horizon.

## Phase 6. Further model families

Question: do other learners do better than XGBoost and LightGBM, or add to an ensemble?

| ID | Experiment | Why | Cost | Status |
|---|---|---|---|---|
| 6.1 | **CatBoost** classifier and regressor, 20-configuration nested search (revised D13) | ordered target statistics for categoricals; often the strongest booster on small tabular data and among the best tuned GBDTs in TabArena | M | done |
| 6.2 | **Calibration** of every finalist: temperature scaling and isotonic regression, fitted on inner folds, with calibration plots | log loss judges the probabilities; the random forest is expected to be too cautious (`classification_models.md`) | S | done |
| 6.3 | **Explainable boosting machine** (`interpret`) | a GAM with pairwise interactions; close to boosting in accuracy on many datasets and readable per feature, useful for the report | M | done |
| 6.4 | Random effects for train number (GPBoost) | would use `train_number` as a grouping | — | dropped (D7) |
| 6.5 | **Distributional delay models**: NGBoost, quantile regression forest, LightGBMLSS (mixture) | already surveyed in `regression_models.md`; scored by the RMSE of the mean and, through 4.2, by the severity log loss (D9); feeds 4.2 | M | done |
| 6.6 | **Other losses for delay**: Huber, log-shifted target | robust to the long tail; RMSE may get worse, so this is a check rather than a likely gain | S | done |

- **6.1.** All four CatBoost arms finished at the 20-configuration budget. None of the 20 paired comparisons against the matched LightGBM baselines met D12.

- **6.5.** All six distributional arms passed audit in the separate pinned environment. None of 30 comparisons reached D12 against the nested LightGBM baseline. The mixture result describes the specified LightGBMLSS adapter.

- **6.3.** All four prespecified EBM arms passed audit. None of 20 comparisons reached D12 against the nested LightGBM baseline. The library's multiclass global visualization omits fitted pairwise terms.

- **6.6.** Both losses in both feature arms passed audit. None of 20 exploratory comparisons reached D12 against the squared-error baseline.

- **6.2.** All 12 cross-fitted isotonic and temperature calibration arms finished. None of the 60 comparisons against the corresponding uncalibrated classifier met D12.

## Phase 7. Neural networks

Question: does a neural network match or beat the boosters here, and does it add to an ensemble? Grinsztajn et al. (2022) found trees usually ahead on data of this size; recent tabular networks (RealMLP, TabM) close the gap on benchmarks. Inputs: standardised numeric columns, the best calendar encoding from 1.1, embeddings or one-hot for categoricals, `current_delay` as 0 plus a missing indicator. Early stopping and architecture search use inner week folds (0.3), never the outer validation fold. These run on this laptop (D6).

| ID | Experiment | Method | Cost | Status |
|---|---|---|---|---|
| 7.1 | **MLP baseline** (PyTorch) | 2–4 layers, dropout, weight decay, softmax for severity and squared error for delay; small random search. This is the reference for 7.3 and 7.4 | M | done |
| 7.2 | **Multi-task MLP** | one shared body with two heads (4 classes and delay), trained together: the two targets describe the same arrival | M | done |
| 7.3 | **TabGNS** (Orucu et al. 2026, arXiv:2602.10854) | gradient-based architecture search for MLPs: a gate per neuron chooses each layer's width (up to 5 layers of 512). Code: github.com/EricssonResearch/tabgns (MIT, 2 commits). Needs adapting to our log loss and folds; its gates are trained on a validation split, which must be an inner fold. Compare with 7.1 at the same compute | M | done |
| 7.4 | **RealMLP and TabM** (`pytabkit`) | the two strongest tabular networks in TabArena v0.1; scikit-learn interface | M | done |
| 7.5 | **Sequence model** over the upstream stops (1 h on) | GRU or small Transformer over the stops of the run before Frankfurt that were reported by `predicted_at` (`stops.parquet`, `upstream_runs.parquet`); stage 3 of the roadmap. Probably fits on this laptop; check before running, and drop it if it does not (D11) | L | done |

**About TabGNS.** The paper compares only with other architecture-search methods (TabNAS, AgE) and a large fixed MLP, on datasets of 31,000 to 940,000 rows. Its main results are smaller networks (51–82% fewer parameters) and faster search, with accuracy about equal to a large MLP. It has no comparison with gradient-boosted trees. So we should expect TabGNS to find a compact MLP about as good as 7.1, not to beat the boosters. The comparison is still worth making, and the method is a reasonable "advanced neural model" for the report.

- **7.1.** All four eight-configuration native-calendar MLP arms finished. None of 20 paired means reached D12 against the nested LightGBM baselines. The measured search and refit CPU times supply the TabGNS budgets.

- **7.2.** All four native-calendar joint-target arms passed audit. Seven of 20 nested comparisons reached D12 against the MLP.

- **7.3.** All four native-calendar arms passed audit after deterministic checkpoint scoring was added. Thirteen of 20 nested comparisons reached D12 against the MLP. Training uses hard Gumbel gates, checkpoint selection uses the extraction mask, and inner validation selects dense refit epochs before the from-scratch full-training refit.

- **7.4.** All eight RealMLP and TabM arms passed audit with a 64-epoch cap and week-held validation. Twenty of 40 comparisons reached D12 against the MLP. The original holdout used models with full-training refit support.

- **7.5.** All four GRU arms passed audit with eight configurations at the three late horizons. One of 12 nested comparisons reached D12 against the MLP.

## Phase 8. Transfer learning and pretraining

Question: can we learn from data beyond our 17,507 CV journeys? Ordered by expected value for cost.

| ID | Experiment | What is transferred | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 8.1 | Tabular foundation models: TabPFN, TabICL | networks pretrained on millions of synthetic tables, which predict our rows from the training rows given as context. Not run on this laptop (D6), and nothing runs elsewhere (D11) | — | medium | dropped (D6, D11) |
| 8.2 | **Pooling the prediction times** (D4) | one model on all five prediction times (about 86,000 rows), with the prediction time as a feature; the 24 h model learns from the 0 min rows' calendar effects and vice versa. Rows of one journey stay in one fold | S | low–medium | done |
| 8.3 | **Other corridors** (D10) | Hamburg Hbf → München (26,526 journeys), Berlin Hbf → München (27,412) and Hamburg Hbf → Frankfurt (21,557) from `corridor_analysis/`. (a) pooled training with a corridor feature; (b) pretrain on the other corridors, then continue training on ours (LightGBM `init_model`, or fine-tune the network from 7.1); (c) the other-corridor model's prediction as a feature, cross-fitted (rule 7). Needs prediction rows (`current_delay`) for those corridors. Runs can belong to several corridors and share disruptions on the same days, so other-corridor journeys are used only from the training weeks of each fold | L | medium | done |
| 8.4 | **Network-wide delay propagation** | pretrain a model on all ICE/IC runs in the monthly files: from the delay at some stop and the planned minutes and stops still to run, predict the delay at a later stop. Its prediction for our journey becomes a feature (1 h on), cross-fitted (rule 7), or it initialises the network of 7.5. `arrival_delays/` has only delays and train type, so this needs a new extraction from `data/monthly_processed_data`. Same week restriction as 8.3 | L | medium at 1 h–0 min | done |
| 8.5 | Self-supervised pretraining (masked-feature reconstruction, SCARF) | every row is labelled and there are about ten features, so there is little to learn without labels | — | low | dropped |

- **8.4.** All four cross-fitted late-horizon arms passed audit using the furthest reported arrival for inference progress. The source cache has 783,935 pairs with CV-only week fields and training-week and validation-run exclusions. None of 12 exploratory comparisons reached D12.

- **8.3.** All three railway-only transfer methods finished for both tasks on the three source corridors. The source cache contains 265,684 prediction rows. Fits used eligible training weeks and excluded validation run IDs. None of 30 exploratory comparisons reached D12.

- **8.2.** All four pooling arms passed audit. Six of 20 exploratory comparisons reached D12.

## Phase 9. Original final evaluation and analysis

| ID | Step | Status |
|---|---|---|
| 9.1 | **Pre-register finalists:** at most three per task and prediction time, with their configurations, feature sets and tuning policy, written into this plan before the hold-out set is opened. For each reported weather comparison, register a matched pair: the same railway feature groups, model family and tuning policy, with and without weather. If the two arms are tuned separately, the result is reported as a comparison of tuned pipelines | done |
| 9.2 | **Hold-out evaluation:** refit on all 95 CV weeks, score once on the 23 hold-out weeks; uncertainty by a paired bootstrap of whole weeks | done |
| 9.3 | **Railway only vs with weather** with the matched pairs of 9.1, on the hold-out set: the project's research question | done |
| 9.4 | **Error analysis:** by class, by period (before / from October 2025), by branch (Nürnberg / Stuttgart), on extreme-weather days; calibration plots; SHAP explanations of the final model | done |
| 9.5 | **Risk over the prediction times:** how a journey's predicted probabilities change from 24 h to 0 min, with example journeys (the "how risk updates during travel" focus) | done |

- **9.1.** Original evaluation. The replacement registry fixes one pair per task and horizon after correcting the network input. LightGBM is used for all severity horizons and for delay at 24 h and departure. Full-group linear regression is used for delay at 6 h, 1 h, and 20 min. Both arms share railway features and tuning budgets, and the registry freezes the development source and inputs.

- **9.2.** Original evaluation. All 20 registered arms refit on the 95 CV weeks and scored once on the 23 hold-out weeks. The two arms matched on 21,260 severity prediction rows and 20,450 delay prediction rows across horizons. All scores reproduced from saved predictions. Intervals use 2,000 paired whole-week resamples.

- **9.3.** Original evaluation. All ten paired 95% intervals include zero. Weather increased delay RMSE by 0.017 to 0.193 min across horizons. Severity log loss was nearly unchanged at departure, with the largest change at 24 h, an increase of 0.00486.

- **9.4.** Original evaluation. Saved paired errors by class, period, branch and extreme weather, plus calibration and SHAP summaries. At departure, weather raised severe-delay log loss by 0.02732 and severe-delay RMSE by 0.490 min. Rare extreme-weather groups and high-risk calibration bins are sparse.

- **9.5.** Original evaluation. Saved one predeparture risk trajectory per realized class with all five horizons. In the severe example, the railway severe-or-cancelled probability went from 10.4% at 24 h to 18.5% at 6 h and 10.1% at departure.

## Phase 10. Corrected rebuild and second evaluation

*What do the corrected dataset and inputs give, in CV and on the hold-out weeks?*

Decided on 12 November 2026 (D14–D16), after the project review of 10 November. Owners and dates for these steps are in FINISHING_PLAN.md at the repository root (B2, B3, B7, B11, B4, B6, C1–C3).

The week split stays as committed, so the hold-out weeks are the same 23 weeks. They were scored once in Phase 9. The finalists of 10.5 are chosen from the corrected CV results only; nothing is chosen on the hold-out set.

| ID | Step | Cost | Status |
|---|---|---|---|
| 10.1 | **Correct the dataset build** (D14): In build_dataset.py: for a kept replacement train, train_type and starts_at_origin come from the scheduled train, which the loader of 0.7 already does from the timetable inputs, so build_dataset.py is not changed for this (R5, decided 12 November 2026); no prediction row while a cancellation is in force, that is announced and not yet withdrawn (R6); current_delay no longer takes the Frankfurt departure itself (R7); cancellation_announced is the first issue time over all snapshots (R8). One test per fix. Rebuild the dataset; split_weeks.csv stays. | M | done |
| 10.2 | **Fix the network state and the timetable after 3 November 2025** (D14): network_state.py selects change rows by the IDs of ICE/IC plan rows instead of the change rows' own train_type, which 96% of them lack (R1). scheduled_timetable.py treats the timetable as static after 3 November 2025 too, as decided for the earlier period on 5 November (R2). A test for each. | S–M | done |
| 10.3 | **Rebuild the CV inputs** (D14): Timetable, delay history, recent reliability, network state, weather, and the corridor and propagation caches, from the corrected dataset. Record the coverage table; the network shares should be about 0.1–0.2. Check that the scheduled train_type and starts_at_origin are known for the kept replacement trains (R5). | M | done |
| 10.4 | **Rerun all CV arms** (D14): All 366 arms, audited as before. The result lines of Phases 0–8 in this plan and the tables in README.md are rewritten, with the comparisons the review asked for: neural networks and pooling against nested LightGBM, and failed arms named as failed. | L | todo |
| 10.5 | **Register new finalists** (D15): From the corrected CV results only, by the rules of 9.1: matched railway and weather pairs with the same feature groups, model family and tuning budget, at most three per task and prediction time, written into finalists.json before the hold-out inputs are built. | S | todo |
| 10.6 | **Second hold-out evaluation** (D15): Build the hold-out inputs with the corrected code, refit on the 95 CV weeks and score once on the 23 hold-out weeks. Class shares and the training mean are scored as references. Uncertainty by a paired bootstrap of whole weeks. The files of the first evaluation move to results/advanced_holdout_first/. | M | todo |
| 10.7 | **Analysis on the new predictions:** Errors by class, period and branch, calibration, and the risk over the prediction times on the journeys present at all five times (replaces the single examples of 9.5). | S | todo |

- **10.1.** Done on 12 November 2026. The cancellation rule (R6), the first issue time (R8) and current_delay without the Frankfurt departure (R7) are in build_dataset.py and snapshots.py, with tests in frankfurt_munich/tests/; the group confirmed the rules. A cancellation holds from its issue time; a withdrawal ends it from the time the first snapshot that shows it was requested. R5 needs no change in build_dataset.py: the loader already takes both attributes from the scheduled train (0.7), and 10.3 checks their coverage. Rebuilt dataset: 17,463, 17,366, 17,096, 16,975 and 16,926 CV prediction rows at 24 h, 6 h, 1 h, 20 min and 0 min (3, 14, 18, 19 and 21 fewer); journeys, folds and hold-out weeks unchanged. Of the 59 CV journeys with a withdrawn cancellation, 39 have a 0 min row. cancellation_announced changed for 3 CV journeys. current_delay at 0 min changed for 859 CV journeys and became missing for 563; the advanced models use the snapshot version, which this does not touch. The advanced inputs still belong to the old rows until 10.3.

- **10.2.** Code corrected on 12 November 2026; the inputs are rebuilt in 10.3. Network state: a change row is kept if it carries ICE or IC itself or if its stop ID belongs to an ICE/IC plan row of the same, the previous or the next month. In two sample months (March 2025 and March 2026) the network share is known for every row and averages 0.08; before it was 0.83 and 1.00, known for 82% and 53% of rows. Timetable: the earliest plan snapshot per stop is the schedule in both archive periods. The rule is the one all builders share, so the planned times in the network state, delay history, recent reliability and transfer inputs follow it too. For the 400 latest CV journeys the timetable inputs are known at every prediction time; before: 0% at 24 h and 6 h, 14–30% later.

- **10.3.** Done on 13 November 2026, in about 30 minutes; every CV input has exactly the 85,826 prediction keys of the corrected dataset. Timetable inputs known for 99.4% of rows at every prediction time (before: about 60%). Network share known for nearly every row, mean 0.11 (before: 0.83 to 1.00 in the sample months). Snapshot current_delay known for 6,654, 7,331 and 8,758 rows at 1 h, 20 min and 0 min (before: 6,046, 6,788 and 8,323); it stays sparse because live updates were fetched in blocks about six hours apart (finishing plan B1). R5: the scheduled train type is known for 206 of 230 kept replacement journeys; the other 24 get unknown. The corridor (8.3) and network-transfer (8.4) caches were rebuilt too. The hold-out inputs are not rebuilt.

## Not planned

- **Graph neural networks** (stage 3 of the roadmap): they need the state of the whole network as a graph; we have one corridor. 8.4 is the closest step towards network information.
- **Ordinal regression over all four classes:** cancellation breaks the order (see `classification_models.md`); 4.3 uses it only for the three delay classes.
- **Class weights:** they would raise macro-F1 but worsen log loss, the choosing metric (see `classification_models.md`).

## References

- Bengio, Y., Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *JMLR* 5, 1089–1105.
- Caruana, R., Niculescu-Mizil, A., Crew, G., Ksikes, A. (2004). Ensemble selection from libraries of models. *ICML*.
- Erickson, N., et al. (2025). TabArena: a living benchmark for machine learning on tabular data. arXiv:2506.16791.
- Gorishniy, Y., Kotelnikov, A., Babenko, A. (2025). TabM: advancing tabular deep learning with parameter-efficient ensembling. *ICLR*.
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 35*.
- Hollmann, N., et al. (2025). Accurate predictions on small data with a tabular foundation model. *Nature* 637. (TabPFN v2.)
- Holzmüller, D., Grinsztajn, L., Steinwart, I. (2024). Better by default: strong pre-tuned MLPs and boosted trees on tabular data. *NeurIPS 37*. (RealMLP.)
- Nori, H., Jenkins, S., Koch, P., Caruana, R. (2019). InterpretML: a unified framework for machine learning interpretability. arXiv:1909.09223. (Explainable boosting machine.)
- Orucu, A., Medhage, M., Moradi, F., Johnsson, A., Girdzijauskas, S. (2026). Automated model design using gated neuron selection in telecom. arXiv:2602.10854. NOMS 2026.
- Prior Labs (2025). TabPFN-2.5: advancing the state of the art in tabular foundation models. arXiv:2511.08667.
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 31*.
- Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. (2025). TabICL: a tabular foundation model for in-context learning on large data. *ICML*.

## Log

- 4 November 2026: plan drafted.
- 4 November 2026: decisions D1–D10 made. Revised after the review of PR #11: inputs known at the cutoff as a first step (0.7), nested tuning (0.8), exploratory vs confirmed results, descriptive fold comparisons, historical state from the snapshots, the scheduled service as key, a fallback for 4.5, pinned foundation-model checkpoints, cross-fitting of transferred features, matched weather pairs. New open decisions D11–D13.
- 4 November 2026: decisions D11–D13 made: nothing runs off this laptop, so 8.1 is dropped and its checkpoint list removed; minimum gain 0.003 log loss and 0.1 min RMSE; nested re-tuning with 50 configurations for LightGBM and XGBoost, railway-only and with weather.

- 5 November 2026: the user approved using legacy timetable fields regardless of fetch time and all three railway-only corridor-transfer methods. Earlier strict-plan outputs were removed from the primary result tree and retained locally outside the repository. All affected features and CV experiments are being rebuilt.

- 5 November 2026: the user chose to keep all phases and use 20 configurations for remaining searches. Unfinished 50-configuration searches were restarted at 20. Completed 50-configuration results are retained and their actual budgets are reported. Final registration uses at most 20 configurations, with eight for the neural searches.

- 6 November 2026: code review of PR #12 (not merged yet): 9 inline findings. Three may change results: the recent-reliability features (2.1) draw past journeys from the CV weeks only for CV rows but from all weeks for hold-out rows; for replacement trains, current_delay and the delay dynamics (2.4) are left missing in a way that depends on the outcome; and punctual past arrivals may not count in the reliability windows. The first two would also affect the hold-out scores of 9.2 and 9.3.

- 6 November 2026: the user approved one full target-free reliability roster for both splits. The review also prompted scheduled-run history before replacement announcements, a three-configuration inner feature screen, deterministic TabGNS checkpoint scoring, and fixes to stop ordering, category encoding, and Bavarian school-free dates. The timetable refactor reproduced every CV row. Repeat affected CV fits only. Preserve the original hold-out evaluation at 065b077.

- 7 November 2026: all nine PR comments addressed. The 336 affected CV arms and 1,248 affected arm-horizon outputs were rebuilt. Untouched horizons reproduced exactly, and all 366 active arms passed audit. The main suite passed 183 tests and the pinned distribution suite passed four. The original hold-out registry and results are unchanged. The reviewed CV inputs replace the default development caches, and the original inputs are archived locally in data/advanced_original_065b077.

- 7 November 2026: second code review of PR #12. Of the nine earlier findings, eight are fixed; the punctual-arrival one needed no change, because every checked CV arrival has a change record in the snapshots. Four new findings: the README did not say which review fixes the original hold-out inputs predate (now added); this status line was out of date (now corrected); 108 of the 656 CV rows that use the scheduled run before a replacement announcement have no resolved scheduled origin, so their current_delay is missing for an outcome-dependent reason (about 45 rows at 1 h to 0 min, open); and the nested search refits the three screened configurations (open, runtime only).

- 7 November 2026: the two open findings of the second review are fixed. The nested search now reuses the three screened configurations instead of fitting them again; results are unchanged, and a test checks that. The 17 CV replacements with no recorded replaced train now use their operated run, as an ordinary train would. Their runs have no plan snapshot, so current_delay stays missing; the rebuilt CV history changes only upstream_stops_reported and minutes_since_last_report in 22 rows. The 0.7 references read only current_delay from this file, so they were not rerun. Arms with the delay-dynamics group keep their results from the previous inputs. The 7 replacements whose replaced train no plan matched still have no history.

- 8 November 2026: the advanced work moved into frankfurt_munich/advanced_modeling/ so that it does not clutter the shared modelling folder: the input builders, experiment code, tests, results, finalist registry, this plan and its README. The shared cross-validation, tuning, feature and model code stays in frankfurt_munich/modeling/. No result was rerun; paths in commands and documents were updated.

- 10 November 2026: PR #12 was merged on 8 November. A review of the whole project found two defects in the inputs (the network-state shares count almost only disrupted trains, because the change rows are selected by a train type that 96% of them lack; the timetable inputs are missing for nearly all rows from 3 November 2025) and found the results for the neural networks, the pooling of prediction times and three failed arms worded too strongly in this plan. Nothing in this plan was changed or rerun. The findings and the decisions they need are in FINISHING_PLAN.md at the repository root.

- 12 November 2026: decisions D14–D16. The project review of 10 November found the network-state shares and the timetable inputs after 3 November 2025 defective and four selection effects in the dataset build. The group decided to correct all of them, rebuild every input, rerun all 366 CV arms, register new finalists from the corrected CV results and score them once on the hold-out weeks, with class shares and the training mean as references (Phase 10). The evaluation of 6 November stays in the git history at commit 065b077. The scope is frozen. Every result line above this entry is from before the corrections.

- 12 November 2026: code for 10.1 and 10.2 written and tested; nothing is rebuilt. 10.2 is done: network-state rows are selected by ICE/IC plan IDs, and every plan snapshot is treated as a static schedule in both archive periods. 10.1 is in progress: the cancellation rule, the first issue time and current_delay without the Frankfurt departure are in build_dataset.py and snapshots.py; R5 and the rebuild are open.

- 12 November 2026: R5 decided: build_dataset.py is not changed for it. The journey files keep the attributes of the train that ran; the loader takes train_type and starts_at_origin from the scheduled train through the timetable inputs, and 10.3 checks their coverage for the kept replacement trains.

- 12 November 2026: rules of 10.1 confirmed as written: a cancellation holds from its issue time until the first snapshot that shows its withdrawal was requested (R6), and current_delay never uses the departure from Frankfurt itself (R7). The dataset rebuild is what is left of 10.1.

- 12 November 2026: 10.1 done. The rules were confirmed and the dataset was rebuilt: 85,826 CV prediction rows (before 85,901) and 21,236 hold-out rows (before 21,260), with journeys, folds and hold-out weeks unchanged. The inputs in data/advanced/ were built for the old rows; 10.3 rebuilds them.

- 13 November 2026: 10.3 done: all CV inputs rebuilt from the corrected dataset; coverage table in README.md. The result lines of Phases 0–8 are still from before the corrections; 10.4 reruns them.

- 13 November 2026: the snapshot current_delay no longer uses the departure from Frankfurt itself (group decision, the same rule as R7 in the dataset). Delay history rebuilt: at 0 min 8,758 CV values (before 8,937); 1 h and 20 min unchanged.

- 13 November 2026: minutes_since_last_report removed from the delay-dynamics group (2.4) by group decision: it reflects the archive's fetch times, not the train. Delay history rebuilt; the other dynamics columns are unchanged.

- 13 November 2026: review of PR #13. Removed as features because they follow the archive's six-hourly fetch clock: the network state (2.5, dropped), the corridor reliability over 3 h (2.1) and the change of current_delay between prediction times (2.4). public_holiday_by removed (equal to public_holiday_bw); Bavaria's summer break 2025 starts on 1 August; a public holiday is no longer a bridge day; the adjacency features see every year of the span. current_delay counts waiting at a stop only while the snapshot shows the train still standing there (10 CV rows changed). Delay history and the corridor cache rebuilt.
