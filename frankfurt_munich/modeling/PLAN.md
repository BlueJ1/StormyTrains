# Advanced modelling plan: Frankfurt (Main) Hbf → München Hbf

Drafted 4 November 2026. Decisions D1–D10 made on 4 November 2026; revised the same day after the review of PR #11. **D11–D13 are open. Nothing in this plan has been run.**

**Two versions.** This file and `plan.html` (published as an artifact, link in `STATUS.md`) hold the same plan. Whenever the plan changes or a step is completed, update both: the experiment's status and result line here, and the matching entry in the data lists at the bottom of `plan.html`, then republish the artifact.

Status values: `todo`, `running`, `done`, `dropped`. Cost: S (under 1 CPU hour), M (1–10 h), L (over 10 h or a new dataset). Expected gain is our guess before running, so that we can check it afterwards.

## Where we are

From `modeling/README.md` (31 October 2026), CV set, 5 week folds:

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
| D5 | New dependencies | All approved: catboost, torch, pytabkit (RealMLP, TabM), tabpfn, tabicl, interpret (EBM), ngboost, quantile-forest, lightgbmlss. Each is installed and pinned in `requirements.txt` when its phase starts. |
| D6 | Compute | No foundation model is run or fine-tuned on this laptop. Simple MLPs run on it. Where 8.1 runs is D11. |
| D7 | `train_number` as a feature | Left out. 1.3 encodes only the route; 6.4 is dropped. 2.1 uses the scheduled service number only to look up past journeys, as the weather branch already does. |
| D8 | Recorded-route fields | Replaced: they leave the base set in 0.7, planned versions come in 2.2, and the old fields are used only for a leakage diagnostic. |
| D9 | Extra reported metrics | None. 0.5 is dropped; the distributional models in 6.5 are scored by the RMSE of their mean and, through 4.2, by the severity log loss. |
| D10 | Other corridors' data for transfer learning (8.3) | Decided after Phase 2. Needs prediction rows for those corridors. |
| D11 | **Open.** Where to run 8.1 | Google Colab / UPPMAX / the Prior Labs API (sends our rows to an external service) / skip 8.1. Suggestion: ask Qing He about UPPMAX. |
| D12 | **Open.** Smallest gain that counts (rule 3) | Suggestion: 0.003 log loss and 0.1 min RMSE, two to three times the average optimism of the flat search measured in `README.md` (0.001 and 0.04 min). |
| D13 | **Open.** Budget for nested re-tuning of finalists (0.8) | Nesting multiplies a search by about 4 (4 inner folds). Options: all 50 configurations / 20 / only LightGBM. Suggestion: 50 configurations for LightGBM and XGBoost, railway-only and with weather. |

## Rules for every experiment

1. **Same evaluation as now:** CV set only, the 5 committed week folds, log loss for severity and RMSE for delay as the choosing metrics, macro-F1 reported. The hold-out set is opened once, in Phase 9.
2. **Inputs known at the cutoff:** every feature must be known at `predicted_at`. State of other journeys (delays, cancellations, outcomes) is reconstructed from the live-update snapshots as of `predicted_at`, never taken from the monthly files' final values (see Phase 2). Service keys use the scheduled train, not a replacement that was announced later. Route and stop features come from the plan, not from the recorded stops.
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
| 0.7 | **Inputs known at the cutoff** (D8). Remove `recorded_route`, `observed_stops` and `unobserved_sequence_positions` from the base set. In the 216 CV departures where `drop_shared_departures()` keeps a replacement train, check which base inputs come from the replacement (`train_type`, `starts_at_origin`, the weather branch chosen by train number) and take them from the scheduled train instead. Rerun class shares, logistic and linear regression, and LightGBM and XGBoost at their saved configurations (exploratory); these become the reference scores. The removed fields are used afterwards only to measure how much they leaked | S–M | todo |
| 0.1 | **Named feature sets** in `features.py` (`feature_columns(pt, feature_set="base")`), and representation options (encodings, transforms) applied inside each model's pipeline | S | todo |
| 0.2 | **`compare.py`**: the five paired fold differences between two runs, their mean, folds improved, and whether the mean reaches the D12 threshold; no t-tests (rule 3). Writes the tables for `README.md` | S | todo |
| 0.3 | **Inner week folds** inside each training set (grouped by ISO week, stratified like `folds.py`) for nested selection, tuning, stacking, cross-fitting, calibration and neural-network early stopping | S | todo |
| 0.4 | **Fixed-configuration runner:** refit a tuned model with the configuration chosen per prediction time in an earlier run's `config.json` (D2). Its results are marked exploratory | S | todo |
| 0.8 | **Nested tuning** in `tuning.py`: each outer fold chooses its configuration on the inner folds of its training weeks only (D3). Used for finalists; the budget is D13 | S to build, M–L to run | todo |
| 0.5 | Extra metrics | — | dropped (D9) |
| 0.6 | **Tests** for each of the above, in the style of `tests/` (answer-as-feature, shuffled labels, no hold-out access, no outer-fold rows in nested choices) | S | todo |

## Phase 1. Feature representations

Question: does the way a feature is encoded change the result? Screen on the base set of 0.7 with logistic/linear regression and with LightGBM at its fixed configuration (D2). These are exploratory results (rule 4); the winners are confirmed when the finalists are re-tuned nested.

Expectation: trees split on the categories natively, so they should barely react to encodings. Linear models already get a free parameter per month through one-hot encoding; sin/cos uses 2 instead of 11, which lowers variance but can only describe one smooth yearly wave. Encodings matter most for the neural networks in Phase 7, so the winners here become their defaults.

| ID | Experiment | Variants | Models | Cost | Expected gain | Status |
|---|---|---|---|---|---|---|
| 1.1 | **Cyclic calendar** | month, hour, weekday as (a) one-hot / native categories (now), (b) integers, (c) sin + cos, (d) periodic splines (`SplineTransformer(extrapolation="periodic")`), (e) one-hot + sin/cos | LR, LightGBM | S | low for trees; low–medium for LR | todo |
| 1.2 | **Finer time** | departure minute of day instead of hour; day of year (cyclic) instead of month; hour × weekday interaction for LR | LR, LightGBM | S | low | todo |
| 1.3 | **Route encodings** (run after 2.2) | the planned `route_id` from 2.2: one-hot / native / frequency / target encoding fitted on inner folds. `train_number` stays out (D7) | LR, LightGBM | S | low | todo |
| 1.4 | **`current_delay` transforms** (1 h, 20 min, 0 min) | raw (now) / negative values clipped to 0 (open issue 1 in `../README.md`) / `log1p` of the positive part / delay bins; explicit missing indicator for trees too | LR, LightGBM | S | low–medium for LR | todo |
| 1.5 | **Weather representations** | raw (now) / threshold indicators for extremes (heat over 30 °C, gust over 20 m/s, snow, heavy rain) / change over the last 24 h | LightGBM | S | low | todo |

## Phase 2. New features

Question: which information that is known at the prediction time is still missing? This is where we expect the largest gains, most of all at 24 h and 6 h.

**Historical state from the snapshots.** The monthly files keep only the latest changed value per stop and no cancellation issue time, so they cannot say what was known at `predicted_at`: a late update or a withdrawn cancellation would change a feature for a prediction made earlier. 2.1, 2.4 and 2.5 are therefore built from the live-update snapshots in `data/monthly_processed_data_change`:

- an event counts only if a snapshot requested at or before `predicted_at` shows it;
- a cancellation is in force at `predicted_at` only if its `cancellation_time` is at or before `predicted_at` and no snapshot up to then shows it withdrawn;
- where a snapshot is stamped only with the start of its 6-hour fetch block, its request is taken as the latest possible time (block start plus 45 min, see `../README.md`), so nothing is counted too early.

Each feature group is added to the base set on its own and then all together, with LR and fixed-configuration LightGBM. The groups that help are re-tuned for XGBoost and LightGBM, nested (0.8).

| ID | Experiment | Content | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 2.1 | **Recent reliability** (`ProjectInstructions.md` feature group; allowed to use hold-out journeys, see `STATUS.md`) | past journeys of the same scheduled service: share severe or cancelled and mean delay over the last 7 and 28 days; for the whole corridor: the same over the last 24 h and last 3 h. Outcomes as known at `predicted_at` from the snapshots. The key is the scheduled train number: for a retained replacement, the number of the train it replaced, switching to the replacement only from a timestamped announcement in the snapshots. The number is a lookup key, not a feature (D7) | M | **high** | todo |
| 2.2 | **Planned-timetable features** (D8) | `planned_stops`, a planned `route_id` and `upstream_planned_min` (time the run has been travelling before Frankfurt), all from the plan of the scheduled train; they replace the fields removed in 0.7 | M | medium | todo |
| 2.3 | **Holidays** | public holidays in Hesse, Baden-Württemberg and Bavaria, school holidays, bridge days, day before and after a holiday | S | low–medium | todo |
| 2.4 | **Delay dynamics** (1 h on) | change of `current_delay` between prediction times, number of upstream stops reported, minutes since the last report, planned minutes still to run before Frankfurt; all as known at `predicted_at` | S | medium at 1 h–0 min | todo |
| 2.5 | **Network state** | share of ICE/IC trains delayed 20+ min or cancelled in the hour before `predicted_at`, as known then from the snapshots. Counted over the stations in `data/stations_present_in_all_months.csv`, so that the coverage expansion of 3 November 2025 does not shift the feature; also at Frankfurt Hbf and München Hbf | M | medium | todo |

## Phase 3. Feature subsets and importance

Question: does leaving features out help, and which features carry the prediction? With only 5 folds, choosing among many subsets on the same folds overfits them, so selection works on **groups** and is nested (D3).

Groups: timetable, calendar, route, `current_delay` (+ dynamics), recent reliability, network state, and weather split into temperature, precipitation, wind and gust, and snow.

| ID | Experiment | Method | Cost | Status |
|---|---|---|---|---|
| 3.1 | **Group ablation** | drop one group at a time from the full set, and add one group at a time to a timetable-only set; LR and fixed-configuration LightGBM (exploratory) | S | todo |
| 3.2 | **Importance** | permutation importance by group on the validation folds; SHAP values from LightGBM (`pred_contrib=True`, no new package) | S | todo |
| 3.3 | **Nested group selection** | forward and backward selection over groups, chosen on inner folds, scored on the outer fold; compare with the full set. Also L1 logistic regression as an embedded selection | M | todo |
| 3.4 | **Weather subsets** | pre-listed: temperature only, wind and gust only, snow only, extremes only (from 1.5). Weather hurt slightly as a whole; a small subset may add signal without noise | S | todo |
| 3.5 | **Per prediction time** | whether the best groups differ between 24 h and 0 min (at 0 min `current_delay` may make others redundant) | S | todo |

## Phase 4. Two-stage and decomposed models

Question: does splitting the problem into parts that are easier to learn help? Background: `../../literature_review/classification_models.md`, "Possible extension: two-stage model".

| ID | Experiment | Method | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 4.1 | **Cancellation, then delay class** | binary model for P(cancelled) on all journeys, 3-class model on completed journeys; P(class k) = (1 − P(cancelled)) · P(k \| completed). Compare with the direct 4-class model by the same log loss | M | low–medium | todo |
| 4.2 | **Cancellation, then delay distribution** | as 4.1, but stage two is a predicted delay distribution (from 6.5); the class probabilities follow from it at 20 and 60 min. The severity and delay tasks then come from one model | M | medium | todo |
| 4.3 | **Ordinal stage two** | the 3 delay classes as two binary models, P(delay ≥ 20) and P(delay ≥ 60), forced to be ordered | S | low | todo |
| 4.4 | **Mixture of experts for delay** | a classifier routes each journey; one regressor per delay class; predicted delay = Σ P(k) · E[delay \| k]. Aimed at the 60+ min journeys that dominate RMSE | M | medium for RMSE | todo |
| 4.5 | **Predict the change, not the level** (1 h on) | target = final delay − `current_delay`; prediction = `current_delay` + predicted change. Where `current_delay` is missing (2,362 of 17,114 CV rows at 1 h), the prediction comes from the level model, fitted on the same training rows. The combined predictor is scored on the same completed journeys as the baseline, so the comparison stays paired. Trees cannot predict beyond the training range, so this helps most for large current delays | S | medium for RMSE | todo |

## Phase 5. Ensembles

Question: do combinations of models beat the best single model? Boosters that tie often make different errors; neural networks and trees combine especially well (TabArena, Erickson et al. 2025).

| ID | Experiment | Method | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 5.1 | **Plain average** | average the probabilities (arithmetic and geometric mean) or delays of XGBoost + LightGBM, then adding LR, CatBoost (6.1) and the neural networks (Phase 7). No weights are fitted | S | low–medium | todo |
| 5.2 | **Seed and configuration bagging** | each booster with 5–10 seeds; the average of the top 5 configurations of a search. With the existing `search_results.csv` this is exploratory (chosen on all folds); for a reported result the top 5 are chosen on inner folds | S–M | low | todo |
| 5.3 | **Weighted ensemble** | weights by greedy ensemble selection (Caruana et al. 2004), chosen on inner folds | M | low–medium | todo |
| 5.4 | **Stacking** | logistic regression (severity) or linear regression (delay) on cross-fitted predictions of the base models (rule 7) | M | low–medium | todo |

## Phase 6. Further model families

Question: do other learners do better than XGBoost and LightGBM, or add to an ensemble?

| ID | Experiment | Why | Cost | Status |
|---|---|---|---|---|
| 6.1 | **CatBoost** classifier and regressor, same 50-configuration search | ordered target statistics for categoricals; often the strongest booster on small tabular data and among the best tuned GBDTs in TabArena | M | todo |
| 6.2 | **Calibration** of every finalist: temperature scaling and isotonic regression, fitted on inner folds, with calibration plots | log loss judges the probabilities; the random forest is expected to be too cautious (`classification_models.md`) | S | todo |
| 6.3 | **Explainable boosting machine** (`interpret`) | a GAM with pairwise interactions; close to boosting in accuracy on many datasets and readable per feature, useful for the report | M | todo |
| 6.4 | Random effects for train number (GPBoost) | would use `train_number` as a grouping | — | dropped (D7) |
| 6.5 | **Distributional delay models**: NGBoost, quantile regression forest, LightGBMLSS (mixture) | already surveyed in `regression_models.md`; scored by the RMSE of the mean and, through 4.2, by the severity log loss (D9); feeds 4.2 | M | todo |
| 6.6 | **Other losses for delay**: Huber, log-shifted target | robust to the long tail; RMSE may get worse, so this is a check rather than a likely gain | S | todo |

## Phase 7. Neural networks

Question: does a neural network match or beat the boosters here, and does it add to an ensemble? Grinsztajn et al. (2022) found trees usually ahead on data of this size; recent tabular networks (RealMLP, TabM) close the gap on benchmarks. Inputs: standardised numeric columns, the best calendar encoding from 1.1, embeddings or one-hot for categoricals, `current_delay` as 0 plus a missing indicator. Early stopping and architecture search use inner week folds (0.3), never the outer validation fold. These run on this laptop (D6).

| ID | Experiment | Method | Cost | Status |
|---|---|---|---|---|
| 7.1 | **MLP baseline** (PyTorch) | 2–4 layers, dropout, weight decay, softmax for severity and squared error for delay; small random search. This is the reference for 7.3 and 7.4 | M | todo |
| 7.2 | **Multi-task MLP** | one shared body with two heads (4 classes and delay), trained together: the two targets describe the same arrival | M | todo |
| 7.3 | **TabGNS** (Orucu et al. 2026, arXiv:2602.10854) | gradient-based architecture search for MLPs: a gate per neuron chooses each layer's width (up to 5 layers of 512). Code: github.com/EricssonResearch/tabgns (MIT, 2 commits). Needs adapting to our log loss and folds; its gates are trained on a validation split, which must be an inner fold. Compare with 7.1 at the same compute | M | todo |
| 7.4 | **RealMLP and TabM** (`pytabkit`) | the two strongest tabular networks in TabArena v0.1; scikit-learn interface | M | todo |
| 7.5 | **Sequence model** over the upstream stops (1 h on) | GRU or small Transformer over the stops of the run before Frankfurt that were reported by `predicted_at` (`stops.parquet`, `upstream_runs.parquet`); stage 3 of the roadmap. Probably fits on this laptop; check before running | L | todo |

**About TabGNS.** The paper compares only with other architecture-search methods (TabNAS, AgE) and a large fixed MLP, on datasets of 31,000 to 940,000 rows. Its main results are smaller networks (51–82% fewer parameters) and faster search, with accuracy about equal to a large MLP. It has no comparison with gradient-boosted trees. So we should expect TabGNS to find a compact MLP about as good as 7.1, not to beat the boosters. The comparison is still worth making, and the method is a reasonable "advanced neural model" for the report.

## Phase 8. Transfer learning and pretraining

Question: can we learn from data beyond our 17,507 CV journeys? Ordered by expected value for cost.

| ID | Experiment | What is transferred | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 8.1 | **Tabular foundation models**: TabPFN, TabICL (D11) | networks pretrained on millions of synthetic tables, which predict our rows from the training rows given as context, without training on our data. Training folds of about 14,000 rows are within TabPFN-2.5's 50,000-row design. Not run or fine-tuned on this laptop (D6). Checkpoints are listed below | M–L | medium; strong on small tables in benchmarks | todo |
| 8.2 | **Pooling the prediction times** (D4) | one model on all five prediction times (about 86,000 rows), with the prediction time as a feature; the 24 h model learns from the 0 min rows' calendar effects and vice versa. Rows of one journey stay in one fold | S | low–medium | todo |
| 8.3 | **Other corridors** (D10) | Hamburg Hbf → München (26,526 journeys), Berlin Hbf → München (27,412) and Hamburg Hbf → Frankfurt (21,557) from `corridor_analysis/`. (a) pooled training with a corridor feature; (b) pretrain on the other corridors, then continue training on ours (LightGBM `init_model`, or fine-tune the network from 7.1); (c) the other-corridor model's prediction as a feature, cross-fitted (rule 7). Needs prediction rows (`current_delay`) for those corridors. Runs can belong to several corridors and share disruptions on the same days, so other-corridor journeys are used only from the training weeks of each fold | L | medium | todo |
| 8.4 | **Network-wide delay propagation** | pretrain a model on all ICE/IC runs in the monthly files: from the delay at some stop and the planned minutes and stops still to run, predict the delay at a later stop. Its prediction for our journey becomes a feature (1 h on), cross-fitted (rule 7), or it initialises the network of 7.5. `arrival_delays/` has only delays and train type, so this needs a new extraction from `data/monthly_processed_data`. Same week restriction as 8.3 | L | medium at 1 h–0 min | todo |
| 8.5 | Self-supervised pretraining (masked-feature reconstruction, SCARF) | every row is labelled and there are about ten features, so there is little to learn without labels | — | low | dropped |

**Checkpoints for 8.1**, Hugging Face revisions as of 4 November 2026. D5 covers the packages; which checkpoint we use is settled with D11, after reading the licences.

| Checkpoint | Revision | Licence |
|---|---|---|
| `Prior-Labs/TabPFN-v2-clf` | `275f19d` | Prior Labs License v1.1: Apache 2.0 with an attribution clause |
| `Prior-Labs/TabPFN-v2-reg` | `02a4470` | Prior Labs License v1.1 |
| `Prior-Labs/tabpfn_2_5` | `6c45f3a` | TabPFN-2.5 License v1.1, non-commercial according to the `tabpfn` package documentation |
| `Prior-Labs/tabpfn_2_6` | `2414887` | TabPFN-2.6 License v1.0, non-commercial (same source) |
| `Prior-Labs/tabpfn_3` | `24a16a8` | TabPFN-3 License v1.0, non-commercial (same source) |
| `jingang/TabICL-clf` | `eaf789a` | BSD-3-Clause |
| `jingang/TabICL` | `4dcd344` | BSD-3-Clause |

We have not yet read the full texts of the TabPFN-2.5, 2.6 and 3 licences; their terms and any access conditions are recorded here before one of them is used.

## Phase 9. Final evaluation and analysis

| ID | Step | Status |
|---|---|---|
| 9.1 | **Pre-register finalists:** at most three per task and prediction time, with their configurations, feature sets and tuning policy, written into this plan before the hold-out set is opened. For each reported weather comparison, register a matched pair: the same railway feature groups, model family and tuning policy, with and without weather. If the two arms are tuned separately, the result is reported as a comparison of tuned pipelines | todo |
| 9.2 | **Hold-out evaluation:** refit on all 95 CV weeks, score once on the 23 hold-out weeks; uncertainty by a paired bootstrap of whole weeks | todo |
| 9.3 | **Railway only vs with weather** with the matched pairs of 9.1, on the hold-out set: the project's research question | todo |
| 9.4 | **Error analysis:** by class, by period (before / from October 2025), by branch (Nürnberg / Stuttgart), on extreme-weather days; calibration plots; SHAP explanations of the final model | todo |
| 9.5 | **Risk over the prediction times:** how a journey's predicted probabilities change from 24 h to 0 min, with example journeys (the "how risk updates during travel" focus) | todo |

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
