# Advanced modelling plan: Frankfurt (Main) Hbf → München Hbf

Drafted 4 November 2026. Status: **proposal, not yet approved.** Nothing in this plan has been run.

**Two versions.** This file and `plan.html` (published as an artifact, link in `STATUS.md`) hold the same plan. Whenever the plan changes or a step is completed, update both: the experiment's status and result line here, and the matching entry in the `EXPERIMENTS` list in `plan.html`, then republish the artifact.

Status values: `todo`, `running`, `done`, `dropped`. Cost: S (under 1 CPU hour), M (1–10 h), L (over 10 h or a new dataset). Expected gain is our guess before running, so that we can check it afterwards.

## Where we are

From `modeling/README.md` (31 October 2026), CV set, 5 week folds:

| Task | Best so far | 24 h | 0 min | Reference |
|---|---|---|---|---|
| Severity, log loss | XGBoost / LightGBM (tie) | 0.918 | 0.759 | class shares 0.968 / 0.897 |
| Delay, RMSE (min) | LightGBM (tree models tie) | 24.41 | 21.55 | training mean 25.43 |

- Gains over logistic and linear regression are small before departure (log loss −0.008) and larger once `current_delay` is known (−0.019).
- Weather has not helped any model so far.
- About 60% of the squared delay error comes from journeys 60+ min late, which every model predicts far too low.
- Before departure the features are only timetable and calendar; the **recent-reliability features in `ProjectInstructions.md` are not built yet** (`../README.md`, "Not built yet").

So the plan puts new information (features) before new model families: with about ten features, a better learner has little left to find.

## Decisions we need from you

Nothing below should start before these are settled. Where an experiment or phase depends on a decision, it names it (for example D2).

| # | Decision | Options | Our suggestion |
|---|---|---|---|
| D1 | Order of the phases | as listed below, or other | Phase 0 → 1 → 2 → 3, then 6.1 and 5, then 4, 7, 8; 9 last |
| D2 | How to screen many variants cheaply | re-tune every variant (about 30 h per full set) / reuse the chosen configuration per prediction time and re-tune only finalists | reuse, re-tune finalists |
| D3 | When the CV folds choose something (feature subset, ensemble weights, stacking, early stopping) | flat (same folds choose and score) / nested (inner week folds inside each training set) | nested; flat is acceptable only for a handful of pre-listed comparisons |
| D4 | One model per prediction time (current) or one pooled model with the prediction time as a feature | keep / test pooling as an experiment (8.2) | test it, keep per-time as default |
| D5 | New dependencies | catboost, torch, pytabkit (RealMLP, TabM), tabpfn, tabicl, interpret (EBM), ngboost, quantile-forest, lightgbmlss | approve per phase, not all at once |
| D6 | Compute for neural and foundation models | this laptop (M1 Pro, 8 cores, 16 GB, no CUDA; PyTorch can use the Apple GPU) / Google Colab / UPPMAX / the Prior Labs API (sends our rows to an external service) | laptop for MLPs; ask Qing He about UPPMAX before Phase 8.1 |
| D7 | `train_number` as a feature (left out so far) | keep out / native categorical / target-encoded | test it in 1.3 |
| D8 | `recorded_route`, `observed_stops`, `unobserved_sequence_positions` come from the recorded stops, so they can depend on the outcome (a diverted train records other stations) | keep / replace with planned-stop versions | replace in 2.2 and compare |
| D9 | Extra reported metrics | none / Brier score, binary `severe_or_canceled` log loss and PR-AUC (the initial binary target in `ProjectInstructions.md`), calibration error; MAE for delay | add them, reported only, not used for choosing |
| D10 | Other corridors' data for transfer learning (8.3) | not used / used for training only | decide after Phase 2; needs prediction rows for those corridors |

## Rules for every experiment

1. **Same evaluation as now:** CV set only, the 5 committed week folds, log loss for severity and RMSE for delay as the choosing metrics, macro-F1 reported. The hold-out set is opened once, in Phase 9.
2. **Paired comparison:** compare two models fold by fold on the same rows. Report the mean difference, how many of the 5 folds improve, and the paired t. A difference smaller than the spread between folds is reported as no difference.
3. **Weather is the research question:** every finalist is run railway-only and with weather on the same rows (`weather=True`).
4. **Pre-list the variants** before running, so that we do not keep trying until something wins on the same folds. Where the folds choose among many variants, use nested evaluation (D3).
5. **Results** go to `results/<name>/<timestamp>/` as now; each experiment adds its result line to this plan and its table to `README.md`.
6. **Leakage:** every feature must be known at `predicted_at`. Any data from other journeys (recent reliability, other corridors, network data) is restricted to events reported before `predicted_at`, and for training-only data, to the training weeks of the fold.

---

## Phase 0. Infrastructure

Small changes to the shared pipeline that later phases need.

| ID | Step | Cost | Status |
|---|---|---|---|
| 0.1 | **Named feature sets** in `features.py` (`feature_columns(pt, feature_set="base")`), and representation options (encodings, transforms) applied inside each model's pipeline | S | todo |
| 0.2 | **`compare.py`**: paired fold differences between two runs (mean, folds improved, t), and the tables for `README.md` | S | todo |
| 0.3 | **Inner week folds** inside each training set (grouped by ISO week, stratified like `folds.py`) for nested selection, stacking, calibration and neural-network early stopping | S | todo |
| 0.4 | **Fixed-configuration runner:** refit a tuned model with the configuration chosen per prediction time in an earlier run's `config.json` (D2) | S | todo |
| 0.5 | **Extra metrics** (D9) and calibration plots from the saved `oof_predictions.parquet` | S | todo |
| 0.6 | **Tests** for each of the above, in the style of `tests/` (answer-as-feature, shuffled labels, no hold-out access) | S | todo |

## Phase 1. Feature representations

Question: does the way a feature is encoded change the result? Screen with logistic/linear regression and with LightGBM at its fixed configuration (D2).

Expectation: trees split on the categories natively, so they should barely react to encodings. Linear models already get a free parameter per month through one-hot encoding; sin/cos uses 2 instead of 11, which lowers variance but can only describe one smooth yearly wave. Encodings matter most for the neural networks in Phase 7, so the winners here become their defaults.

| ID | Experiment | Variants | Models | Cost | Expected gain | Status |
|---|---|---|---|---|---|---|
| 1.1 | **Cyclic calendar** | month, hour, weekday as (a) one-hot / native categories (now), (b) integers, (c) sin + cos, (d) periodic splines (`SplineTransformer(extrapolation="periodic")`), (e) one-hot + sin/cos | LR, LightGBM | S | low for trees; low–medium for LR | todo |
| 1.2 | **Finer time** | departure minute of day instead of hour; day of year (cyclic) instead of month; hour × weekday interaction for LR | LR, LightGBM | S | low | todo |
| 1.3 | **Categorical encodings** | `recorded_route` (61 levels): one-hot / native / frequency / target encoding fitted on inner folds; `train_number` added (D7) the same ways | LR, LightGBM | S | medium for `train_number` | todo |
| 1.4 | **`current_delay` transforms** (1 h, 20 min, 0 min) | raw (now) / negative values clipped to 0 (open issue 1 in `../README.md`) / `log1p` of the positive part / delay bins; explicit missing indicator for trees too | LR, LightGBM | S | low–medium for LR | todo |
| 1.5 | **Weather representations** | raw (now) / threshold indicators for extremes (heat over 30 °C, gust over 20 m/s, snow, heavy rain) / change over the last 24 h | LightGBM | S | low | todo |

## Phase 2. New features

Question: which information that is known at the prediction time is still missing? This is where we expect the largest gains, most of all at 24 h and 6 h.

| ID | Experiment | Content | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 2.1 | **Recent reliability** (`ProjectInstructions.md` feature group; allowed to use hold-out journeys, see `STATUS.md`) | for the same train number: share severe or cancelled and mean delay over the last 7 and 28 days; for the whole corridor: the same over the last 24 h and last 3 h; only journeys whose arrival was reported before `predicted_at` | M | **high** | todo |
| 2.2 | **Planned-timetable features** (D8) | `planned_stops`, a planned `route_id` and `upstream_planned_min` (time the run has been travelling before Frankfurt), all from the plan rather than the recorded stops; compare with the recorded versions | M | medium (and removes a possible leak) | todo |
| 2.3 | **Holidays** | public holidays in Hesse, Baden-Württemberg and Bavaria, school holidays, bridge days, day before and after a holiday | S | low–medium | todo |
| 2.4 | **Delay dynamics** (1 h on) | change of `current_delay` between prediction times, number of upstream stops reported, minutes since the last report, planned minutes still to run before Frankfurt | S | medium at 1 h–0 min | todo |
| 2.5 | **Network state** | share of all ICE/IC trains in the monthly files that are delayed 20+ min or cancelled in the hour before `predicted_at`, nationally and at Frankfurt Hbf and München Hbf | M | medium | todo |

Each feature group is added to the base set on its own and then all together, with LR and fixed-configuration LightGBM. The groups that help are re-tuned for XGBoost and LightGBM.

## Phase 3. Feature subsets and importance

Question: does leaving features out help, and which features carry the prediction? With only 5 folds, choosing among many subsets on the same folds overfits them, so selection works on **groups** and is nested (D3).

Groups: timetable, calendar, route, `current_delay` (+ dynamics), recent reliability, network state, and weather split into temperature, precipitation, wind and gust, and snow.

| ID | Experiment | Method | Cost | Status |
|---|---|---|---|---|
| 3.1 | **Group ablation** | drop one group at a time from the full set, and add one group at a time to a timetable-only set; LR and fixed-configuration LightGBM | S | todo |
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
| 4.5 | **Predict the change, not the level** (1 h on) | target = final delay − `current_delay`; prediction = `current_delay` + predicted change. Trees cannot predict beyond the training range, so this helps most for large current delays | S | medium for RMSE | todo |

## Phase 5. Ensembles

Question: do combinations of models beat the best single model? Boosters that tie often make different errors; neural networks and trees combine especially well (TabArena, Erickson et al. 2025).

| ID | Experiment | Method | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 5.1 | **Plain average** | average the probabilities (arithmetic and geometric mean) or delays of XGBoost + LightGBM, then adding LR, CatBoost (6.1) and the neural networks (Phase 7). No weights are fitted, so no selection bias | S | low–medium | todo |
| 5.2 | **Seed and configuration bagging** | each booster with 5–10 seeds; the average of the top 5 configurations of the existing random search (`search_results.csv`) | S–M | low | todo |
| 5.3 | **Weighted ensemble** | weights by greedy ensemble selection (Caruana et al. 2004), chosen on inner folds | M | low–medium | todo |
| 5.4 | **Stacking** | logistic regression (severity) or linear regression (delay) on the inner-fold predictions of the base models; needs 0.3 | M | low–medium | todo |

## Phase 6. Further model families

Question: do other learners do better than XGBoost and LightGBM, or add to an ensemble? Each new package needs D5.

| ID | Experiment | Why | Cost | Status |
|---|---|---|---|---|
| 6.1 | **CatBoost** classifier and regressor, same 50-configuration search | ordered target statistics for categoricals; often the strongest booster on small tabular data and among the best tuned GBDTs in TabArena | M | todo |
| 6.2 | **Calibration** of every finalist: temperature scaling and isotonic regression, fitted on inner folds | log loss judges the probabilities; the random forest is expected to be too cautious (`classification_models.md`) | S | todo |
| 6.3 | **Explainable boosting machine** (`interpret`) | a GAM with pairwise interactions; close to boosting in accuracy on many datasets and readable per feature, useful for the report | M | todo |
| 6.4 | **Random effects for train number** (GPBoost) | boosting plus a random effect per train number: partial pooling for trains with few journeys; needs D7 | M | todo |
| 6.5 | **Distributional delay models**: NGBoost, quantile regression forest, LightGBMLSS (mixture) | already surveyed in `regression_models.md`; scored by CRPS and the RMSE of the mean; feeds 4.2 | M | todo |
| 6.6 | **Other losses for delay**: Huber, log-shifted target | robust to the long tail; RMSE may get worse, so this is a check rather than a likely gain | S | todo |

## Phase 7. Neural networks

Question: does a neural network match or beat the boosters here, and does it add to an ensemble? Grinsztajn et al. (2022) found trees usually ahead on data of this size; recent tabular networks (RealMLP, TabM) close the gap on benchmarks. Inputs: standardised numeric columns, the best calendar encoding from 1.1, embeddings or one-hot for categoricals, `current_delay` as 0 plus a missing indicator. Early stopping and architecture search use inner week folds (0.3), never the outer validation fold. Needs D5 and D6.

| ID | Experiment | Method | Cost | Status |
|---|---|---|---|---|
| 7.1 | **MLP baseline** (PyTorch) | 2–4 layers, dropout, weight decay, softmax for severity and squared error for delay; small random search. This is the reference for 7.3 and 7.4 | M | todo |
| 7.2 | **Multi-task MLP** | one shared body with two heads (4 classes and delay), trained together: the two targets describe the same arrival | M | todo |
| 7.3 | **TabGNS** (Orucu et al. 2026, arXiv:2602.10854) | gradient-based architecture search for MLPs: a gate per neuron chooses each layer's width (up to 5 layers of 512). Code: github.com/EricssonResearch/tabgns (MIT, 2 commits). Needs adapting to our log loss and folds; its gates are trained on a validation split, which must be an inner fold. Compare with 7.1 at the same compute | M | todo |
| 7.4 | **RealMLP and TabM** (`pytabkit`) | the two strongest tabular networks in TabArena v0.1; scikit-learn interface | M | todo |
| 7.5 | **Sequence model** over the upstream stops (1 h on) | GRU or small Transformer over the reported stops of the run before Frankfurt (`stops.parquet`, `upstream_runs.parquet`); stage 3 of the roadmap | L | todo |

**About TabGNS.** The paper compares only with other architecture-search methods (TabNAS, AgE) and a large fixed MLP, on datasets of 31,000 to 940,000 rows. Its main results are smaller networks (51–82% fewer parameters) and faster search, with accuracy about equal to a large MLP. It has no comparison with gradient-boosted trees. So we should expect TabGNS to find a compact MLP about as good as 7.1, not to beat the boosters. The comparison is still worth making, and the method is a reasonable "advanced neural model" for the report.

## Phase 8. Transfer learning and pretraining

Question: can we learn from data beyond our 17,507 CV journeys? Ordered by expected value for cost.

| ID | Experiment | What is transferred | Cost | Expected gain | Status |
|---|---|---|---|---|---|
| 8.1 | **Tabular foundation models**: TabPFN (v2, Apache 2.0; 2.5 and later, non-commercial licence), TabICL | networks pretrained on millions of synthetic tables, which predict our rows from the training rows given as context, without training on our data. Training folds of about 14,000 rows are within TabPFN-2.5's 50,000-row design. Optional fine-tuning. Each of us accepts the model licence on Hugging Face ourselves. Slow without a CUDA GPU (D6) | M–L | medium; strong on small tables in benchmarks | todo |
| 8.2 | **Pooling the prediction times** (D4) | one model on all five prediction times (about 86,000 rows), with the prediction time as a feature; the 24 h model learns from the 0 min rows' calendar effects and vice versa. Rows of one journey stay in one fold | S | low–medium | todo |
| 8.3 | **Other corridors** (D10) | Hamburg Hbf → München (26,526 journeys), Berlin Hbf → München (27,412) and Hamburg Hbf → Frankfurt (21,557) from `corridor_analysis/`. (a) pooled training with a corridor feature; (b) pretrain on the other corridors, then continue training on ours (LightGBM `init_model`, or fine-tune the network from 7.1); (c) the other-corridor model's prediction as a feature. Needs prediction rows (`current_delay`) for those corridors. **Leakage:** runs can belong to several corridors and share disruptions on the same days, so other-corridor journeys are used only from the training weeks of each fold | L | medium | todo |
| 8.4 | **Network-wide delay propagation** | pretrain a model on all ICE/IC runs in the monthly files: from the delay at some stop and the planned minutes and stops still to run, predict the delay at a later stop. Its prediction for our journey becomes a feature (1 h on), or it initialises the network of 7.5. `arrival_delays/` has only delays and train type, so this needs a new extraction from `data/monthly_processed_data`. Same week restriction as 8.3 | L | medium at 1 h–0 min | todo |
| 8.5 | **Self-supervised pretraining** (masked-feature reconstruction, SCARF) | not recommended: every row is labelled and there are about ten features, so there is little to learn without labels. Listed so that the choice is explicit | — | low | dropped (proposed) |

## Phase 9. Final evaluation and analysis

| ID | Step | Status |
|---|---|---|
| 9.1 | **Pre-register finalists:** at most three per task and prediction time, with their configurations and feature sets, written into this plan before the hold-out set is opened | todo |
| 9.2 | **Hold-out evaluation:** refit on all 95 CV weeks, score once on the 23 hold-out weeks; uncertainty by bootstrapping whole weeks | todo |
| 9.3 | **Railway only vs with weather** with the final models, on the hold-out set: the project's research question | todo |
| 9.4 | **Error analysis:** by class, by period (before / from October 2025), by branch (Nürnberg / Stuttgart), on extreme-weather days; calibration plots; SHAP explanations of the final model | todo |
| 9.5 | **Risk over the prediction times:** how a journey's predicted probabilities change from 24 h to 0 min, with example journeys (the "how risk updates during travel" focus) | todo |

## Not planned

- **Graph neural networks** (stage 3 of the roadmap): they need the state of the whole network as a graph; we have one corridor. 8.4 is the closest step towards network information.
- **Ordinal regression over all four classes:** cancellation breaks the order (see `classification_models.md`); 4.3 uses it only for the three delay classes.
- **Class weights:** they would raise macro-F1 but worsen log loss, the choosing metric (see `classification_models.md`).

## References

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
- Sigrist, F. (2022). Gaussian process boosting. *JMLR* 23. (GPBoost.)

## Log

- 4 November 2026: plan drafted; waiting for decisions D1–D10.
