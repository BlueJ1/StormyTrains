# Plan for finishing the project

Drafted 10 November 2026, after a review of all work in this repository (five independent reviews: dataset build, first models, advanced feature builders, advanced experiments and hold-out evaluation, early work and repository). The suggestions below are ours to accept or change; nothing in the "Decisions" table is decided yet.

**Two versions.** This file and the published page at https://claude.ai/artifact/HKS8W5KAvZb3zty1MbaqBv hold the same plan. When a decision is taken or a step is completed, update both (in the HTML, the data lists at the bottom of `finishing_plan.html`) and republish.

Status values: `todo`, `running`, `done`, `dropped`. Cost: S (under half a day), M (up to two days), L (more).

## Deadlines

| Date | What |
|---|---|
| Mondays 16, 23, 30 November; 7, 14 December 2026, 23:59 | Weekly reports 9 to 13 |
| Week 51 (14–20 December) | Report writing, presentation preparation, code submission if ready (course schedule) |
| Friday 8 January 2027 | Code and scripts |
| Monday 11 January 2027 | Report |
| Wednesday 13 January 2027 | Slides, as PDF |
| Thursday 14 January 2027 | Final presentation: 14 min plus 1 min questions; under 11 or over 17 min may fail |

Still unpublished by the course on 10 November: the times of the three final deadlines, our presentation slot, and which group's report we review.

## Where we stand

Against the brief's "minimum successful project":

| Requirement | State |
|---|---|
| Reproducible journey dataset with a clear severity target | Done (`frankfurt_munich/`). The download source and revision of the two main data folders are not documented in one place |
| Baselines and standard ML models with temporal evaluation | Done. The split is by stratified week, not chronological (agreed with the supervisor on 12 October); the report must justify it |
| Fair railway-only versus railway-plus-weather experiment | Done: matched pairs, same rows and budget, one hold-out evaluation. Result: no gain from weather |
| Explanation of errors, useful features and limitations | Partly: errors by class, period and branch, calibration and SHAP exist as files; they are not yet turned into findings |

Optional extensions: an advanced neural model is covered (MLPs, RealMLP, TabM, a gated architecture search, a GRU). Risk updates exist only before departure (five prediction times up to 0 min); the brief's "during the journey" setting is not covered. Extreme-weather analysis is too sparse to report. Connections and a decision-support prototype are not started.

Course deliverables: a first report draft now exists (`report/`). There are no slides, no statement of who did what, and no root README.

### What the review confirmed

- The week split is implemented as described: no week is shared between the hold-out set and the CV set or between folds, and nothing is fitted across the split.
- Targets match their definitions; every dataset number checked in the READMEs is right.
- Preprocessing is fitted inside the folds; metrics are correct; railway and weather arms use identical rows.
- The as-of joins of the snapshot-based inputs, the reliability windows (no use of the journey itself or of later journeys) and the weather hours are correct.
- Tests pass: 55 in `modeling/tests` (4 min), 130 passed and 5 skipped in `advanced_modeling/tests` (10 s).
- Every headline score in `modeling/README.md` matches its result file.

### What the review found

The numbered findings are listed under "Review findings" at the end. The ones that shape this plan:

1. **The network-state inputs are built wrongly** (R1). They count almost only disrupted trains, so the "share of trains delayed or cancelled" is 1.0 in 68% of the rows where it is known (99% at Frankfurt and München). Eight of the ten registered finalist pairs use them. This is not leakage, but the input is not what the plan and the report say it is.
2. **From 3 November 2025 the timetable inputs are almost always missing** (R2): 0% of rows have a planned route at 24 h and 6 h, 10–20% from 1 h on, against 99.8% before that date. So for about 40% of the journeys the weather models had only the two endpoint temperatures, and the train type works as a period indicator.
3. **The hold-out evaluation used inputs that the CV results no longer use** (R3). It ran before the review fixes; its models were trained with recent-reliability inputs built differently for training and test rows.
4. **`current_delay` decides the level of every score from 1 h on** (R4). Taken from the final records it exists for 86–90% of rows and gives 0.810 log loss and 22.6 min at 1 h; rebuilt from the snapshots it exists for 35–49% and gives 0.868 and 24.1 min. Where both exist they agree exactly in only 30% of rows. We do not yet know why the snapshot version is so sparse; it is not the November 2025 change in the archive (37% before, 33% after at 1 h).
5. **Small selection effects in the dataset build** (R5–R8): attributes of a kept replacement train, journeys with a withdrawn cancellation, the 0 min `current_delay` that can be the Frankfurt departure itself, and six cancellation times that are not the first issue time.
6. **Hundreds of comparisons on the same five folds** (R9): only large, consistent effects should be claimed. The reviewers' list of what the report can and cannot claim is under "Claims" below.
7. **Most of the fall in severity log loss from 24 h to 0 min is a change of the rows, not better prediction** (R18): announced cancellations leave the cohort, so the class-share reference itself falls from 0.968 to 0.897. Against that reference the best model gains 0.048 at 24 h and 0.078 at 0 min.
8. **The hold-out set has no reference score** (R19), and the hold-out delay error is 0.7 min above the registered CV score at every prediction time, which cannot be explained without one.
9. **The brief names Bahn-Vorhersage; we used `piebro/deutsche-bahn-data`** (R10), and no decision records why.

### Claims

From the review of the experiments, reproduced from the result files.

| The report can say | The report should soften | The report should not say |
|---|---|---|
| Recent reliability, and all new groups together, improve both tasks in CV (linear models on 5 of 5 folds; nested LightGBM group selection in 20 of 20 comparisons) | "Risk updates as departure approaches": give it against the class-share reference or on a common set of journeys | That neural networks match the boosted trees: against nested LightGBM they lost all 112 comparisons, and they used the base inputs only |
| Boosted trees are the strongest learners; no other family beat nested LightGBM | "Weather does not help": limit it to these observed-weather inputs; the hold-out set can rule out gains of about 0.003–0.006 log loss and 0.13 min, not smaller ones | That pooling the prediction times helps: railway-only pooling reached the minimum gain in 0 of 10 comparisons; the six hits are all in the weather arm |
| On the hold-out weeks weather gave no detectable improvement (ten intervals include zero; worse for delay at all five times). In CV, weather never reached the minimum gain for a stable learner | The XGBoost group selection (it chose the base inputs at 24 h and 6 h) | Gains of the multi-task network, the gated search or RealMLP: they beat only the plain MLP |
| | SHAP magnitudes across model families (TreeSHAP and sampled permutation SHAP are on different scales) | That weather-extreme indicators help (one hit, within noise), or that the value of recent reliability is confirmed on the hold-out set (no such pair was registered) |

Three arms failed rather than "showed no gain" and should be reported as failed or left out: LightGBMLSS (CV RMSE 32–36 min, 119 min at 0 min), the explainable boosting machine with weather for delay at 24 h (25.67 min, worse than the training mean), and the TabM classifier at 24 h (0.976, worse than class shares).

## Decisions to take

Suggested order: Q1–Q4 in the group this week, then Q1, Q2 and Q8 with the supervisor.

| # | Question | Our suggestion | State |
|---|---|---|---|
| Q1 | **How much do we correct and rerun?** (a) Nothing: report the existing results with caveats. (b) Targeted: fix or drop the network state (Q3), make the timetable static after 3 November 2025 (Q4), rerun the affected CV arms and the finalists. (c) Full: also change the dataset build (Q6) and rerun everything | (b). It removes the two defects that a reader would object to and fits in two weeks on the laptop; (c) risks the deadlines | open |
| Q2 | **Hold-out set after Q1.** The set has been scored once. (a) Keep the first evaluation only. (b) Sensitivity rerun: the registry stays frozen (same arms, feature groups and parameters), the inputs are rebuilt with the corrected code, and the models are refitted and scored once more. (c) Register new finalists from the corrected CV results and evaluate those | (b). The first evaluation stays the primary result: it is the only blind one. The rerun is written into the modelling plan before it runs, both are reported whatever the outcome, and the rerun is labelled as not blind. (c) would choose models after the hold-out set has been seen. (b) needs our explicit approval, because the hold-out gate refuses the changed source | open |
| Q3 | **Network state:** fix it (take the train type from the plan and select change rows by ID) or drop the group? | Fix it if the fix and its test take under a day (step B2); otherwise drop the group and say so | open |
| Q4 | **Timetable after 3 November 2025:** keep the strict rule (a plan counts only if the archive requested it before the prediction time), or treat the timetable as static, as already decided for the earlier period on 5 November? | Static. A timetable is published long before the day of travel; the request time says when the archive fetched it, not when it was known | open |
| Q5 | **`current_delay` in the report:** snapshot version only, or both versions as a lower and upper bound? | First find out why the snapshot version is sparse (step B1), then decide. Report both only if we can say what each bound means | open |
| Q6 | **Dataset selection effects (R5–R8):** fix in `build_dataset.py` (changes the dataset and every result) or document as limitations? | Document. The advanced inputs already take the timetable and delay history from the scheduled train, which covers the largest of them | open |
| Q7 | **Reference scores on the hold-out set** (class shares, training mean): compute them? They choose nothing, but they are new hold-out numbers | Yes, in the same run as Q2. Without them the report cannot say how much better than a constant the final models are on the hold-out weeks | open |
| Q8 | **Scope freeze:** no new model families or feature groups; predictions during the journey, connections and a prototype are named as future work | Yes, and confirm with the supervisor that leaving out the during-journey setting is acceptable | open |
| Q9 | **Which earlier results appear in the report?** The first models (recorded route, final-record delay) are optimistic | Only as the comparison in "What counts as known", labelled as such | open |
| Q10 | **Repository for submission:** root README and one environment file; what to do with `eda.ipynb` (empty since 8 October), `arrival_delays/` and `berlin_munich_weather/`; licence holder; attribution of the railway and DWD data | README and environment file: yes. Mark the early folders as historical and keep them; delete `eda.ipynb` unless someone needs the old version (commit d59d394). Licence and attribution: the group decides | open |
| Q11 | **Who did what, and use of AI tools.** The course requires each student's code and text contributions to be identifiable. The git history shows them poorly (squash merges under one account), and most commits carry an AI co-author line | Write a contribution table by folder and section into the root README and the report footnote. Ask the course staff how AI assistance should be declared | open |
| Q12 | **Why not Bahn-Vorhersage?** Record the reason for using `piebro/deutsche-bahn-data` and tell the supervisor | The group knows the reason; one sentence in `STATUS.md` and in the report | open |

## Work packages

Owners are left empty for the group to fill in.

### A. Decide (week 46)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| A1 | Group meeting on Q1–Q12; assign owners below | S | | 13 Nov | todo |
| A2 | Supervisor meeting: Q1, Q2, Q8, Q12, and the report outline | S | | 20 Nov | todo |

### B. Correct and rerun (weeks 46–48; depends on Q1–Q7)

If Q1 is (a), only B1, B8 and B9 remain.

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| B1 | Find out why the snapshot `current_delay` exists for only 35–49% of rows and disagrees with the final-record value: follow ten journeys through `upstream_events_cv.parquet` and the change snapshots. Outcome: one paragraph for the report and the answer to Q5 | S | | 17 Nov | todo |
| B2 | Fix `network_state.py` (Q3): select change rows by the IDs of ICE/IC plan rows instead of the change rows' own `train_type`; add a test with untyped change rows; check that the share is plausible (about 0.1–0.2) | S–M | | 18 Nov | todo |
| B3 | Static timetable after 3 November 2025 (Q4) in `scheduled_timetable.py`; rebuild timetable, history, network and weather inputs for the CV set; record the new coverage table | M | | 20 Nov | todo |
| B4 | Rerun the affected CV arms: the 0.7 references, 0.8 (nested, with group selection), 2.2, 2.4, 2.5, the combined groups, 3.1–3.5 and any arm that reads timetable, weather-branch or network inputs. Audit as before. Update `README.md`, `PLAN.md` and `plan.html` | L (laptop time) | | 27 Nov | todo |
| B5 | Write the sensitivity rerun into `PLAN.md` before it runs (Q2): frozen registry, corrected inputs, both results reported. Also record there that a first registry was withdrawn and replaced on 6 November after a fix to the network input, before the evaluation ran | S | | 30 Nov | todo |
| B6 | Build the hold-out inputs with the current code, refit the 20 registered arms on the corrected CV inputs and score them once (Q2), together with the two reference scores (Q7). Keep the first evaluation's files unchanged next to it | M | | 2 Dec | todo |
| B10 | Back up what exists only on this laptop: the hold-out prediction files, `frankfurt_munich/data/advanced/` and `advanced_original_065b077/` | S | | 13 Nov | todo |
| B7 | Decide on R5–R8 (Q6). If documented only: one paragraph each in `frankfurt_munich/README.md` | S | | 20 Nov | todo |
| B8 | Bring `frankfurt_munich/README.md` up to date (the hold-out warning, "Not built yet", the 74 clock-change rows) and say in `modeling/README.md` that its results use final-record `current_delay` | S | | 20 Nov | todo |
| B9 | Record the data source and its revision in one place: repository id, revision or download date, SHA-256 sums of the monthly files, how `stations_present_in_all_months.csv` was made | S | | 27 Nov | todo |

### C. Analysis for the report (weeks 48–50)

All from saved predictions; no new model.

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| C1 | Risk over the prediction times for all hold-out journeys, on the journeys present at all five times: how the predicted probability of "60+ min or cancelled" moves from 24 h to 0 min for journeys that were and were not severely disrupted, and the scores against the class-share reference on that common set (replaces the single example of 9.5; this is the brief's "how risk updates" question) | S | | 4 Dec | todo |
| C2 | Error analysis as findings: share of the squared error from 60+ min journeys, calibration figure, errors by period and branch | S | | 4 Dec | todo |
| C3 | Weather on rare days: say what can and cannot be concluded (two to four heavy-rain journeys per prediction time on the hold-out set); a descriptive table of outcome shares on snow, gust and heat days in the CV set | S | | 4 Dec | todo |
| C4 | Correct the wording of the results in `advanced_modeling/README.md` and `PLAN.md` where the review found it too strong (neural networks, pooling, the three failed arms), and update `plan.html` | S | | 27 Nov | todo |
| C5 | Regenerate Fig. 1, Fig. 2 and all tables of the report from the final result files (`report/figures/make_figures.py`); consider a figure of feature-group importance by prediction time | S | | 7 Dec | todo |

### D. Report (deadline 11 January)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| D1 | Download the course LaTeX template (Canvas, Templates) and move `report/main.tex` into it; set up Overleaf or keep it in the repository | S | | 20 Nov | todo |
| D2 | Related work: read the full texts of the weather studies named in `literature_review/regression_models.md` and `classification_models.md`, choose three or four, and check every `CHECK` entry in `report/references.bib` | M | | 27 Nov | todo |
| D3 | Resolve every `\todo` in `main.tex`, section by section; assign sections to authors | M | | 7 Dec | todo |
| D4 | Complete draft to the supervisor | | | 7 Dec | todo |
| D5 | Read the whole report together: terminology, figure and table references, five pages of text, formal style (the course returns reports with style problems) | S | | 16 Dec | todo |
| D6 | Revise after the supervisor's comments; add the division of work | M | | 8 Jan | todo |
| D7 | Submit, one submission for the group | | | 11 Jan | todo |

### E. Code submission (deadline 8 January)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| E1 | Root `README.md`: what the project is, folder map, data download, run order from raw files to the report's tables, who did what (Q11) | M | | 11 Dec | todo |
| E2 | One environment file for the main pipeline that matches `.venv` (the pins in `modeling/requirements.txt` differ from the installed numpy and pyarrow), plus the existing separate file for the distributional models | S | | 11 Dec | todo |
| E3 | Early folders (Q10): a header in each README saying it is historical and uses the files from before the source rebuild; complete the `arrival_delays` README; decide on `eda.ipynb` | S | | 11 Dec | todo |
| E4 | Licence and data attribution (Q10): licence holder; a note on the railway data (CC BY 4.0) and DWD | S | | 11 Dec | todo |
| E5 | Fresh-clone test: clone into an empty folder, create the environment, run both test suites and `report/figures/make_figures.py` | S | | 18 Dec | todo |
| E6 | Submit the repository link (the repository is public) with the contribution statement | | | 8 Jan | todo |

### F. Presentation (14 January)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| F1 | Outline from the report's structure; every student presents a part | S | | 16 Dec | todo |
| F2 | Slides | M | | 8 Jan | todo |
| F3 | Two timed rehearsals (target 13–14 min) | S | | 12 Jan | todo |
| F4 | Read the assigned group's report and prepare two questions (pairing not published yet) | S | | 13 Jan | todo |
| F5 | Submit the slides as PDF | | | 13 Jan | todo |

### G. Every week

| ID | Step | Status |
|---|---|---|
| G1 | Weekly report by Monday 23:59: achievements, problems, meetings, hours per member, plan, contact person; say whether we are ahead of, on or behind this plan | running |
| G2 | Supervisor contact | running |

## Timeline

| Week | Dates | Focus |
|---|---|---|
| 46 | 9–15 Nov | A1; B10; start B1, B2; D1 |
| 47 | 16–22 Nov | A2; B1–B3, B7, B8; D2 |
| 48 | 23–29 Nov | B4 (laptop runs), B9; C4; D2, D3 |
| 49 | 30 Nov–6 Dec | B5, B6; C1–C3; D3 |
| 50 | 7–13 Dec | D4 (draft to supervisor, Monday); C5; E1–E4 |
| 51 | 14–20 Dec | D5; E5; F1. Code submission if ready |
| 52–53 | 21 Dec–3 Jan | Break; buffer |
| 1 | 4–10 Jan | D6; F2; E6 (Friday 8 Jan) |
| 2 | 11–14 Jan | D7 (Monday); F3, F4; F5 (Wednesday); presentation (Thursday) |

## Risks

| Risk | What we do |
|---|---|
| The reruns of B4 take longer than a week on one laptop | Start B2 and B3 this week. If B4 is not finished by 4 December, fall back to Q1 (a) for the parts that are missing and say so in the report |
| The sensitivity rerun on the hold-out set differs from the first evaluation | Report both, whichever way they point; the rerun is written into the plan before the inputs are built |
| The supervisor's comments arrive late | Draft to the supervisor on 7 December, a week before the course's writing week |
| The report exceeds five pages of text | The draft is about five pages; anything new replaces something |
| Contributions cannot be shown from git | Contribution table written by the group (Q11), early |

## Review findings

Severity as judged by the reviewers. "Checked" means the finding was reproduced with code on 10 November, not only read.

| # | Area | Finding | Severity | Checked | Step |
|---|---|---|---|---|---|
| R1 | Advanced inputs | `network_state.py` keeps only change rows that carry a train type (4% of rows, almost all of disrupted trains). Shares are 1.0 in 68% of known rows; the tests always supply a train type and miss it | critical | yes | B2 |
| R2 | Advanced inputs | Timetable inputs (planned route, stops, weather branch, scheduled train type) are missing for nearly all rows from 3 November 2025; route weather is missing for 33–41% of CV rows | major | yes | B3 |
| R3 | Hold-out | The hold-out input files are those of the first evaluation; the CV files have changed since (reliability in up to 49,458 of 85,901 rows). The first evaluation's models were trained on reliability inputs built from a CV-only roster and tested on inputs built from all journeys; this affects 8 of the 10 pairs (not the two at 24 h). The CV tables in the README use the corrected inputs, so hold-out scores are comparable only with the CV scores in `finalists.json` | major | yes | B5, B6 |
| R4 | First models | `current_delay` from final records is not checked for visibility at the prediction time, and a stop cancelled later drops out of it. It accounts for about 1.4 min RMSE and 0.05 log loss from 1 h on | critical for interpretation | partly | B1 |
| R5 | Dataset | For a departure whose scheduled train was cancelled and replaced, the kept journey is the replacement (265 of 298 groups), and its train type and `starts_at_origin` enter the base inputs | major, small | yes | B7 |
| R6 | Dataset | Journeys whose cancellation was announced and later withdrawn stay in at every prediction time (up to 60 rows at 0 min), while other announced cancellations are dropped | major, small | yes | B7 |
| R7 | Dataset | At 0 min, `current_delay` can be the Frankfurt departure itself (20.7% of values); for trains starting in Frankfurt a missing value means "left late" | minor to major | yes | B7 |
| R8 | Dataset | `cancellation_announced` is the issue time in the latest snapshot; six journeys were first cancelled earlier | minor | yes | B7 |
| R9 | Experiments | Hundreds of paired comparisons on the same five folds; exploratory screens precede the nested comparisons | major for claims | read | D3 |
| R10 | Repository | The railway source differs from the brief and is named in full only in `berlin_munich_weather/README.md` and the advanced README | major | yes | B9, Q12 |
| R11 | First models | All 20 first result folders use the recorded-route inputs; the leak (0.13–0.16 min, 0.007 log loss) is as large as the gain of boosted trees over linear models at 24 h and 6 h | major | yes | Q9 |
| R12 | First models | Paired t-values over folds in `modeling/README.md` overstate significance; several chosen booster configurations lie at the edge of the search space | minor | yes | Q9 |
| R13 | Repository | No root README; three requirements files with different pins; `modeling/` imports `advanced_modeling/` | major | yes | E1, E2 |
| R14 | Repository | Numbers in `berlin_munich_weather/`, `corridor_analysis/` and `arrival_delays/` come from source files that no longer exist locally; `eda.ipynb` is empty; the reasons for choosing Frankfurt–München are written nowhere | minor | yes | E3 |
| R15 | Weather | The 24-hour snow flag uses the DWD precipitation-type code although its reference interval was never resolved; codes 7 and 8 include forms other than snow | minor | read | D3 |
| R16 | Advanced inputs | Missing delay history depends on replacement status (6.7% of replacement rows have a `current_delay` against 42% of others); 7 CV replacements have no history at all | minor | yes | D3 |
| R18 | Experiments | The severity cohort shrinks as cancellations are announced, so scores at different prediction times are not on the same journeys; most of the fall in log loss is the falling reference | major for claims | yes | C1 |
| R19 | Hold-out | No reference score (class shares, training mean) was computed on the hold-out set | major | yes | B6 |
| R20 | Experiments | README and plan report neural networks and pooling against weak baselines; three arms failed and are listed as "no gain" (see "Claims") | major for claims | yes | C4 |
| R21 | Hold-out | A first finalist registry was withdrawn and replaced on 6 November after the network input was fixed; three delay finalists changed; hold-out inputs without labels had been built by then. Must be disclosed | minor | read | B5 |
| R22 | Hold-out | The bootstrap is correct (whole weeks, same draw for both arms, 2,000 resamples), but 23 weeks give narrow percentile intervals and the ten intervals are pointwise | minor | yes | D3 |
| R17 | Literature | `literature_review/references.bib` uses biblatex fields and lacks every later reference; three 2026 sources are preprints; several claims in the surveys were taken from abstracts | minor | yes | D2 |

## Done on 10 November 2026

- Removed 10 clean worktrees and 15 local branches (all merged through PRs #6–#12, or superseded). The out-of-fold predictions of the first models existed only inside those worktrees; they were copied into `frankfurt_munich/modeling/results/` first. The remote branch `feature-engineering` (created 9 November) was left alone.
- `STATUS.md` rewritten to the current state; `.gitignore` now covers `.claude/`, `.pytest_cache/` and the LaTeX build files; three small errors in `modeling/README.md` corrected; the modelling plan no longer says that PR #12 is unmerged.
- First report draft in `report/` (LaTeX, two figures drawn from the result files, open points marked with `\todo`).

## Log

- 10 November 2026: plan drafted after the project review.
