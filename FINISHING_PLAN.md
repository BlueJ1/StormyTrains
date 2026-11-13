# Plan for finishing the project

Drafted 10 November 2026, after a review of all work in this repository (five independent reviews: dataset build, first models, advanced feature builders, advanced experiments and hold-out evaluation, early work and repository). The decisions were taken on 12 November 2026 (Q11 is deferred); the work packages, timeline and risks below follow from them.

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
| Fair railway-only versus railway-plus-weather experiment | Done once: matched pairs, same rows and budget, one hold-out evaluation, no gain from weather. It is repeated on the corrected inputs with newly registered finalists (Q1, Q2) |
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
9. **The brief names Bahn-Vorhersage; we used `piebro/deutsche-bahn-data`** (R10). Reason, recorded on 12 November: the Bahn-Vorhersage data has not been made available to us by the data owners; the supervisor is aware.

### Claims

From the review of the experiments, reproduced from the result files. The table describes the results from before the rebuild and is checked again after B4.

| The report can say | The report should soften | The report should not say |
|---|---|---|
| Recent reliability, and all new groups together, improve both tasks in CV (linear models on 5 of 5 folds; nested LightGBM group selection in 20 of 20 comparisons) | "Risk updates as departure approaches": give it against the class-share reference or on a common set of journeys | That neural networks match the boosted trees: against nested LightGBM they lost all 112 comparisons, and they used the base inputs only |
| Boosted trees are the strongest learners; no other family beat nested LightGBM | "Weather does not help": limit it to these observed-weather inputs; the hold-out set can rule out gains of about 0.003–0.006 log loss and 0.13 min, not smaller ones | That pooling the prediction times helps: railway-only pooling reached the minimum gain in 0 of 10 comparisons; the six hits are all in the weather arm |
| On the hold-out weeks weather gave no detectable improvement (ten intervals include zero; worse for delay at all five times). In CV, weather never reached the minimum gain for a stable learner | The XGBoost group selection (it chose the base inputs at 24 h and 6 h) | Gains of the multi-task network, the gated search or RealMLP: they beat only the plain MLP |
| | SHAP magnitudes across model families (TreeSHAP and sampled permutation SHAP are on different scales) | That weather-extreme indicators help (one hit, within noise), or that the value of recent reliability is confirmed on the hold-out set (no such pair was registered) |

Three arms failed rather than "showed no gain" and should be reported as failed or left out: LightGBMLSS (CV RMSE 32–36 min, 119 min at 0 min), the explainable boosting machine with weather for delay at 24 h (25.67 min, worse than the training mean), and the TabM classifier at 24 h (0.976, worse than class shares).

## Decisions

Taken on 12 November 2026; Q11 is deferred. Where a decision differs from the suggestion in the draft of 10 November, it says so.

| # | Question | Decision | State |
|---|---|---|---|
| Q1 | **How much do we correct and rerun?** (a) Nothing: report the existing results with caveats. (b) Targeted: fix or drop the network state (Q3), make the timetable static after 3 November 2025 (Q4), rerun the affected CV arms and the finalists. (c) Full: also change the dataset build (Q6) and rerun everything. | (c) Full. The network state is fixed (Q3), the timetable is static after 3 November 2025 (Q4), the dataset build is corrected (Q6), and every input and every CV arm in `advanced_modeling/` is rebuilt and rerun. The first models in `modeling/` are not rerun; their results are deleted (Q9). The draft suggested (b). | decided |
| Q2 | **Hold-out set after Q1.** The set was scored once, on 6 November. (a) Keep the first evaluation only. (b) Sensitivity rerun with the registry frozen. (c) Register new finalists from the corrected CV results and evaluate those. | (c). New finalists are registered from the corrected CV results only and evaluated once on the hold-out weeks. The report gives the new evaluation only. The first evaluation stays in the git history (commit 065b077) and in the modelling plan; its result files move to `results/advanced_holdout_first/`. The draft suggested (b). | decided |
| Q3 | **Network state: fix or drop?** Fixing means taking the train type from the plan and selecting change rows by ID. | Fix it (step B2). | decided |
| Q4 | **Timetable after 3 November 2025.** Keep the strict rule (a plan counts only if the archive requested it before the prediction time), or treat the timetable as static, as already decided for the earlier period on 5 November? | Static (step B3). | decided |
| Q5 | **`current_delay` in the report.** Snapshot version only, or both versions as a lower and upper bound? | As suggested: first find out why the snapshot version is sparse (step B1). The choice between one version and both comes back to the group with that result. B1 done on 13 November: the snapshot version is sparse because the archive fetched live updates only every six hours; the final-record version treats every report as visible at once. The choice is now open. | open |
| Q6 | **Dataset selection effects (R5–R8).** Fix in `build_dataset.py`, which changes the dataset and every result, or document as limitations? | Fix all four (step B7); R5 is corrected where the advanced models read the journeys, not in `build_dataset.py` (12 November). For R6: a journey has no prediction row at a prediction time where its cancellation was in force (announced and not yet withdrawn) and has one again after the withdrawal, the same rule as for every other announced cancellation. The draft suggested documenting only. | decided |
| Q7 | **Reference scores on the hold-out set.** Class shares and training mean. They choose nothing, but they are new hold-out numbers. | Yes. New numbers on the hold-out set are accepted; the two references are scored in the same run as the new finalists (step B6). | decided |
| Q8 | **Scope freeze.** No new model families or feature groups; predictions during the journey, connections and a prototype are named as future work. | The scope is frozen here for now. | decided |
| Q9 | **Which earlier results appear in the report?** The first models (recorded route, final-record delay) are optimistic. | None. The report does not mention the first models, and their result folders and tables are deleted (step B12). The shared code in `modeling/` stays. The draft suggested keeping them as a labelled comparison. | decided |
| Q10 | **Repository for submission.** Root README and one environment file; what to do with `eda.ipynb` (empty since 8 October), `arrival_delays/` and `berlin_munich_weather/`; licence holder; attribution of the railway and DWD data. | Create the root README (step E1). Remove `arrival_delays/`, `berlin_munich_weather/` and `eda.ipynb` (step E3). `corridor_analysis/` stays, because `build_dataset.py` uses its `extract.py`. Not answered yet: the single environment file, the licence holder and the data attribution (steps E2 and E4). | decided |
| Q11 | **Who did what, and use of AI tools.** The course requires each student's code and text contributions to be identifiable. The git history shows them poorly (squash merges under one account), and most commits carry an AI co-author line. | Left empty for now. The course needs the statement with the code submission on 8 January, so the question comes back in December. | deferred |
| Q12 | **Why not Bahn-Vorhersage?** Record the reason for using `piebro/deutsche-bahn-data` and tell the supervisor. | The Bahn-Vorhersage data has not been made available to us by the data owners, so we use `piebro/deutsche-bahn-data`. The supervisor is aware. Recorded in `STATUS.md` and in the report draft. | decided |

## Work packages

Owners are left empty for the group to fill in.

### A. Decide (week 46)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| A1 | Decisions on Q1–Q12: taken on 12 November 2026; Q11 is deferred | S |  | 13 Nov | done |
| A2 | Supervisor meeting: tell the supervisor about the full rebuild and the second hold-out evaluation (Q1, Q2) and the scope freeze (Q8); go through the report outline | S |  | 20 Nov | todo |
| A3 | Assign owners to the steps below | S |  | 16 Nov | todo |

### B. Correct, rebuild and rerun (weeks 46–49; decisions Q1–Q7 and Q9)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| B10 | Back up what exists only on this laptop: the hold-out prediction files, `frankfurt_munich/data/advanced/` and `advanced_original_065b077/`. Before anything is rebuilt. State on 12 November: every file under `frankfurt_munich/` that is not in git (514 files, 1.06 GB) is copied to the folder `Project in Data Science backup 2026-11-12` next to the repository, with checksums. That protects against the rebuild overwriting files. Still to do: a copy somewhere other than this laptop | S |  | 13 Nov | running |
| B1 | Find out why the snapshot `current_delay` exists for only 35–49% of rows and disagrees with the final-record value. Done on 13 November, on the rebuilt inputs (39%, 43% and 52% of CV rows at 1 h, 20 min and 0 min; the final-record version 86–87%). Cause: the archive fetched live updates in blocks about six hours apart. In 98% of the rows without a snapshot value, the run's stops are in the plan but no live update of any of them had been fetched by the cutoff; in a sample, the first one arrived a median of 1.9 h after the cutoff. Not the cause: missing plans (B3 raised the coverage only from 35% to 39% at 1 h) or on-time stops lacking live updates (every stop has some). Where both versions exist, the snapshot version is lower in 58% of rows and equal in about a third; the mean difference grows with the age of the latest report, from −1.0 min under 30 min to −11.7 min over 4 h. The paragraph is in the report draft (Section on the data). Q5 goes back to the group | S |  | 17 Nov | done |
| B2 | Fix `network_state.py` (Q3): select change rows by the IDs of ICE/IC plan rows instead of the change rows' own `train_type`; add a test with untyped change rows; check that the share is plausible (about 0.1–0.2). Code done on 12 November: in two sample months (March 2025 and March 2026) the network share is known for every row and averages 0.08 (before: 0.83 and 1.00). The inputs themselves are rebuilt in B11 | S–M |  | 18 Nov | done |
| B3 | Static timetable after 3 November 2025 (Q4), with a test. Code done on 12 November in the plan rule that all input builders share (`history_features.plan_availability_utc`), so the planned times in the network state, delay history, recent reliability and transfer inputs follow it too. For the 400 latest CV journeys the timetable inputs are known at every prediction time (before: 0–30%). The inputs themselves are rebuilt in B11 | S–M |  | 19 Nov | done |
| B7 | Correct the dataset build (Q6) in `frankfurt_munich/build_dataset.py` and rebuild the dataset. R5: for a kept replacement train, `train_type` and `starts_at_origin` come from the scheduled train; decided on 12 November that `build_dataset.py` stays as it is, because the advanced loader already takes both from the timetable inputs (checked in B11). R6: no prediction row while a cancellation is in force. R7: `current_delay` no longer takes the Frankfurt departure itself. R8: `cancellation_announced` is the first issue time over all snapshots. `split_weeks.csv` stays as it is, so the hold-out weeks do not change. One test per fix; rerun the dataset audit. State on 12 November: R6, R7 and R8 are corrected in code, with tests in `frankfurt_munich/tests/`. The group confirmed the rules for R6 and R7 on 12 November, and the dataset was rebuilt the same day: 85,826 CV prediction rows (before 85,901) and 21,236 hold-out rows (before 21,260); journeys, folds and hold-out weeks are unchanged. The audit, the change check and the weather rows were rerun and the dataset README is updated. The advanced inputs still belong to the old prediction rows until B11 | M |  | 20 Nov | done |
| B12 | Delete the first models' results (Q9): the 20 result folders in `frankfurt_munich/modeling/results/` and the result tables in `modeling/README.md`. The shared code and its tests stay | S |  | 20 Nov | todo |
| B8 | Bring `frankfurt_munich/README.md` up to date with the rebuilt dataset: every count, the hold-out warning, "Not built yet", the 74 clock-change rows. Done on 12–13 November: counts and rules with the rebuild (B7); the hold-out note now says it is kept for the final evaluation; "Not built yet" became "Features built elsewhere"; the 74 clock-change rows are 36 CV and 38 hold-out, one of them at 1 h | S |  | 24 Nov | done |
| B11 | Rebuild every advanced input for the CV set from the corrected dataset (timetable, delay history, recent reliability, network state, weather, the corridor and propagation caches); record the new coverage table, and check that the scheduled `train_type` and `starts_at_origin` are known for the kept replacement trains (R5). Done on 13 November in about 30 minutes; every input has exactly the dataset's 85,826 prediction keys. The timetable inputs are known for 99.4% of rows at every prediction time (before: about 60%); the network share is known for nearly every row and averages 0.11; the snapshot `current_delay` covers 39–52% from 1 h on. R5: the scheduled type is known for 206 of the 230 kept replacement journeys; for the other 24 no plan matched the replaced train, and the loader sets it to `unknown`. Coverage table in `advanced_modeling/README.md`. The hold-out inputs are not rebuilt (B6) | M |  | 24 Nov | done |
| B9 | Record the data source in one place: repository id, revision or download date, SHA-256 sums of the monthly files, how `stations_present_in_all_months.csv` was made, and why it is not Bahn-Vorhersage (Q12) | S |  | 27 Nov | todo |
| B4 | Rerun all 366 CV arms on the rebuilt inputs and audit them as before. Rewrite the result tables in `advanced_modeling/README.md` and the result lines in `PLAN.md` and `plan.html`. For scale: the rerun of 336 arms on 7 November finished within a day on this laptop | L |  | 1 Dec | todo |
| B5 | Write the second evaluation into `PLAN.md` before anything is rerun: decisions D14–D16 and Phase 10 (new finalists from the corrected CV results only, by the rules of 9.1; one more scoring of the hold-out weeks) | S |  | 12 Nov | done |
| B6 | Register the new finalists in `finalists.json` from the corrected CV results. Build the hold-out inputs with the corrected code, refit on the 95 CV weeks and score once on the 23 hold-out weeks, together with class shares and the training mean (Q7). The first evaluation's files move to `results/advanced_holdout_first/` | M |  | 4 Dec | todo |

### C. Analysis for the report (weeks 49–50; from the new saved predictions, no new model)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| C1 | Risk over the prediction times for all hold-out journeys, on the journeys present at all five times: how the predicted probability of "60+ min or cancelled" moves from 24 h to 0 min for journeys that were and were not severely disrupted, and the scores against the class-share reference on that common set. This is the brief's "how risk updates" question | S |  | 9 Dec | todo |
| C2 | Error analysis as findings: share of the squared error from 60+ min journeys, calibration figure, errors by period and branch | S |  | 9 Dec | todo |
| C3 | Weather on rare days: say what can and cannot be concluded on the hold-out set; a descriptive table of outcome shares on snow, gust and heat days in the CV set | S |  | 9 Dec | todo |
| C4 | Check the wording of the rewritten results (B4) against the review's list of claims: neural networks, pooling, failed arms | S |  | 4 Dec | todo |
| C5 | Regenerate Fig. 1, Fig. 2 and all tables of the report from the new result files (`report/figures/make_figures.py`); consider a figure of feature-group importance by prediction time | S |  | 11 Dec | todo |

### D. Report (deadline 11 January)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| D1 | Download the course LaTeX template (Canvas, Templates) and move `report/main.tex` into it; set up Overleaf or keep it in the repository | S |  | 20 Nov | todo |
| D2 | Related work: read the full texts of the weather studies named in `literature_review/regression_models.md` and `classification_models.md`, choose three or four, and check every `CHECK` entry in `report/references.bib` | M |  | 27 Nov | todo |
| D3 | Resolve every `\todo` in `main.tex`, section by section; assign sections to authors. Method and related work can be finished before the new results exist | M |  | 11 Dec | todo |
| D4 | Complete draft to the supervisor. A week later than in the draft of this plan, because every number changes until B6 |  |  | 14 Dec | todo |
| D5 | Read the whole report together: terminology, figure and table references, five pages of text, formal style (the course returns reports with style problems) | S |  | 18 Dec | todo |
| D6 | Revise after the supervisor's comments; add the division of work (Q11) | M |  | 8 Jan | todo |
| D7 | Submit, one submission for the group |  |  | 11 Jan | todo |

### E. Code submission (deadline 8 January)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| E1 | Root `README.md` (Q10): what the project is, folder map, data download, run order from raw files to the report's tables. The part on who did what stays empty for now (Q11) | M |  | 11 Dec | todo |
| E2 | One environment file for the main pipeline that matches `.venv` (the pins in `modeling/requirements.txt` differ from the installed numpy and pyarrow), plus the existing separate file for the distributional models. Not yet confirmed by the group | S |  | 11 Dec | todo |
| E3 | Remove `arrival_delays/`, `berlin_munich_weather/` and `eda.ipynb` (Q10). First move the SSL context that `frankfurt_munich/weather_features.py` imports from `berlin_munich_weather/fetch_weather.py` into `frankfurt_munich/`, and update the references in the READMEs. `corridor_analysis/` stays and gets a header saying that its numbers come from the files before the source rebuild | S |  | 20 Nov | todo |
| E4 | Licence and data attribution: licence holder; a note on the railway data (CC BY 4.0) and DWD. Not yet decided by the group | S |  | 11 Dec | todo |
| E5 | Fresh-clone test: clone into an empty folder, create the environment, run both test suites and `report/figures/make_figures.py` | S |  | 18 Dec | todo |
| E6 | Submit the repository link (the repository is public) with the contribution statement |  |  | 8 Jan | todo |

### F. Presentation (14 January)

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| F1 | Outline from the report's structure; every student presents a part | S |  | 18 Dec | todo |
| F2 | Slides | M |  | 8 Jan | todo |
| F3 | Two timed rehearsals (target 13–14 min) | S |  | 12 Jan | todo |
| F4 | Read the assigned group's report and prepare two questions (pairing not published yet) | S |  | 13 Jan | todo |
| F5 | Submit the slides as PDF |  |  | 13 Jan | todo |

### G. Every week

| ID | Step | Cost | Owner | Due | Status |
|---|---|---|---|---|---|
| G1 | Weekly report by Monday 23:59: achievements, problems, meetings, hours per member, plan, contact person; say whether we are ahead of, on or behind this plan |  |  | Mondays | running |
| G2 | Supervisor contact |  |  | weekly | running |

## Timeline

| Week | Dates | Focus |
|---|---|---|
| 46 | 9–15 Nov | A1 (done), A3; B5 (done), B10; start B1, B2 |
| 47 | 16–22 Nov | A2; B1–B3, B7, B12; E3; D1, D2 |
| 48 | 23–29 Nov | B8, B11, B9; start B4 (laptop runs); D2, D3 |
| 49 | 30 Nov–6 Dec | B4, C4; B6; D3 |
| 50 | 7–13 Dec | C1–C3, C5; D3; E1, E2, E4 |
| 51 | 14–20 Dec | D4 (draft to supervisor, Monday); D5; E5; F1. Code submission if ready |
| 52–53 | 21 Dec–3 Jan | Break; buffer |
| 1 | 4–10 Jan | D6; F2; E6 (Friday 8 Jan) |
| 2 | 11–14 Jan | D7 (Monday); F3, F4; F5 (Wednesday); presentation (Thursday) |

## Risks

| Risk | What we do |
|---|---|
| The full rebuild and rerun (B7, B11, B4) take longer than planned on one laptop | B2, B3 and B7 start next week. If the CV reruns are not finished by 4 December, the group decides which arms the report needs and the rest waits |
| The corrected dataset and inputs change the findings | Every number in the report draft is provisional until B6. C5 redraws all figures and tables from the new result files, and the text is revised against them |
| The supervisor sees the full draft only on 14 December | Method and related work can go to the supervisor earlier; the outline is discussed in A2 |
| The report exceeds five pages of text | The draft is about five pages; anything new replaces something |
| Contributions cannot be shown from git | Q11 is deferred, but the course requires the statement with the code on 8 January. It comes back in December |

## Review findings

Severity as judged by the reviewers. "Checked" means the finding was reproduced with code on 10 November, not only read.

| # | Area | Finding | Severity | Checked | Step |
|---|---|---|---|---|---|
| R1 | Advanced inputs | `network_state.py` keeps only change rows that carry a train type (4% of rows, almost all of disrupted trains). Shares are 1.0 in 68% of known rows; the tests always supply a train type and miss it | critical | yes | B2 |
| R2 | Advanced inputs | Timetable inputs (planned route, stops, weather branch, scheduled train type) are missing for nearly all rows from 3 November 2025; route weather is missing for 33–41% of CV rows | major | yes | B3 |
| R3 | Hold-out | The hold-out input files are those of the first evaluation; the CV files have changed since (reliability in up to 49,458 of 85,901 rows). The first evaluation's models were trained on reliability inputs built from a CV-only roster and tested on inputs built from all journeys; this affects 8 of the 10 pairs (not the two at 24 h). The CV tables in the README use the corrected inputs, so hold-out scores are comparable only with the CV scores in `finalists.json` | major | yes | B6 |
| R4 | First models | `current_delay` from final records is not checked for visibility at the prediction time, and a stop cancelled later drops out of it. It accounts for about 1.4 min RMSE and 0.05 log loss from 1 h on | critical for interpretation | partly | B1 |
| R5 | Dataset | For a departure whose scheduled train was cancelled and replaced, the kept journey is the replacement (265 of 298 groups), and its train type and `starts_at_origin` enter the base inputs | major, small | yes | B7 |
| R6 | Dataset | Journeys whose cancellation was announced and later withdrawn stay in at every prediction time (up to 60 rows at 0 min), while other announced cancellations are dropped | major, small | yes | B7 |
| R7 | Dataset | At 0 min, `current_delay` can be the Frankfurt departure itself (20.7% of values); for trains starting in Frankfurt a missing value means "left late" | minor to major | yes | B7 |
| R8 | Dataset | `cancellation_announced` is the issue time in the latest snapshot; six journeys were first cancelled earlier | minor | yes | B7 |
| R9 | Experiments | Hundreds of paired comparisons on the same five folds; exploratory screens precede the nested comparisons | major for claims | read | D3 |
| R10 | Repository | The railway source differs from the brief and is named in full only in `berlin_munich_weather/README.md` and the advanced README | major | yes | B9 |
| R11 | First models | All 20 first result folders use the recorded-route inputs; the leak (0.13–0.16 min, 0.007 log loss) is as large as the gain of boosted trees over linear models at 24 h and 6 h | major | yes | B12 |
| R12 | First models | Paired t-values over folds in `modeling/README.md` overstate significance; several chosen booster configurations lie at the edge of the search space | minor | yes | B12 |
| R13 | Repository | No root README; three requirements files with different pins; `modeling/` imports `advanced_modeling/` | major | yes | E1, E2 |
| R14 | Repository | Numbers in `berlin_munich_weather/`, `corridor_analysis/` and `arrival_delays/` come from source files that no longer exist locally; `eda.ipynb` is empty; the reasons for choosing Frankfurt–München are written nowhere | minor | yes | E3 |
| R15 | Weather | The 24-hour snow flag uses the DWD precipitation-type code although its reference interval was never resolved; codes 7 and 8 include forms other than snow | minor | read | D3 |
| R16 | Advanced inputs | Missing delay history depends on replacement status (6.7% of replacement rows have a `current_delay` against 42% of others); 7 CV replacements have no history at all | minor | yes | D3 |
| R18 | Experiments | The severity cohort shrinks as cancellations are announced, so scores at different prediction times are not on the same journeys; most of the fall in log loss is the falling reference | major for claims | yes | C1 |
| R19 | Hold-out | No reference score (class shares, training mean) was computed on the hold-out set | major | yes | B6 |
| R20 | Experiments | README and plan report neural networks and pooling against weak baselines; three arms failed and are listed as "no gain" (see "Claims") | major for claims | yes | C4 |
| R21 | Hold-out | A first finalist registry was withdrawn and replaced on 6 November after the network input was fixed; three delay finalists changed; hold-out inputs without labels had been built by then. Recorded in the modelling plan | minor | read | B5 |
| R22 | Hold-out | The bootstrap is correct (whole weeks, same draw for both arms, 2,000 resamples), but 23 weeks give narrow percentile intervals and the ten intervals are pointwise | minor | yes | D3 |
| R17 | Literature | `literature_review/references.bib` uses biblatex fields and lacks every later reference; three 2026 sources are preprints; several claims in the surveys were taken from abstracts | minor | yes | D2 |

## Done on 10 November 2026

- Removed 10 clean worktrees and 15 local branches (all merged through PRs #6–#12, or superseded). The out-of-fold predictions of the first models existed only inside those worktrees; they were copied into `frankfurt_munich/modeling/results/` first. The remote branch `feature-engineering` (created 9 November) was left alone.
- `STATUS.md` rewritten to the current state; `.gitignore` now covers `.claude/`, `.pytest_cache/` and the LaTeX build files; three small errors in `modeling/README.md` corrected; the modelling plan no longer says that PR #12 is unmerged.
- First report draft in `report/` (LaTeX, two figures drawn from the result files, open points marked with `\todo`).

## Log

- 10 November 2026: plan drafted after the project review.
- 12 November 2026: decisions taken on Q1–Q12 (Q11 deferred). Full correction: network state fixed, timetable static after 3 November 2025, dataset build corrected for R5–R8, every input rebuilt and all CV arms rerun; new finalists are registered from the corrected CV results and evaluated on the hold-out weeks, with reference scores. The first models stay out of the report and their results are deleted. `arrival_delays/`, `berlin_munich_weather/` and `eda.ipynb` are removed; `corridor_analysis/` stays. Bahn-Vorhersage was not made available by its owners. Work packages, timeline and risks adjusted; the report draft goes to the supervisor on 14 December instead of 7 December.
- 12 November 2026: code for B2, B3 and B7 written and tested; no data or input file is rebuilt yet. B7 on the CV journeys, computed without writing files: the cancellation rule (R6) removes 3, 14, 18, 19 and 21 prediction rows at 24 h, 6 h, 1 h, 20 min and 0 min. A withdrawal counts from the time the first snapshot that shows it was requested, because the snapshots carry no time for the withdrawal itself; of the 59 CV journeys with a withdrawn cancellation, 39 have a 0 min row and 20 do not. `cancellation_announced` (R8) changes for 3 CV journeys. Without the Frankfurt departure (R7), `current_delay` at 0 min changes for 859 CV journeys and becomes missing for 563, 547 of them trains that start in Frankfurt; the other prediction times do not change. Open for the group: R5, because the advanced models already take `train_type` and `starts_at_origin` from the scheduled train through the timetable inputs, which B3 now makes available for nearly every row, so `build_dataset.py` may not need a change; and the rules for R6 and R7 as written.
- 12 November 2026: R5 decided. `build_dataset.py` is not changed for it: the journey files keep the type and starting point of the train that ran, and the advanced models take both from the scheduled train through the timetable inputs, as they already do. B11 checks that these are known for the kept replacement trains. The dataset README says so when it is updated with the rebuild.
- 12 November 2026: rules for R6 and R7 confirmed as written. R6: a cancellation holds from its issue time, and a withdrawal ends it from the time the first snapshot that shows it was requested. R7: `current_delay` never uses the departure from Frankfurt itself, so at 0 min it is missing for every train that starts in Frankfurt. What is left of B7 is the rebuild, after the backup of B10.
- 12 November 2026: B10, local copy made. Every file under `frankfurt_munich/` that is not in git is in `Project in Data Science backup 2026-11-12`, next to the repository: the advanced inputs, the inputs the first hold-out evaluation used, the out-of-fold and hold-out predictions, and the DWD archives; 514 files, 1.06 GB, checksums identical to the sources. The 19 GB of source data are not included, because they can be downloaded again. A copy off this laptop is still to be made.
- 12 November 2026: B7 done, dataset rebuilt with the corrected build. CV prediction rows: 17,463, 17,366, 17,096, 16,975 and 16,926 at 24 h, 6 h, 1 h, 20 min and 0 min (3, 14, 18, 19 and 21 fewer than before), as computed beforehand; hold-out rows: 2, 4, 6, 6 and 6 fewer. The journeys, their folds and `split_weeks.csv` did not change; in the journey files only `cancellation_announced` (3 CV journeys, 1 hold-out) and `current_delay_0min` differ. `audit.py`, `check_change_data.py` and `weather_features.py` were rerun, and `frankfurt_munich/README.md` describes the new rules and numbers. The advanced inputs in `frankfurt_munich/data/advanced/` were built for the old rows and are rebuilt in B11.
- 13 November 2026: B11, B1 and B8 done. All CV inputs of the advanced models rebuilt from the corrected dataset (about 30 minutes); coverage in `advanced_modeling/README.md`. B1: the snapshot `current_delay` is sparse because live updates were fetched in blocks about six hours apart, so at most cutoffs no report of the run had been fetched yet; the paragraph is in the report draft, and Q5 is open again. Noticed in B1, for the group: at 0 min, 4,132 of the 8,937 snapshot values come from the Frankfurt departure itself (a live update fetched before the cutoff that shows the train leaving on time or early). That is known at the cutoff, so it is not leakage, but the final-record version no longer uses that departure (R7), so the two versions differ in this. Nothing is changed. The CV results and the hold-out inputs are still the old ones; next is the rerun of the CV arms (B4).
- 13 November 2026: decided that the snapshot `current_delay` does not use the departure from Frankfurt itself either, as for the final-record version (R7). The delay-history aggregates (`current_delay`, the reported-stop count and the time since the last report) leave that departure out; the sequence and transfer inputs already did. Delay history rebuilt: at 0 min, 411 CV values changed and 179 became missing (now 8,758). Test added.
