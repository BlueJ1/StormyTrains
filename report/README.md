# Report draft

First draft of the final report (10 November 2026). Deadline: 11 January 2027. Requirements are in `../ProjectInstructions.md`, "Final report": a short conference paper in English, five pages excluding tables, figures and references, in the course template (LaTeX preferred).

## Files

| File | Contents |
|---|---|
| `main.tex` | The draft, in the course LaTeX conference template (Canvas, Templates page: `conference_101719.tex`) |
| `IEEEtran.cls` | The class file from the course template (V1.8b), so the build does not depend on the installed version |
| `references.bib` | BibTeX entries. Entries marked `CHECK` have not been compared with the publication |
| `figures/make_figures.py` | Draws both figures from the committed result files; fits no model and computes no new score |
| `figures/fig_scores.pdf`, `figures/fig_weather.pdf` | Fig. 1 and Fig. 2 |

## Build

```bash
.venv/bin/python report/figures/make_figures.py
cd report && latexmk -pdf main.tex
```

`main.pdf` and the LaTeX build files are not committed.

## Open points in the text

Every `\todo{...}` in `main.tex` is something the group has to decide, check or fill in. Search for `\todo`. The larger ones are also steps in `../FINISHING_PLAN.md`.

## Where the numbers come from

Every number in the draft is from before the corrections decided on 12 November 2026 (`../FINISHING_PLAN.md`, Q1 and Q2). They are replaced after the CV rerun on the corrected inputs (B4) and the second hold-out evaluation (B6). The first models of 15–31 October are not mentioned in the report (Q9).

Paths are relative to `frankfurt_munich/`.

| In the report | Source |
|---|---|
| Journey counts, weeks, folds, class shares, delay summary, cancellation announcement times | `README.md` (from `audit.json`, `build_audit.json`, `change_check.json`); recomputed from the parquet files in the review of 10 November 2026 |
| Table I, rows and cancelled share | `README.md`, "When cancellations were issued" |
| Table I, delay in final records | `README.md`, "current_delay" (share of prediction rows with a value) |
| Table I, delay in archive | `advanced_modeling/README.md`, input coverage table |
| Table II and Fig. 1, CV rows | `advanced_modeling/results/advanced_0_7_*` (baselines, logistic and linear regression) and `advanced_0_8_lightgbm*` (LightGBM with nested tuning, on base inputs and on automatically chosen feature groups, `auto_groups`), latest run of each, `summary.csv` |
| Table II, hold-out rows | `advanced_modeling/results/advanced_holdout/scores.csv` |
| Table III and Fig. 2 | `advanced_modeling/results/advanced_holdout/severity_weather.csv`, `delay_weather.csv` |
| Feature-group, model-family and ensemble counts | `advanced_modeling/PLAN.md`, result lines of Phases 1 to 8 |
| Errors by class, risk example | `advanced_modeling/results/advanced_holdout/delay_errors.csv`, `risk_railway.csv` |
| Forecast comparisons (CV), forecast inputs and availability | `advanced_modeling/README.md` (table of the forecast CV runs, B17; all pairs in `results/forecast_arm_comparisons.csv`, linear models and boosters, without excluded rows); `forecast_weather_GFS/README.md`. These numbers are from the corrected inputs (24 November 2026) |
