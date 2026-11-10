# Report draft

First draft of the final report (10 November 2026). Deadline: 11 January 2027. Requirements are in `../ProjectInstructions.md`, "Final report": a short conference paper in English, five pages excluding tables, figures and references, in the course template (LaTeX preferred).

## Files

| File | Contents |
|---|---|
| `main.tex` | The draft. It uses the standard IEEE conference class as a stand-in for the course template on Canvas (Templates page); move the text into that template before submission |
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

Paths are relative to `frankfurt_munich/`.

| In the report | Source |
|---|---|
| Journey counts, weeks, folds, class shares, delay summary, cancellation announcement times | `README.md` (from `audit.json`, `build_audit.json`, `change_check.json`); recomputed from the parquet files in the review of 10 November 2026 |
| Table I, rows and cancelled share | `README.md`, "When cancellations were issued" |
| Table I, delay in final records | `README.md`, "current_delay" (share of prediction rows with a value) |
| Table I, delay in archive | `advanced_modeling/README.md`, input coverage table |
| Table II and Fig. 1, CV rows | `advanced_modeling/results/advanced_0_7_*` (references, logistic and linear regression) and `advanced_0_8_lightgbm*` (nested LightGBM, base and `auto_groups`), latest run of each, `summary.csv` |
| Table II, hold-out rows | `advanced_modeling/results/advanced_holdout/scores.csv` |
| Table III and Fig. 2 | `advanced_modeling/results/advanced_holdout/severity_weather.csv`, `delay_weather.csv` |
| Final-record scores (0.810, 22.6 min at 1 h) | `modeling/README.md`, tuned LightGBM, 26 and 31 October 2026 |
| Feature-group, model-family and ensemble counts | `advanced_modeling/PLAN.md`, result lines of Phases 1 to 8 |
| Errors by class, risk example | `advanced_modeling/results/advanced_holdout/delay_errors.csv`, `risk_railway.csv` |
