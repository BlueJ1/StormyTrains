"""Figures for the report, drawn from the committed result files.

Run from the repository root:

    .venv/bin/python report/figures/make_figures.py

Fig. 1 (fig_scores.pdf): cross-validation score at each prediction time.
Fig. 2 (fig_weather.pdf): hold-out difference, railway plus weather minus railway only,
with the paired week-bootstrap interval.

Only saved summaries are read; no model is fitted and no new score is computed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "frankfurt_munich" / "advanced_modeling" / "results"
OUT = Path(__file__).resolve().parent

TIMES = ["24h", "6h", "1h", "20min", "0min"]
LABELS = ["24 h", "6 h", "1 h", "20 min", "0 min"]

# Categorical colours in fixed order (checked for colour-blind separation);
# markers and line styles repeat the identity for greyscale print.
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d2"
SERIES = [
    ("#52514e", "o", ":"),
    ("#2a78d6", "s", "--"),
    ("#eb6834", "^", "-."),
    ("#1baf7a", "D", "-"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "pdf.fonttype": 42,
})


def latest(name):
    """The most recent completed run of an experiment arm, as report_cv.run_index
    selects it: one with both config.json and summary.csv."""
    runs = sorted(p for p in (RESULTS / name).iterdir()
                  if (p / "config.json").is_file() and (p / "summary.csv").is_file())
    return runs[-1]


def cv_curve(name, column):
    summary = pd.read_csv(latest(name) / "summary.csv").set_index("prediction_time")
    return summary.loc[TIMES, column].to_numpy()


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def fig_scores():
    panels = [
        ("Severity: log loss", "log_loss_mean", [
            ("Class shares", "advanced_0_7_class_shares"),
            ("Logistic regression, base", "advanced_0_7_logistic_regression"),
            ("LightGBM, base", "advanced_0_8_lightgbm_classifier"),
            ("LightGBM, selected groups", "advanced_0_8_lightgbm_classifier_auto_groups"),
        ]),
        ("Delay: RMSE (min)", "rmse_mean", [
            ("Training mean", "advanced_0_7_training_mean"),
            ("Linear regression, base", "advanced_0_7_linear_regression"),
            ("LightGBM, base", "advanced_0_8_lightgbm"),
            ("LightGBM, selected groups", "advanced_0_8_lightgbm_auto_groups"),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = range(len(TIMES))
    for ax, (title, column, arms) in zip(axes, panels):
        for (label, name), (colour, marker, line) in zip(arms, SERIES):
            ax.plot(x, cv_curve(name, column), color=colour, marker=marker, linestyle=line,
                    linewidth=1.4, markersize=4, label=label)
        ax.set_xticks(list(x), LABELS)
        ax.set_xlabel("Time before scheduled departure")
        ax.set_title(title, fontsize=8.5, loc="left")
        style(ax)
        # Below the plot, so that the legend never covers a line.
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                  ncol=2, handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_scores.pdf")
    plt.close(fig)


def fig_weather():
    holdout = RESULTS / "advanced_holdout"
    panels = [
        ("Severity: change in log loss", holdout / "severity_weather.csv"),
        ("Delay: change in RMSE (min)", holdout / "delay_weather.csv"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    x = range(len(TIMES))
    for ax, (title, path) in zip(axes, panels):
        paired = pd.read_csv(path).set_index("prediction_time").loc[TIMES]
        lower = paired["delta"] - paired["ci_lower"]
        upper = paired["ci_upper"] - paired["delta"]
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.errorbar(list(x), paired["delta"], yerr=[lower, upper], fmt="o", color="#2a78d6",
                    ecolor="#2a78d6", elinewidth=1.2, capsize=2.5, markersize=4)
        ax.set_xticks(list(x), LABELS)
        ax.set_xlabel("Time before scheduled departure")
        ax.set_title(title, fontsize=8.5, loc="left")
        style(ax)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_weather.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_scores()
    fig_weather()
    print(f"Wrote fig_scores.pdf and fig_weather.pdf to {OUT}")
