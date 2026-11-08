"""Static calibration and journey-risk plots for the results directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import _paths  # noqa: F401  shared modelling code and dataset builders
from data import PREDICTION_TIMES
from metrics import CLASSES

matplotlib.rcParams["svg.hashsalt"] = "stormytrains"


def save_plot(fig, path):
    path = Path(path)
    metadata = {"Date": None} if path.suffix == ".svg" else None
    fig.savefig(path, metadata=metadata)
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def calibration_plot(tables, path):
    """Plot nonempty reliability bins for each horizon and severity class."""
    fig, axes = plt.subplots(len(PREDICTION_TIMES), len(CLASSES), figsize=(10, 11),
                             sharex=True, sharey=True, squeeze=False)
    for i, pt in enumerate(PREDICTION_TIMES):
        for j, label in enumerate(CLASSES):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="0.75", linewidth=0.8, linestyle="--")
            for arm, table in tables.items():
                subset = table.loc[table["prediction_time"].eq(pt) & table["class"].eq(label)
                                   & table["n_rows"].gt(0)]
                ax.plot(subset["mean_predicted_probability"], subset["observed_frequency"],
                        marker=".", linewidth=1.0, label=arm)
            if i == 0:
                ax.set_title(label.replace("_", " "))
            if j == 0:
                ax.set_ylabel(pt)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    axes[0, -1].legend(loc="upper left", fontsize=7)
    fig.supxlabel("Mean predicted probability")
    fig.supylabel("Observed frequency")
    fig.tight_layout()
    save_plot(fig, path)


def risk_plot(tables, path):
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(10, 3), sharex=True, sharey=True)
    for ax, label in zip(axes, CLASSES):
        for arm, table in tables.items():
            subset = table.loc[table["severity_class"].eq(label)].set_index("prediction_time")
            if not subset.empty:
                ax.plot(range(len(PREDICTION_TIMES)),
                        subset["p_severe_or_canceled"].reindex(PREDICTION_TIMES),
                        marker="o", linewidth=1.2, label=arm)
        ax.set_xticks(range(len(PREDICTION_TIMES)), PREDICTION_TIMES, rotation=45)
        ax.set_title(label.replace("_", " "))
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Severe or cancelled probability")
    axes[-1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    save_plot(fig, path)
