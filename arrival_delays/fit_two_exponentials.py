"""Fit a two-sided exponential to the terminus arrival delays.

Model: with probability p the delay is Exp(rate_pos) above 0, otherwise it is
minus Exp(rate_neg). Delays are whole minutes, so the probability of minute k
is the model mass on [k - 0.5, k + 0.5]; minute 0 gets mass from both sides.
p, rate_pos and rate_neg are fitted by maximum likelihood on these bins.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "terminus_arrivals.parquet"
THRESHOLDS = [0, 20, 60]
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def bin_probs(k: np.ndarray, p: float, rate_pos: float, rate_neg: float) -> np.ndarray:
    k = k.astype(float)
    a = np.abs(k)
    pos = p * (np.exp(-rate_pos * (a - 0.5)) - np.exp(-rate_pos * (a + 0.5)))
    neg = (1 - p) * (np.exp(-rate_neg * (a - 0.5)) - np.exp(-rate_neg * (a + 0.5)))
    zero = p * (1 - np.exp(-rate_pos / 2)) + (1 - p) * (1 - np.exp(-rate_neg / 2))
    return np.where(k > 0, pos, np.where(k < 0, neg, zero))


def fit(values: np.ndarray) -> dict:
    k, n = np.unique(values, return_counts=True)

    def nll(theta: np.ndarray) -> float:
        p = 1 / (1 + np.exp(-theta[0]))
        return -np.sum(n * np.log(bin_probs(k, p, *np.exp(theta[1:]))))

    start = np.array([0.0, -np.log(values[values > 0].mean()),
                      -np.log(-values[values < 0].mean())])
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 5000})
    p = float(1 / (1 + np.exp(-res.x[0])))
    rate_pos, rate_neg = map(float, np.exp(res.x[1:]))
    return {"p": p, "mean_pos_min": 1 / rate_pos, "mean_neg_min": 1 / rate_neg,
            "rate_pos": rate_pos, "rate_neg": rate_neg}


def compare(values: np.ndarray, params: dict) -> dict:
    lo, hi = int(values.min()), int(values.max())
    grid = np.arange(lo - 200, hi + 1000)
    model_pmf = bin_probs(grid, params["p"], params["rate_pos"], params["rate_neg"])
    emp_pmf = pd.Series(values).value_counts(normalize=True).reindex(grid, fill_value=0).to_numpy()
    model_cdf, emp_cdf = np.cumsum(model_pmf), np.cumsum(emp_pmf)
    ks_at = int(grid[np.argmax(np.abs(model_cdf - emp_cdf))])

    def model_quantile(q: float) -> int:
        return int(grid[np.searchsorted(model_cdf, q)])

    def share(pmf: np.ndarray, cond: np.ndarray) -> float:
        return float(pmf[cond].sum())

    probs = {"P(<0)": (grid < 0), "P(=0)": (grid == 0)}
    probs.update({f"P(>={t})": grid >= t for t in THRESHOLDS if t > 0})
    return {
        "n": int(len(values)),
        "ks_distance": float(np.abs(model_cdf - emp_cdf).max()),
        "ks_at_min": ks_at,
        "total_variation": float(0.5 * np.abs(model_pmf - emp_pmf).sum()),
        "probabilities": {name: {"data": share(emp_pmf, c), "model": share(model_pmf, c)}
                          for name, c in probs.items()},
        "quantiles": {str(q): {"data": int(np.quantile(values, q, method="inverted_cdf")),
                               "model": model_quantile(q)} for q in QUANTILES},
        "mean": {"data": float(values.mean()),
                 "model": params["p"] * params["mean_pos_min"]
                 - (1 - params["p"]) * params["mean_neg_min"]},
    }


def plot(values: np.ndarray, params: dict, path: Path) -> None:
    ink, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    data_c, model_c = "#2a78d6", "#eb6834"
    k = np.arange(-40, 181)
    emp = pd.Series(values).value_counts(normalize=True).reindex(k, fill_value=0)
    model = bin_probs(k, params["p"], params["rate_pos"], params["rate_neg"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb")
    for ax, log in zip(axes, [False, True]):
        ax.set_facecolor("#fcfcfb")
        view = k <= (60 if not log else 180)
        ax.bar(k[view], emp.to_numpy()[view], width=0.8, color=data_c, alpha=0.55,
               label="Observed (share of runs per minute)")
        ax.plot(k[view], model[view], color=model_c, lw=2, label="Two-sided exponential fit")
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 0.2)
        ax.set_xlabel("Terminus arrival delay (min)", color=muted)
        ax.set_title("Linear scale, −40 to 60 min" if not log
                     else "Log scale, −40 to 180 min (tails)", color=ink, loc="left", fontsize=11)
        ax.grid(axis="y", color=grid_c, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(grid_c)
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("Share of runs", color=muted)
    axes[0].legend(frameon=False, labelcolor=ink)
    fig.suptitle("ICE/IC terminus arrival delay vs two-sided exponential fit",
                 color=ink, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    frame = pd.read_parquet(DATA)
    out = {}
    for name, sub in [("all", frame), ("ICE", frame[frame.train_type == "ICE"]),
                      ("IC", frame[frame.train_type == "IC"])]:
        values = sub.delay_min.to_numpy(dtype=int)
        params = fit(values)
        out[name] = {"params": params, "fit": compare(values, params)}
        if name == "all":
            plot(values, params, HERE / "two_exponential_fit.png")
    (HERE / "two_exponential_fit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
