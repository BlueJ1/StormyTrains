from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distributions import (  # noqa: E402
    make_lightgbmlss,
    make_ngboost,
    make_quantile_forest,
)


def _data(n=48):
    x = np.linspace(0, 1, n)
    X = pd.DataFrame({"x": x, "route": np.where(np.arange(n) % 2, "A", "B")})
    y = 8 + 25 * x + 3 * np.sin(7 * x)
    return X, y


def test_ngboost_mean_and_continuous_cdf_match_fitted_distribution():
    pytest.importorskip("ngboost")
    X, y = _data()
    model = make_ngboost(
        ["x"], ["route"], n_estimators=8, learning_rate=0.03,
        base_max_depth=2, random_state=7,
    ).fit(X, y)

    sample = X.iloc[:6]
    assert model.predict(sample).shape == (len(sample),)
    assert model.predict(sample) == pytest.approx(model.model_.predict(model._transform(sample)))
    cdf = model.predict_cdf(sample, 20)
    expected = model.model_.pred_dist(model._transform(sample)).cdf(np.full(len(sample), 20.0))
    assert cdf == pytest.approx(expected)
    assert np.all((0 <= cdf) & (cdf <= 1))


def test_quantile_forest_mean_and_strict_cdf_handle_ties():
    pytest.importorskip("quantile_forest")
    X = pd.DataFrame({"x": np.zeros(8), "route": ["A"] * 8})
    y = np.array([0.0, 0.0, 20.0, 20.0, 60.0, 60.0, 60.0, 90.0])
    model = make_quantile_forest(
        ["x"], ["route"], n_estimators=12, min_samples_leaf=1, random_state=2,
    ).fit(X, y)

    assert model.predict(X.iloc[:2]) == pytest.approx(
        model.model_.predict(model._transform(X.iloc[:2]), quantiles="mean")
    )
    sample = X.iloc[:2]
    strict = model.predict_cdf(sample, 20)
    weak = model.model_.quantile_ranks(
        model._transform(sample), np.full(len(sample), 20.0), kind="weak"
    )
    assert strict == pytest.approx(model.model_.quantile_ranks(
        model._transform(sample), np.full(len(sample), 20.0), kind="strict"
    ))
    assert np.all(strict < weak)
    assert np.all((0 <= strict) & (strict <= 1))


def test_quantile_forest_keeps_rows_with_missing_current_delay():
    pytest.importorskip("quantile_forest")
    X, y = _data()
    X.loc[::2, "x"] = np.nan
    X["all_missing"] = np.nan
    model = make_quantile_forest(["x", "all_missing"], ["route"], n_estimators=8).fit(X, y)
    assert np.isfinite(model.predict(X)).all()


def test_lightgbmlss_mixture_mean_and_strict_continuous_cdf_are_consistent():
    pytest.importorskip("lightgbmlss")
    torch = pytest.importorskip("torch")
    X, y = _data()
    model = make_lightgbmlss(
        ["x"], ["route"], n_estimators=6, learning_rate=0.05,
        num_leaves=5, min_data_in_leaf=5, mixture_components=2,
    ).fit(X, y)

    sample = X.iloc[:6]
    distribution = model._predict_distribution(sample)
    mean = model.predict(sample)
    cdf_20 = model.predict_cdf(sample, 20)
    cdf_60 = model.predict_cdf(sample, 60)

    assert model.distribution_.M == 2
    assert mean == pytest.approx(distribution.mean.detach().cpu().numpy())
    assert cdf_20 == pytest.approx(distribution.cdf(
        torch.full(distribution.mean.shape, 20.0, dtype=distribution.mean.dtype)
    ).detach().cpu().numpy())
    assert np.all((0 <= cdf_20) & (cdf_20 <= cdf_60) & (cdf_60 <= 1))
