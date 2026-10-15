"""Evaluation metrics, with one fixed class order.

Probability matrices always have their columns in CLASSES order. Log loss is
computed here rather than with sklearn.metrics.log_loss, which assumes the
columns are in alphabetical label order.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

CLASSES = ["normal_minor", "moderate", "severe", "canceled"]
EPS = 1e-15


def class_log_loss(y_true, proba) -> float:
    """Mean negative log probability of the true class (columns in CLASSES order)."""
    index = np.array([CLASSES.index(y) for y in y_true])
    p = np.clip(np.asarray(proba, dtype=float)[np.arange(len(index)), index], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def macro_f1(y_true, proba) -> float:
    """Macro-F1 over the four classes, predicting the most probable class."""
    predicted = np.array(CLASSES)[np.asarray(proba).argmax(axis=1)]
    return float(f1_score(list(y_true), predicted, labels=CLASSES,
                          average="macro", zero_division=0))


def rmse(y_true, y_pred) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(error ** 2)))
