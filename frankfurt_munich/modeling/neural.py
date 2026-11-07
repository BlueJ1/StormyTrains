"""CPU tabular networks with training-fold preprocessing."""
from __future__ import annotations

from time import process_time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from data import TIME_COLUMN
from folds import N_FOLDS, SEED, week_of
from metrics import CLASSES
from models import linear_preprocessing
from representations import REPRESENTATIONS


def dense(array):
    return np.asarray(array.toarray() if hasattr(array, "toarray") else array, dtype=np.float32)


def week_validation(X, y, task):
    """A deterministic validation split wholly inside the rows supplied to fit."""
    n_splits = N_FOLDS - 1
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)
    strata = y if task == "severity" else np.digitize(y, [20, 60])
    return next(splitter.split(X, strata, week_of(X[TIME_COLUMN])))


def preprocessor(numeric, categorical, calendar):
    numeric = [column for column in numeric if column != TIME_COLUMN]
    if calendar:
        factory, num, cat = REPRESENTATIONS[calendar].wrap(linear_preprocessing, numeric, categorical)
        return factory(num, cat)
    return linear_preprocessing(numeric, categorical)


def perceptron(n_inputs, widths, n_outputs, dropout):
    layers = []
    for width in widths:
        layers.extend([nn.Linear(n_inputs, width), nn.ReLU(), nn.Dropout(dropout)])
        n_inputs = width
    return nn.Sequential(*layers, nn.Linear(n_inputs, n_outputs))


class TorchMLP:
    """A shared implementation for single-task and joint delay/severity prediction."""

    def __init__(self, numeric, categorical, *, task, depth, width, dropout, weight_decay,
                 learning_rate, epochs, calendar="", widths=None, cpu_budget=None):
        self.numeric, self.categorical = numeric, categorical
        self.task = task
        self.widths = [width] * depth if widths is None else widths
        self.dropout, self.weight_decay = dropout, weight_decay
        self.learning_rate, self.epochs = learning_rate, epochs
        self.calendar = calendar
        self.cpu_budget = cpu_budget

    def fit(self, X, y):
        return self._fit(X, y if self.task == "severity" else None,
                         y if self.task == "delay" else None)

    def _fit(self, X, severity, delay):
        start = process_time()
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        self.preprocess_ = preprocessor(self.numeric, self.categorical, self.calendar)
        features = self.input_tensor(X, fit=True)
        classes, delays = self.targets(severity, delay)
        n_classes = len(CLASSES) if classes is not None else 0
        self.model_ = self.network(features.shape[1], n_classes + int(delays is not None))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        self.epochs_ = 0
        for _ in range(self.epochs):
            if self.epochs_ and self.cpu_budget is not None and process_time() - start >= self.cpu_budget:
                break
            for batch in torch.randperm(len(X)).split(256):
                optimizer.zero_grad()
                output = self.model_(features[batch])
                loss = self.loss(output, classes, delays, batch)
                loss.backward()
                optimizer.step()
                if self.cpu_budget is not None and process_time() - start >= self.cpu_budget:
                    break
            self.epochs_ += 1
        self.n_parameters_ = sum(parameter.numel() for parameter in self.model_.parameters())
        self.cpu_seconds_ = process_time() - start
        return self

    def input_tensor(self, X, *, fit=False):
        transformed = self.preprocess_.fit_transform(X) if fit else self.preprocess_.transform(X)
        return torch.from_numpy(dense(transformed))

    def network(self, n_inputs, n_outputs):
        return perceptron(n_inputs, self.widths, n_outputs, self.dropout)

    def targets(self, severity, delay):
        classes, delays = None, None
        if severity is not None:
            self.classes_ = np.asarray(CLASSES)
            classes = torch.tensor([CLASSES.index(label) for label in severity], dtype=torch.long)
        if delay is not None:
            values = np.asarray(delay, dtype=np.float32)
            self.delay_mean_ = float(np.nanmean(values))
            self.delay_scale_ = max(float(np.nanstd(values)), 1.0)
            delays = torch.tensor((values - self.delay_mean_) / self.delay_scale_)
        return classes, delays

    @staticmethod
    def loss(output, classes, delays, batch):
        loss = 0.0
        if classes is not None:
            loss = nn.functional.cross_entropy(output[:, :len(CLASSES)], classes[batch])
        if delays is not None:
            observed = torch.isfinite(delays[batch])
            if observed.any():
                loss = loss + nn.functional.mse_loss(output[observed, -1], delays[batch][observed])
        return loss

    def outputs(self, X):
        self.model_.eval()
        features = self.input_tensor(X)
        with torch.inference_mode():
            return self.model_(features)

    def predict_proba(self, X):
        return self.outputs(X)[:, :len(CLASSES)].softmax(dim=1).numpy()

    def predict(self, X):
        if self.task == "severity":
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        return self.outputs(X)[:, -1].numpy() * self.delay_scale_ + self.delay_mean_


class MultiTaskMLP(TorchMLP):
    def fit_joint(self, X, severity, delay):
        return self._fit(X, severity, delay)


class TabKitModel:
    """Pass a week-held validation set to RealMLP or TabM."""

    def __init__(self, numeric, categorical, *, task, family, calendar="", epochs=64):
        self.numeric, self.categorical = numeric, categorical
        self.task, self.family = task, family
        self.calendar, self.epochs = calendar, epochs

    def fit(self, X, y):
        from pytabkit import (RealMLP_TD_Classifier, RealMLP_TD_Regressor, TabM_D_Classifier,
                              TabM_D_Regressor)

        factories = {"realmlp": (RealMLP_TD_Regressor, RealMLP_TD_Classifier),
                     "tabm": (TabM_D_Regressor, TabM_D_Classifier)}
        classification = self.task == "severity"
        train, validation = week_validation(X, y, self.task)
        self.preprocess_ = preprocessor(self.numeric, self.categorical, self.calendar).fit(X.iloc[train])
        values = dense(self.preprocess_.transform(X))
        factory = factories[self.family][classification]
        metric = "cross_entropy" if classification else "rmse"
        # PyTabKit 1.8 exposes full-training refit for RealMLP only.
        self.model_ = factory(n_threads=1, device="cpu", random_state=SEED, n_epochs=self.epochs,
                              n_cv=1, n_refit=int(self.family == "realmlp"), val_metric_name=metric)
        self.model_.fit(values, np.asarray(y), val_idxs=validation)
        if classification:
            self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(dense(self.preprocess_.transform(X)))

    def predict(self, X):
        return self.model_.predict(dense(self.preprocess_.transform(X)))
