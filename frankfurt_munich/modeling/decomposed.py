"""Phase 4 estimators composed of existing classifier and regressor factories."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier

from metrics import CLASSES

CANCELED = "canceled"
COMPLETED = [label for label in CLASSES if label != CANCELED]


def fit_classifier(factory, columns, X, y):
    model = DummyClassifier(strategy="prior") if len(np.unique(y)) == 1 else factory(*columns)
    return model.fit(X, y)


def probabilities(model, X, labels):
    order = list(model.classes_)
    values = np.asarray(model.predict_proba(X), dtype=float)
    result = np.zeros((len(X), len(labels)))
    for i, label in enumerate(labels):
        if label in order:
            result[:, i] = values[:, order.index(label)]
    return result


class CancellationGate(ClassifierMixin, BaseEstimator):
    """Shared cancellation model for the three ways of modeling completed journeys."""

    def __init__(self, numeric, categorical, classifier_factory):
        self.numeric = numeric
        self.categorical = categorical
        self.classifier_factory = classifier_factory

    def _fit_gate(self, X, y):
        labels = np.asarray(y)
        if len(labels) != len(X) or not set(labels) <= set(CLASSES):
            raise ValueError("severity labels must align with the rows and use the four classes")
        completed = labels != CANCELED
        if not completed.any():
            raise ValueError("stage two needs completed training journeys")
        self.cancellation_model_ = self._fit_classifier(X, ~completed)
        self.classes_ = np.asarray(CLASSES)
        return labels, completed

    def _fit_classifier(self, X, y):
        return fit_classifier(self.classifier_factory, (self.numeric, self.categorical), X, y)

    def predict_proba(self, X):
        p_cancel = probabilities(self.cancellation_model_, X, [True])
        conditional = self._predict_completed(X)
        if (not np.isfinite(conditional).all() or (conditional < 0).any()
                or not np.allclose(conditional.sum(axis=1), 1)):
            raise ValueError("completed-delay probabilities must be finite, nonnegative and sum to one")
        joint = np.column_stack(((1 - p_cancel) * conditional, p_cancel))
        return joint[:, [(COMPLETED + [CANCELED]).index(label) for label in CLASSES]]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CancellationThenClassClassifier(CancellationGate):
    def fit(self, X, y):
        labels, completed = self._fit_gate(X, y)
        self.completed_class_model_ = self._fit_classifier(X.iloc[completed], labels[completed])
        return self

    def _predict_completed(self, X):
        return probabilities(self.completed_class_model_, X, COMPLETED)


class CancellationThenDistribution(CancellationGate):
    """The delay estimator supplies a mean and strict P(delay < cutoff)."""

    def __init__(self, numeric, categorical, classifier_factory, delay_factory):
        super().__init__(numeric, categorical, classifier_factory)
        self.delay_factory = delay_factory

    def fit(self, X, y, *, delay_y):
        _, completed = self._fit_gate(X, y)
        delays = np.asarray(delay_y, dtype=float)
        if len(delays) != len(X):
            raise ValueError("delay target must align with the rows")
        observed = completed & np.isfinite(delays)
        if not observed.any():
            raise ValueError("stage two needs observed completed delays")
        self.delay_model_ = self.delay_factory(self.numeric, self.categorical)
        self.delay_model_.fit(X.iloc[observed], delays[observed])
        return self

    def fit_joint(self, X, severity, delay):
        return self.fit(X, severity, delay_y=delay)

    def _predict_completed(self, X):
        p20 = np.asarray(self.delay_model_.predict_cdf(X, 20), dtype=float)
        p60 = np.asarray(self.delay_model_.predict_cdf(X, 60), dtype=float)
        if (p60 < p20 - 1e-7).any():
            raise ValueError("delay CDF must be nondecreasing across thresholds")
        if (p20 < -1e-7).any() or (p60 > 1 + 1e-7).any():
            raise ValueError("delay CDF must be in [0, 1]")
        p20 = p20.clip(0, 1)
        p60 = np.maximum(p20, p60.clip(0, 1))
        return np.column_stack((p20, p60 - p20, 1 - p60))

    def predict(self, X):
        return self.delay_model_.predict(X)


class CancellationThenOrdinal(CancellationGate):
    def fit(self, X, y):
        labels, completed = self._fit_gate(X, y)
        self.at_least_20_model_ = self._fit_classifier(X.iloc[completed], labels[completed] != COMPLETED[0])
        self.at_least_60_model_ = self._fit_classifier(X.iloc[completed], labels[completed] == COMPLETED[2])
        return self

    def _predict_completed(self, X):
        p20 = probabilities(self.at_least_20_model_, X, [True])[:, 0]
        p60 = probabilities(self.at_least_60_model_, X, [True])[:, 0]
        crossed = p60 > p20
        p20[crossed] = p60[crossed] = (p20[crossed] + p60[crossed]) / 2
        return np.column_stack((1 - p20, p20 - p60, p60))


class MixtureOfExpertsRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, numeric, categorical, classifier_factory, regressor_factory):
        self.numeric = numeric
        self.categorical = categorical
        self.classifier_factory = classifier_factory
        self.regressor_factory = regressor_factory

    def fit(self, X, y):
        delays = np.asarray(y, dtype=float)
        if len(delays) != len(X) or not np.isfinite(delays).all():
            raise ValueError("mixture-of-experts needs one finite completed delay per row")
        classes = np.asarray(COMPLETED)[np.digitize(delays, [20, 60])]
        columns = (self.numeric, self.categorical)
        self.classifier_ = fit_classifier(self.classifier_factory, columns, X, classes)
        self.regressors_ = []
        for label in COMPLETED:
            selected = classes == label
            if not selected.any():
                raise ValueError(f"training data contain no rows for delay class {label!r}")
            self.regressors_.append(self.regressor_factory(*columns).fit(X.iloc[selected], delays[selected]))
        return self

    def predict(self, X):
        class_shares = probabilities(self.classifier_, X, COMPLETED)
        means = np.column_stack([model.predict(X) for model in self.regressors_])
        return np.sum(class_shares * means, axis=1)


class ChangeThenLevelRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, numeric, categorical, regressor_factory, current_delay_column="current_delay"):
        self.numeric = numeric
        self.categorical = categorical
        self.regressor_factory = regressor_factory
        self.current_delay_column = current_delay_column

    def fit(self, X, y):
        columns = (self.numeric, self.categorical)
        delays = np.asarray(y, dtype=float)
        current = X[self.current_delay_column].to_numpy(dtype=float)
        observed = np.isfinite(current)
        self.level_model_ = self.regressor_factory(*columns).fit(X, delays)
        self.change_model_ = None
        if observed.any():
            self.change_model_ = self.regressor_factory(*columns).fit(
                X.iloc[observed], delays[observed] - current[observed])
        return self

    def predict(self, X):
        current = X[self.current_delay_column].to_numpy(dtype=float)
        observed = np.isfinite(current)
        prediction = self.level_model_.predict(X).copy()
        if observed.any() and self.change_model_ is not None:
            prediction[observed] = current[observed] + self.change_model_.predict(X.iloc[observed])
        return prediction


make_cancellation_then_class = CancellationThenClassClassifier
make_cancellation_then_distribution = CancellationThenDistribution
make_cancellation_then_ordinal = CancellationThenOrdinal
make_mixture_of_experts = MixtureOfExpertsRegressor
make_change_then_level = ChangeThenLevelRegressor
