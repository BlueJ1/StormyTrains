import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gated_neural
from gated_neural import GatedLayer, GatedMLP
from neural import TorchMLP


def test_width_follows_rounded_sigmoid_mask():
    layer = GatedLayer(2, 5)
    with torch.no_grad():
        layer.strength.copy_(torch.tensor([-1.0, 0.0, 0.2, 0.8, 1.2]))
    assert layer.selected_width() == 3


def test_eval_gates_are_deterministic_with_one_neuron_fallback():
    layer = GatedLayer(2, 3).eval()
    with torch.no_grad():
        layer.strength.copy_(torch.tensor([-3.0, -1.0, -2.0]))
        layer.linear.weight.fill_(1)
        layer.linear.bias.zero_()
        features = torch.ones(1, 2)
        expected = torch.tensor([[0.0, 2.0, 0.0]])
        torch.testing.assert_close(layer(features), expected)
        torch.testing.assert_close(layer(features), expected)
    assert layer.selected_width() == 1


def test_refit_epoch_is_selected_on_inner_weeks(monkeypatch):
    X = pd.DataFrame({"feature": np.arange(8, dtype=float)})
    y = np.arange(8, dtype=float)
    train, validation = np.arange(6), np.arange(6, 8)
    losses = iter([1.0, 1.0, 1.0, 3.0, 1.0, 2.0])
    dense_batches, refits = [], []

    def score(_output, _classes, _delays, batch):
        assert batch.tolist() == validation.tolist()
        return torch.tensor(next(losses))

    def update(_self, _model, optimizer, _features, _classes, _delays, batch):
        if isinstance(optimizer, torch.optim.AdamW):
            dense_batches.append(sorted(batch.tolist()))

    def refit(self, rows, target):
        assert rows is X and target is y
        refits.append(self.epochs)
        return self

    monkeypatch.setattr(gated_neural, "MAX_EPOCHS", 3)
    monkeypatch.setattr(gated_neural, "week_validation", lambda *_: (train, validation))
    monkeypatch.setattr(GatedMLP, "loss", staticmethod(score))
    monkeypatch.setattr(GatedMLP, "_update", update)
    monkeypatch.setattr(TorchMLP, "fit", refit)

    GatedMLP(["feature"], [], task="delay", cpu_budget=60).fit(X, y)

    assert dense_batches == [train.tolist()] * 3
    assert refits == [2]
