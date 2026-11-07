"""TabGNS-style bilevel search using week folds and a fixed CPU budget.

Algorithm reference: Orucu et al. (2026), https://arxiv.org/abs/2602.10854.
The hard Gumbel gates and alternating updates follow the authors' implementation
at EricssonResearch/tabgns, commit 1a6a135649f5b06204323c6f8be8e7324472a5a5.
The inner weeks select the widths and the dense network's refit epoch count.
The selected network is then initialized afresh and fitted on all outer
training weeks.
"""
from __future__ import annotations

from copy import deepcopy
from time import process_time

import numpy as np
import torch
from torch import nn

from folds import SEED
from metrics import CLASSES
from neural import TorchMLP, dense, preprocessor, week_validation

OFF_LOGIT = 1.0
MAX_EPOCHS = 300
PATIENCE = 20


class GatedLayer(nn.Module):
    def __init__(self, n_inputs, width):
        super().__init__()
        self.linear = nn.Linear(n_inputs, width)
        self.strength = nn.Parameter(torch.full((width,), -3.0))

    def forward(self, features):
        if self.training:
            logits = torch.stack((self.strength, torch.full_like(self.strength, OFF_LOGIT)), dim=1)
            gates = nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)[:, 0]
        else:
            gates = self.selected_mask()
        return self.linear(features).relu() * gates

    def selected_mask(self):
        mask = self.strength.sigmoid().round()
        if not mask.any():
            mask[self.strength.argmax()] = 1
        return mask

    def selected_width(self):
        """Count the neurons selected by the reference rounded sigmoid mask."""
        return int(self.selected_mask().sum())


class GatedMLP(TorchMLP):
    def __init__(self, numeric, categorical, *, task, cpu_budget, calendar=""):
        super().__init__(numeric, categorical, task=task, depth=5, width=512, dropout=0.0,
                         weight_decay=0.0, learning_rate=0.001, epochs=MAX_EPOCHS, calendar=calendar)
        self.total_budget = cpu_budget

    def fit(self, X, y):
        start = process_time()
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        train, valid = week_validation(X, y, self.task)
        prepare = preprocessor(self.numeric, self.categorical, self.calendar).fit(X.iloc[train])
        features = torch.from_numpy(dense(prepare.transform(X)))
        if self.task == "severity":
            classes = torch.tensor([CLASSES.index(label) for label in y], dtype=torch.long)
            delays = None
        else:
            values = np.asarray(y, dtype=np.float32)
            classes = None
            delays = torch.tensor((values - values[train].mean()) / max(values[train].std(), 1.0))
        n_outputs = len(CLASSES) if classes is not None else 1
        widths = [features.shape[1], 512, 512, 512, 512, 512]
        layers = [GatedLayer(a, b) for a, b in zip(widths[:-1], widths[1:])]
        model = nn.Sequential(*layers, nn.Linear(widths[-1], n_outputs))
        gate_parameters = [layer.strength for layer in layers]
        weights = [param for name, param in model.named_parameters() if not name.endswith(".strength")]
        weight_optimizer = torch.optim.Adam(weights, lr=0.001)
        gate_optimizer = torch.optim.Adam(gate_parameters, lr=0.05)
        train, valid = torch.as_tensor(train), torch.as_tensor(valid)
        best_loss, best_state = np.inf, None
        search_deadline = start + 0.75 * self.total_budget
        stale = 0
        for _ in range(MAX_EPOCHS):
            model.train()
            gate_batches = valid[torch.randperm(len(valid))].split(256)
            for i, batch in enumerate(train[torch.randperm(len(train))].split(256)):
                self._update(model, weight_optimizer, features, classes, delays, batch)
                self._update(model, gate_optimizer, features, classes, delays,
                             gate_batches[i % len(gate_batches)])
                if process_time() >= search_deadline:
                    break
            model.eval()
            with torch.inference_mode():
                loss = float(self.loss(model(features[valid]), classes, delays, valid))
            if loss < best_loss:
                best_loss, best_state = loss, deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE or process_time() >= search_deadline:
                break
        model.load_state_dict(best_state)
        self.widths = [layer.selected_width() for layer in layers]
        now = process_time()
        remaining = max(0, self.total_budget - (now - start))
        self.epochs = self._select_refit_epochs(features, classes, delays, train, valid,
                                                n_outputs, now + remaining / 2)
        self.cpu_budget = max(0, self.total_budget - (process_time() - start))
        super().fit(X, y)
        self.cpu_seconds_ = process_time() - start
        return self

    def _select_refit_epochs(self, features, classes, delays, train, valid, n_outputs, deadline):
        """Choose a dense-network epoch count on the inner validation weeks."""
        torch.manual_seed(SEED)
        model = self.network(features.shape[1], n_outputs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        best_loss, best_epoch = np.inf, 1
        for epoch in range(1, MAX_EPOCHS + 1):
            if epoch - best_epoch > PATIENCE or process_time() >= deadline:
                break
            model.train()
            for batch in train[torch.randperm(len(train))].split(256):
                self._update(model, optimizer, features, classes, delays, batch)
                if process_time() >= deadline:
                    return best_epoch
            model.eval()
            with torch.inference_mode():
                loss = float(self.loss(model(features[valid]), classes, delays, valid))
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
        return best_epoch

    def _update(self, model, optimizer, features, classes, delays, batch):
        model.zero_grad()
        loss = self.loss(model(features[batch]), classes, delays, batch)
        loss.backward()
        optimizer.step()
