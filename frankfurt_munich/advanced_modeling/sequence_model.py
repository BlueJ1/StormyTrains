"""A small GRU over upstream stop reports, joined to timetable and weather inputs."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import _paths  # noqa: F401  shared modelling code and dataset builders
from neural import TorchMLP, perceptron
from sequence_inputs import SEQUENCE_CHANNELS, SEQUENCE_COLUMN, SEQUENCE_LENGTH

PACKED_SIZE = 1 + SEQUENCE_LENGTH * SEQUENCE_CHANNELS


class SequenceNetwork(nn.Module):
    def __init__(self, static_columns, widths, n_outputs, dropout):
        super().__init__()
        self.gru = nn.GRU(SEQUENCE_CHANNELS, 32, batch_first=True)
        self.head = perceptron(static_columns + 32, widths, n_outputs, dropout)

    def forward(self, features):
        lengths = features[:, 0].long()
        padded = features[:, 1:PACKED_SIZE].reshape(-1, SEQUENCE_LENGTH, SEQUENCE_CHANNELS)
        active = lengths > 0
        last = features.new_zeros((len(features), self.gru.hidden_size))
        if active.any():
            length = lengths[active]
            hidden, _ = self.gru(padded[active, :int(length.max())])
            last[active] = hidden[torch.arange(len(length)), length - 1]
        return self.head(torch.cat([last, features[:, PACKED_SIZE:]], dim=1))


class SequenceMLP(TorchMLP):
    def __init__(self, numeric, categorical, **params):
        super().__init__([column for column in numeric if column != SEQUENCE_COLUMN], categorical,
                         **params)

    def input_tensor(self, X, *, fit=False):
        static = super().input_tensor(X.drop(columns=SEQUENCE_COLUMN), fit=fit)
        packed = np.zeros((len(X), PACKED_SIZE), dtype=np.float32)
        for i, sequence in enumerate(X[SEQUENCE_COLUMN]):
            values = np.asarray(sequence, dtype=np.float32)[-(PACKED_SIZE - 1):]
            packed[i, 0] = len(values) // SEQUENCE_CHANNELS
            packed[i, 1:1 + len(values)] = values
        return torch.cat([torch.from_numpy(packed), static], dim=1)

    def network(self, n_inputs, n_outputs):
        return SequenceNetwork(n_inputs - PACKED_SIZE, self.widths, n_outputs, self.dropout)
