"""Compact numerical encoder: 18 inputs -> 64 -> 128 -> 128-D embedding,
with a classification head (4-class Mental_Health_Status) and a regression
head (Depression/Anxiety/Stress) attached to the shared embedding.
"""
from __future__ import annotations

import torch
from torch import nn


class NumericalEncoder(nn.Module):
    def __init__(self, input_dim: int = 18, embedding_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout))
        self.embedding_proj = nn.Sequential(nn.Linear(128, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return self.embedding_proj(h)


class NumericalMultiTask(nn.Module):
    def __init__(self, input_dim: int = 18, embedding_dim: int = 128, classes: int = 4, regression_outputs: int = 3, dropout: float = 0.25):
        super().__init__()
        self.encoder = NumericalEncoder(input_dim, embedding_dim, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout * 0.6), nn.Linear(embedding_dim, classes))
        self.regressor = nn.Linear(embedding_dim, regression_outputs)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, self.classifier(z), self.regressor(z)


