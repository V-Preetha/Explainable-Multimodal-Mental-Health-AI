from __future__ import annotations

import torch
from torch import nn


class GatedFusionMultiTask(nn.Module):
    """Projects three modality embeddings, gates them, then predicts status and scores."""

    def __init__(self, face_dim=256, speech_dim=256, numerical_dim=128, project_dim=256, classes=4):
        super().__init__()
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, project_dim), nn.LayerNorm(project_dim), nn.GELU())
            for dim in (face_dim, speech_dim, numerical_dim)
        )
        self.gate = nn.Sequential(nn.Linear(project_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.trunk = nn.Sequential(
            nn.Linear(project_dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, 128), nn.GELU()
        )
        self.status_head = nn.Linear(128, classes)
        self.regression_head = nn.Linear(128, 3)

    def forward(self, face, speech, numerical, available=None, modality_dropout=0.0):
        projected = torch.stack([layer(x) for layer, x in zip(self.projectors, (face, speech, numerical))], dim=1)
        batch = projected.size(0)
        mask = torch.ones((batch, 3), dtype=torch.bool, device=projected.device) if available is None else available.bool()
        if self.training and modality_dropout:
            drop = (torch.rand_like(mask.float()) < modality_dropout) & mask
            mask = mask & ~drop
            mask[~mask.any(dim=1), 2] = True
        scores = self.gate(projected).squeeze(-1).masked_fill(~mask, -1e4)
        weights = scores.softmax(dim=1)
        shared = self.trunk((projected * weights.unsqueeze(-1)).sum(dim=1))
        return self.status_head(shared), torch.sigmoid(self.regression_head(shared)), weights, shared

