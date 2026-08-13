"""Model heads for the speech branch: attentive statistics pooling over
frozen emotion2vec+ frame embeddings, an MLP classification head, and a
hybrid fusion MLP that also incorporates engineered acoustic features.

Kept self-contained (not imported from ``src/models.py``) so this branch has
its own module and checkpoint namespace, independent of the face/fusion work
in ``src/`` and ``scripts/``.
"""
from __future__ import annotations

import torch
from torch import nn


class AttentiveStatsPool(nn.Module):
    """Learned temporal attention over frame embeddings -> [mean; std] -> projection."""

    def __init__(self, input_dim: int, embedding_dim: int = 256, attention_hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(input_dim, attention_hidden), nn.Tanh(), nn.Linear(attention_hidden, 1))
        self.project = nn.Sequential(
            nn.Linear(input_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        score = self.attn(x).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(~mask, -1e4)
        weights = score.softmax(dim=1)
        mean = (x * weights.unsqueeze(-1)).sum(dim=1)
        variance = ((x - mean.unsqueeze(1)) ** 2 * weights.unsqueeze(-1)).sum(dim=1).clamp_min(1e-6)
        pooled = torch.cat([mean, variance.sqrt()], dim=-1)
        return self.project(pooled), weights


class SpeechEmotionMLP(nn.Module):
    """Attentive-stats-pooled emotion2vec+ embedding -> MLP classification head."""

    def __init__(self, input_dim: int = 768, embedding_dim: int = 256, classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.pool = AttentiveStatsPool(input_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        embedding, attention_weights = self.pool(x, mask)
        logits = self.classifier(embedding)
        return embedding, logits, attention_weights


class HybridFusionMLP(nn.Module):
    """emotion2vec+ (attentive-pooled) embedding + engineered acoustic features -> fusion MLP."""

    def __init__(self, frame_dim: int = 768, engineered_dim: int = 0, embedding_dim: int = 256, engineered_proj_dim: int = 128, classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.pool = AttentiveStatsPool(frame_dim, embedding_dim)
        self.engineered_proj = nn.Sequential(
            nn.Linear(engineered_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, engineered_proj_dim),
            nn.LayerNorm(engineered_proj_dim),
            nn.GELU(),
        )
        fused_dim = embedding_dim + engineered_proj_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, frames: torch.Tensor, engineered: torch.Tensor, mask: torch.Tensor | None = None):
        speech_embedding, attention_weights = self.pool(frames, mask)
        engineered_embedding = self.engineered_proj(engineered)
        fused = self.fusion(torch.cat([speech_embedding, engineered_embedding], dim=-1))
        logits = self.classifier(fused)
        return fused, logits, attention_weights

