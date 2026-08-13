from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtEmotion(nn.Module):
    """Selected FER2013 model: ConvNeXt-Tiny with a 256-D emotion embedding."""

    def __init__(self, classes: int = 7, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.30),
        )
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.embedding(self.backbone(x))
        logits = self.classifier(embedding)
        return (embedding, logits) if return_embedding else logits

