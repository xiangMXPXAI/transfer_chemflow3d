from __future__ import annotations

import torch
from torch import nn


class PointCloudGenerator(nn.Module):
    """ChemFlow-style generator: z -> decoded point cloud."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class PointCloudPropGenerator(nn.Module):
    """ChemFlow-style property generator: z -> property/class output."""

    def __init__(self, decoder: nn.Module, prop_or_classifier: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.prop_or_classifier = prop_or_classifier

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        points = self.decoder(z)
        return self.prop_or_classifier(points)
