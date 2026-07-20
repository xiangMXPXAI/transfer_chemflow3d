from __future__ import annotations

import math

import torch


def yaw_rotation(points: torch.Tensor, angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    rot = points.new_tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return points @ rot.T


def anisotropic_scale(points: torch.Tensor, scale: tuple[float, float, float]) -> torch.Tensor:
    s = points.new_tensor(scale).view(1, 3)
    return points * s


def translate(points: torch.Tensor, offset: tuple[float, float, float]) -> torch.Tensor:
    return points + points.new_tensor(offset).view(1, 3)
