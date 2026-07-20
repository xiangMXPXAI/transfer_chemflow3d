from __future__ import annotations

import torch


def chamfer_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer-L2 distance for point clouds.

    Args:
        x: Tensor of shape (B, N, 3).
        y: Tensor of shape (B, M, 3).
    """
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("Expected x and y with shape (B, N, 3) and (B, M, 3).")
    dist = torch.cdist(x, y, p=2).square()
    x_to_y = dist.min(dim=2).values.mean(dim=1)
    y_to_x = dist.min(dim=1).values.mean(dim=1)
    return (x_to_y + y_to_x).mean()


def chamfer_l2_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, y, p=2).square()
    return dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)
