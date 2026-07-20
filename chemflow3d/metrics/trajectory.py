from __future__ import annotations

import torch

from chemflow3d.losses.chamfer import chamfer_l2_per_sample


def latent_path_length(zs: torch.Tensor) -> torch.Tensor:
    """Path length for trajectory tensor with shape (T, B, D)."""
    return (zs[1:] - zs[:-1]).norm(dim=-1).sum(dim=0)


def latent_action(zs: torch.Tensor) -> torch.Tensor:
    return (zs[1:] - zs[:-1]).square().sum(dim=-1).sum(dim=0)


def decoded_path_length(xs: torch.Tensor) -> torch.Tensor:
    """Chamfer path length for trajectory tensor with shape (T, B, N, 3)."""
    dists = [chamfer_l2_per_sample(xs[t], xs[t + 1]) for t in range(xs.shape[0] - 1)]
    return torch.stack(dists, dim=0).sum(dim=0)
