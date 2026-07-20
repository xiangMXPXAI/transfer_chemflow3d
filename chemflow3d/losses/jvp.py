from __future__ import annotations

import torch
from torch.autograd.functional import jvp


def jvp_norm(module, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Mean squared norm of J_module(z) v."""
    _, value = jvp(module, (z,), (v,), create_graph=True)
    return value.flatten(start_dim=1).square().sum(dim=1).mean()


def scalar_directional_derivative(module, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Mean directional derivative for scalar or vector property outputs."""
    _, value = jvp(module, (z,), (v,), create_graph=True)
    if value.ndim == 1:
        return value.mean()
    return value.flatten(start_dim=1).mean(dim=1).mean()
