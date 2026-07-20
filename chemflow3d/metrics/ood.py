from __future__ import annotations

import torch


def nearest_neighbor_distance(z: torch.Tensor, reference: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Nearest-neighbor distance from z to reference latent codes."""
    outs = []
    for start in range(0, z.shape[0], chunk):
        zz = z[start : start + chunk]
        dist = torch.cdist(zz, reference).min(dim=1).values
        outs.append(dist)
    return torch.cat(outs, dim=0)
