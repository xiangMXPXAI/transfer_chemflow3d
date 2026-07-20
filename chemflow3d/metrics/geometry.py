from __future__ import annotations

import torch


def bbox_stats(points: torch.Tensor) -> dict[str, torch.Tensor]:
    mins = points.min(dim=-2).values
    maxs = points.max(dim=-2).values
    extent = maxs - mins
    return {
        "mins": mins,
        "maxs": maxs,
        "extent": extent,
        "width": extent[..., 0],
        "height": extent[..., 1],
        "depth": extent[..., 2],
        "volume": extent.prod(dim=-1),
    }


def geometric_properties(points: torch.Tensor) -> torch.Tensor:
    """Return width, height, depth, bbox volume, compactness."""
    stats = bbox_stats(points)
    centered = points - points.mean(dim=-2, keepdim=True)
    compactness = centered.norm(dim=-1).mean(dim=-1)
    return torch.stack(
        [
            stats["width"],
            stats["height"],
            stats["depth"],
            stats["volume"],
            compactness,
        ],
        dim=-1,
    )
