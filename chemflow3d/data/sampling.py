from __future__ import annotations

import numpy as np


def normalize_unit_sphere(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    points = points.astype(np.float32, copy=True)
    points -= points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points, axis=1).max()
    points /= radius + eps
    return points


def sample_surface_points(
    vertices: np.ndarray,
    triangles: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points from mesh surface by triangle area.

    Falls back to vertex resampling when faces are unavailable or degenerate.
    """
    if len(vertices) == 0:
        raise ValueError("Cannot sample an empty mesh.")

    if triangles.size == 0:
        idx = rng.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
        return vertices[idx].astype(np.float32)

    tri_vertices = vertices[triangles]
    a = tri_vertices[:, 1] - tri_vertices[:, 0]
    b = tri_vertices[:, 2] - tri_vertices[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    area_sum = float(areas.sum())

    if not np.isfinite(area_sum) or area_sum <= 1e-12:
        idx = rng.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
        return vertices[idx].astype(np.float32)

    probs = areas / area_sum
    face_idx = rng.choice(len(triangles), size=num_points, replace=True, p=probs)
    chosen = tri_vertices[face_idx]

    u = rng.random((num_points, 1), dtype=np.float32)
    v = rng.random((num_points, 1), dtype=np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    sampled = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])
    return sampled.astype(np.float32)
