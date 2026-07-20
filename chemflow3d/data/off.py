from __future__ import annotations

from pathlib import Path

import numpy as np


def read_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an OFF mesh file.

    Returns:
        vertices: float32 array with shape (V, 3).
        faces: int64 padded object-style triangle fan faces as (F, K).

    ModelNet10 mostly uses triangular faces, but this parser keeps polygon
    indices and lets the sampler triangulate them.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
        if header.startswith("OFF") and header != "OFF":
            rest = header[3:].strip().split()
            counts = rest
        else:
            if header != "OFF":
                raise ValueError(f"{path} is not a valid OFF file: {header!r}")
            counts = []

        while len(counts) < 3:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading OFF counts: {path}")
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            counts.extend(line.split())

        n_vertices, n_faces, _ = map(int, counts[:3])

        vertices = []
        while len(vertices) < n_vertices:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading vertices: {path}")
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            if len(values) < 3:
                continue
            vertices.append([float(values[0]), float(values[1]), float(values[2])])

        faces: list[list[int]] = []
        while len(faces) < n_faces:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [int(v) for v in line.split()]
            if not values:
                continue
            count = values[0]
            idx = values[1 : 1 + count]
            if len(idx) >= 3:
                faces.append(idx)

    vertices_arr = np.asarray(vertices, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=object)
    return vertices_arr, faces_arr


def triangulate_faces(faces: np.ndarray) -> np.ndarray:
    """Triangulate polygon faces with a triangle fan."""
    triangles: list[list[int]] = []
    for face in faces:
        idx = list(face)
        if len(idx) < 3:
            continue
        for i in range(1, len(idx) - 1):
            triangles.append([idx[0], idx[i], idx[i + 1]])
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)
