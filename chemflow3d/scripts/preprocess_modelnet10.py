from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from chemflow3d.data.modelnet10 import MODELNET10_CLASSES
from chemflow3d.data.off import read_off, triangulate_faces
from chemflow3d.data.sampling import normalize_unit_sphere, sample_surface_points

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ModelNet10 OFF meshes into point-cloud tensors.")
    parser.add_argument("--input-root", type=Path, default=Path("ModelNet10"))
    parser.add_argument("--output-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    index = []

    for class_id, class_name in enumerate(MODELNET10_CLASSES):
        for split in ["train", "test"]:
            in_dir = args.input_root / class_name / split
            if not in_dir.exists():
                continue
            out_dir = args.output_root / class_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for off_path in tqdm(sorted(in_dir.glob("*.off")), desc=f"{class_name}/{split}"):
                object_id = off_path.stem
                rel_file = Path(class_name) / split / f"{object_id}.pt"
                out_path = args.output_root / rel_file
                if out_path.exists() and not args.overwrite:
                    index.append(
                        {
                            "file": rel_file.as_posix(),
                            "class_id": class_id,
                            "class_name": class_name,
                            "object_id": object_id,
                            "split": split,
                        }
                    )
                    continue

                vertices, faces = read_off(off_path)
                triangles = triangulate_faces(faces)
                points = sample_surface_points(vertices, triangles, args.num_points, rng)
                points = normalize_unit_sphere(points)
                torch.save(
                    {
                        "points": torch.from_numpy(points),
                        "class_id": class_id,
                        "class_name": class_name,
                        "object_id": object_id,
                        "split": split,
                        "source": str(off_path),
                    },
                    out_path,
                )
                index.append(
                    {
                        "file": rel_file.as_posix(),
                        "class_id": class_id,
                        "class_name": class_name,
                        "object_id": object_id,
                        "split": split,
                    }
                )

    with (args.output_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} samples to {args.output_root}")


if __name__ == "__main__":
    main()
