from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.data.transforms import anisotropic_scale, translate, yaw_rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic ModelNet10 point-cloud transformation sequences.")
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--output-root", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--transform",
        choices=["yaw", "scale_x", "scale_y", "scale_z", "translate_x", "yaw_scale_x"],
        default="yaw",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-angle", type=float, default=math.pi / 2)
    parser.add_argument("--max-scale", type=float, default=1.5)
    parser.add_argument("--max-translation", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def apply_transform(points: torch.Tensor, kind: str, alpha: float, args: argparse.Namespace) -> torch.Tensor:
    if kind == "yaw":
        return yaw_rotation(points, alpha * args.max_angle)
    if kind == "scale_x":
        return anisotropic_scale(points, (1.0 + alpha * (args.max_scale - 1.0), 1.0, 1.0))
    if kind == "scale_y":
        return anisotropic_scale(points, (1.0, 1.0 + alpha * (args.max_scale - 1.0), 1.0))
    if kind == "scale_z":
        return anisotropic_scale(points, (1.0, 1.0, 1.0 + alpha * (args.max_scale - 1.0)))
    if kind == "translate_x":
        return translate(points, (alpha * args.max_translation, 0.0, 0.0))
    if kind == "yaw_scale_x":
        rotated = yaw_rotation(points, alpha * args.max_angle)
        return anisotropic_scale(rotated, (1.0 + alpha * (args.max_scale - 1.0), 1.0, 1.0))
    raise ValueError(kind)


def transform_parameters(kind: str, alpha: float, args: argparse.Namespace) -> dict[str, float]:
    params: dict[str, float] = {"alpha": float(alpha)}
    if kind in {"yaw", "yaw_scale_x"}:
        params["angle"] = float(alpha * args.max_angle)
    if kind in {"scale_x", "scale_y", "scale_z", "yaw_scale_x"}:
        params["scale"] = float(1.0 + alpha * (args.max_scale - 1.0))
    if kind == "translate_x":
        params["translation_x"] = float(alpha * args.max_translation)
    return params


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ds = ModelNet10PointClouds(args.cache_root, split=args.split)
    index = []

    num_items = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
    for i in tqdm(range(num_items), desc=f"{args.split}/{args.transform}"):
        item = ds[i]
        points = item["points"]
        seq = []
        alphas = []
        parameters = []
        for step in range(args.steps + 1):
            alpha = step / max(args.steps, 1)
            alphas.append(float(alpha))
            parameters.append(transform_parameters(args.transform, alpha, args))
            seq.append(apply_transform(points, args.transform, alpha, args))
        sequence = torch.stack(seq, dim=0)
        rel_file = Path(args.split) / args.transform / f"{item['object_id']}.pt"
        out_path = args.output_root / rel_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sequence": sequence,
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "object_id": item["object_id"],
                "split": args.split,
                "transform": args.transform,
                "alphas": alphas,
                "parameters": parameters,
            },
            out_path,
        )
        index.append(
            {
                "file": rel_file.as_posix(),
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "object_id": item["object_id"],
                "split": args.split,
                "transform": args.transform,
                "steps": args.steps,
                "max_angle": args.max_angle,
                "max_scale": args.max_scale,
                "max_translation": args.max_translation,
            }
        )

    index_path = args.output_root / f"index_{args.split}_{args.transform}.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} sequences to {args.output_root}")


if __name__ == "__main__":
    main()
