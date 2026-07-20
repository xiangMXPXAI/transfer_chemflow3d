from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


MODELNET10_CLASSES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]


class ModelNet10PointClouds(Dataset):
    """Dataset backed by preprocessed point-cloud .pt files."""

    def __init__(self, root: str | Path, split: str = "train", classes: list[str] | None = None):
        self.root = Path(root)
        self.split = split
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing cache index: {index_path}. Run preprocess_modelnet10.py first."
            )

        with index_path.open("r", encoding="utf-8") as f:
            all_items: list[dict[str, Any]] = json.load(f)

        keep_classes = set(classes) if classes else None
        self.items = [
            item
            for item in all_items
            if item["split"] == split and (keep_classes is None or item["class_name"] in keep_classes)
        ]
        if not self.items:
            raise ValueError(f"No samples found for split={split!r}, classes={classes!r}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        item = self.items[idx]
        data = torch.load(self.root / item["file"], map_location="cpu", weights_only=False)
        return {
            "points": data["points"].float(),
            "class_id": int(item["class_id"]),
            "class_name": item["class_name"],
            "object_id": item["object_id"],
            "split": item["split"],
        }
