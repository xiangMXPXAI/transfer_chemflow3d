from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class SyntheticSequenceDataset(Dataset):
    """Dataset backed by synthetic point-cloud transformation sequences.

    Each sample stores a sequence with shape (T + 1, N, 3), where adjacent
    frames are generated from an analytic geometric transform.  The sequence
    therefore provides ground-truth finite-difference directions in point-cloud
    space and, after encoding with a frozen AE, in latent space.
    """

    def __init__(self, root: str | Path, index: str | Path):
        self.root = Path(root)
        self.index_path = Path(index)
        if not self.index_path.is_absolute() and not self.index_path.exists():
            self.index_path = self.root / self.index_path
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing synthetic sequence index: {self.index_path}")

        with self.index_path.open("r", encoding="utf-8") as f:
            self.items: list[dict[str, Any]] = json.load(f)
        if not self.items:
            raise ValueError(f"No synthetic sequence items found in {self.index_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        item = self.items[idx]
        data = torch.load(self.root / item["file"], map_location="cpu", weights_only=False)
        sequence = data["sequence"].float()
        return {
            "sequence": sequence,
            "class_id": int(item["class_id"]),
            "class_name": item["class_name"],
            "object_id": item["object_id"],
            "split": item["split"],
            "transform": item["transform"],
            "steps": int(item["steps"]),
        }
