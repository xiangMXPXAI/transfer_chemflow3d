from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from chemflow3d.losses.chamfer import chamfer_l2_per_sample
from chemflow3d.models import PointNetAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AE representation on synthetic point-cloud sequences.")
    parser.add_argument("--sequence-root", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences"))
    parser.add_argument("--index", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences/index_train_yaw.json"))
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e3_eval"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with args.index.open("r", encoding="utf-8") as f:
        items = json.load(f)

    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device, weights_only=False)["model"])
    ae.eval()

    rows = []
    with torch.no_grad():
        for item in tqdm(items, desc="eval synthetic sequences"):
            data = torch.load(args.sequence_root / item["file"], map_location=device, weights_only=False)
            seq = data["sequence"].to(device)
            flat = seq.reshape(-1, seq.shape[-2], seq.shape[-1])
            recon, z = ae(flat)
            rec = chamfer_l2_per_sample(recon, flat)
            z_seq = z.reshape(seq.shape[0], -1)
            z_delta = z_seq[1:] - z_seq[:-1]
            z_step = z_delta.norm(dim=1)
            if z_delta.shape[0] > 1:
                adjacent_cosine = torch.nn.functional.cosine_similarity(z_delta[:-1], z_delta[1:], dim=1)
                mean_adjacent_cosine = float(adjacent_cosine.mean().cpu())
                min_adjacent_cosine = float(adjacent_cosine.min().cpu())
            else:
                mean_adjacent_cosine = 1.0
                min_adjacent_cosine = 1.0
            rows.append(
                {
                    "object_id": item["object_id"],
                    "class_name": item["class_name"],
                    "transform": item["transform"],
                    "mean_recon_chamfer": float(rec.mean().cpu()),
                    "max_recon_chamfer": float(rec.max().cpu()),
                    "mean_latent_step": float(z_step.mean().cpu()),
                    "std_latent_step": float(z_step.std(unbiased=False).cpu()),
                    "latent_endpoint_l2": float((z_seq[-1] - z_seq[0]).norm().cpu()),
                    "mean_adjacent_latent_cosine": mean_adjacent_cosine,
                    "min_adjacent_latent_cosine": min_adjacent_cosine,
                }
            )

    out_csv = args.output / f"synthetic_{items[0]['split']}_{items[0]['transform']}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"num_samples": len(rows)}
    for key in [
        "mean_recon_chamfer",
        "max_recon_chamfer",
        "mean_latent_step",
        "std_latent_step",
        "latent_endpoint_l2",
        "mean_adjacent_latent_cosine",
        "min_adjacent_latent_cosine",
    ]:
        values = torch.tensor([float(row[key]) for row in rows])
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"median_{key}"] = float(values.median())
    out_json = args.output / f"synthetic_{items[0]['split']}_{items[0]['transform']}_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
