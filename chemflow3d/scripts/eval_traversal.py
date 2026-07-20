from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.flows import EnergyField, TraversalConfig, rollout
from chemflow3d.flows.traversal import property_gradient, random_direction
from chemflow3d.metrics.geometry import geometric_properties
from chemflow3d.metrics.trajectory import decoded_path_length, latent_path_length
from chemflow3d.models import PointNetAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate random/gradient/PDE point-cloud latent traversal.")
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--flow-ckpt", type=Path)
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/eval_traversal"))
    parser.add_argument("--method", choices=["random", "property_gradient", "pde"], default="random")
    parser.add_argument("--tag", type=str, default=None, help="Optional filename tag, e.g. wave or hj.")
    parser.add_argument("--property", choices=["width", "height", "depth", "volume", "compactness"], default="height")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--no-normalize", action="store_true", help="Do NOT normalize velocity during rollout.")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ModelNet10PointClouds(args.cache_root, split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device, weights_only=False)["model"])
    ae.eval()

    energy = None
    if args.method == "pde":
        if args.flow_ckpt is None:
            raise ValueError("--flow-ckpt is required for --method pde")
        energy = EnergyField(args.latent_dim, num_flows=1, hidden_dim=args.hidden_dim).to(device)
        energy.load_state_dict(torch.load(args.flow_ckpt, map_location=device, weights_only=False)["model"])
        energy.eval()

    prop_idx = {"width": 0, "height": 1, "depth": 2, "volume": 3, "compactness": 4}[args.property]
    cfg = TraversalConfig(steps=args.steps, step_size=args.step_size, normalize=not args.no_normalize)
    rows = []

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        points = batch["points"].to(device)
        with torch.no_grad():
            z0 = ae.encode(points)

        if args.method == "random":
            velocity_fn = lambda z, t: random_direction(z, normalize=True)
        elif args.method == "property_gradient":
            def prop_fn(z):
                return geometric_properties(ae.decode(z))[:, prop_idx : prop_idx + 1]
            velocity_fn = lambda z, t: property_gradient(prop_fn, z, normalize=True)
        else:
            velocity_fn = lambda z, t: energy.velocity(0, z, t, create_graph=False)

        zs, xs = rollout(velocity_fn, ae.decode, z0, cfg)
        p0 = geometric_properties(xs[0])[:, prop_idx]
        pT = geometric_properties(xs[-1])[:, prop_idx]
        lat_len = latent_path_length(zs)
        dec_len = decoded_path_length(xs)

        for i in range(points.shape[0]):
            rows.append(
                {
                    "object_id": batch["object_id"][i],
                    "class_name": batch["class_name"][i],
                    "method": args.method,
                    "property": args.property,
                    "p0": float(p0[i].detach().cpu()),
                    "pT": float(pT[i].detach().cpu()),
                    "delta": float((pT[i] - p0[i]).detach().cpu()),
                    "latent_path_length": float(lat_len[i].detach().cpu()),
                    "decoded_path_length": float(dec_len[i].detach().cpu()),
                }
            )

    tag = args.tag or args.method
    out_csv = args.output / f"{tag}_{args.property}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    deltas = torch.tensor([row["delta"] for row in rows])
    latent_lengths = torch.tensor([row["latent_path_length"] for row in rows])
    decoded_lengths = torch.tensor([row["decoded_path_length"] for row in rows])
    summary = {
        "tag": tag,
        "method": args.method,
        "property": args.property,
        "num_samples": len(rows),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(deltas.median()),
        "positive_delta_rate": float((deltas > 0).float().mean()),
        "mean_latent_path_length": float(latent_lengths.mean()),
        "mean_decoded_path_length": float(decoded_lengths.mean()),
    }
    out_json = args.output / f"{tag}_{args.property}_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
