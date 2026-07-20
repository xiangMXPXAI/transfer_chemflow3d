from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chemflow3d.data import SyntheticSequenceDataset
from chemflow3d.flows import EnergyField
from chemflow3d.flows.traversal import normalize_direction
from chemflow3d.losses.chamfer import chamfer_l2_per_sample
from chemflow3d.models import PointNetAE
from chemflow3d.scripts.train_synthetic_flow import adjacent_training_pairs, encode_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic ground-truth latent-flow traversal.")
    parser.add_argument("--sequence-root", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--flow-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e3_eval_flow"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def load_ae(args: argparse.Namespace, device: torch.device) -> PointNetAE:
    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def load_energy(args: argparse.Namespace, device: torch.device) -> tuple[EnergyField, dict]:
    ckpt = torch.load(args.flow_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    latent_dim = int(ckpt_args.get("latent_dim", args.latent_dim))
    hidden_dim = int(ckpt_args.get("hidden_dim", args.hidden_dim))
    energy = EnergyField(latent_dim, num_flows=1, hidden_dim=hidden_dim).to(device)
    energy.load_state_dict(ckpt["model"])
    energy.eval()
    return energy, ckpt_args


def energy_velocity(energy: EnergyField, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return energy.velocity(0, z, t, create_graph=False)


def rollout_energy(
    energy: EnergyField,
    decoder,
    z0: torch.Tensor,
    steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    zs = [z0]
    xs = [decoder(z0)]
    z = z0
    for step in range(steps):
        t = z.new_full((z.shape[0],), float(step))
        v = energy_velocity(energy, z, t)
        z = z + float(step_size) * v
        zs.append(z)
        xs.append(decoder(z))
    return torch.stack(zs, dim=1), torch.stack(xs, dim=1)


def trajectory_chamfer(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean per-frame Chamfer, pred/gt shape B x (T+1) x N x 3."""
    bsz, frames, npts, dims = pred.shape
    flat_pred = pred.reshape(bsz * frames, npts, dims)
    flat_gt = gt.reshape(bsz * frames, npts, dims)
    cd = chamfer_l2_per_sample(flat_pred, flat_gt).reshape(bsz, frames)
    return cd.mean(dim=1)


def velocity_alignment(
    energy: EnergyField,
    z_seq: torch.Tensor,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_t, t, v_gt = adjacent_training_pairs(z_seq, step_size)
    v_pred = energy_velocity(energy, z_t, t)
    valid = v_gt.norm(dim=1) > 1e-8
    cos = v_gt.new_zeros(v_gt.shape[0])
    if bool(valid.any()):
        cos[valid] = (normalize_direction(v_pred[valid]) * normalize_direction(v_gt[valid])).sum(dim=1)
    bsz, frames, _ = z_seq.shape
    steps = frames - 1
    cos_seq = cos.reshape(bsz, steps)
    mse_seq = (v_pred - v_gt).square().mean(dim=1).reshape(bsz, steps)
    return cos_seq.mean(dim=1), (cos_seq > 0.0).float().mean(dim=1), mse_seq.mean(dim=1)


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    numeric_keys = [
        "endpoint_chamfer",
        "trajectory_chamfer",
        "identity_endpoint_chamfer",
        "oracle_endpoint_chamfer",
        "latent_endpoint_l2",
        "mean_velocity_cosine",
        "positive_velocity_cosine_rate",
        "velocity_mse",
    ]
    summary: dict[str, float | int | str] = {"num_samples": len(rows)}
    for key in numeric_keys:
        values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float32)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"median_{key}"] = float(values.median())
    endpoint = torch.tensor([float(row["endpoint_chamfer"]) for row in rows])
    identity = torch.tensor([float(row["identity_endpoint_chamfer"]) for row in rows])
    oracle = torch.tensor([float(row["oracle_endpoint_chamfer"]) for row in rows])
    summary["improvement_over_identity_rate"] = float((endpoint < identity).float().mean())
    summary["mean_identity_to_flow_ratio"] = float((identity / endpoint.clamp_min(1e-12)).mean())
    summary["mean_flow_to_oracle_ratio"] = float((endpoint / oracle.clamp_min(1e-12)).mean())
    return summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SyntheticSequenceDataset(args.sequence_root, args.index)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    ae = load_ae(args, device)
    energy, ckpt_args = load_energy(args, device)
    step_size = float(args.step_size if args.step_size is not None else ckpt_args.get("step_size", 1.0))
    tag = args.tag or str(ckpt_args.get("pde", "flow"))

    rows = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="eval synthetic flow")):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        sequence = batch["sequence"].to(device)
        steps = int(sequence.shape[1] - 1)
        z_seq = encode_sequence(ae, sequence)
        z0 = z_seq[:, 0]
        zT_gt = z_seq[:, -1]
        x0_rec = ae.decode(z0)
        xT_rec_oracle = ae.decode(zT_gt)
        zs_pred, xs_pred = rollout_energy(energy, ae.decode, z0, steps, step_size)

        endpoint_cd = chamfer_l2_per_sample(xs_pred[:, -1], sequence[:, -1])
        trajectory_cd = trajectory_chamfer(xs_pred, sequence)
        identity_cd = chamfer_l2_per_sample(x0_rec, sequence[:, -1])
        oracle_cd = chamfer_l2_per_sample(xT_rec_oracle, sequence[:, -1])
        latent_l2 = (zs_pred[:, -1] - zT_gt).norm(dim=1)
        mean_cos, pos_cos_rate, velocity_mse = velocity_alignment(energy, z_seq, step_size)

        for i in range(sequence.shape[0]):
            rows.append(
                {
                    "object_id": batch["object_id"][i],
                    "class_id": int(batch["class_id"][i]),
                    "class_name": batch["class_name"][i],
                    "transform": batch["transform"][i],
                    "tag": tag,
                    "endpoint_chamfer": float(endpoint_cd[i].detach().cpu()),
                    "trajectory_chamfer": float(trajectory_cd[i].detach().cpu()),
                    "identity_endpoint_chamfer": float(identity_cd[i].detach().cpu()),
                    "oracle_endpoint_chamfer": float(oracle_cd[i].detach().cpu()),
                    "latent_endpoint_l2": float(latent_l2[i].detach().cpu()),
                    "mean_velocity_cosine": float(mean_cos[i].detach().cpu()),
                    "positive_velocity_cosine_rate": float(pos_cos_rate[i].detach().cpu()),
                    "velocity_mse": float(velocity_mse[i].detach().cpu()),
                }
            )

    if not rows:
        raise RuntimeError("No synthetic flow samples were evaluated.")

    transform = rows[0]["transform"]
    out_csv = args.output / f"{tag}_{transform}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["tag"] = tag
    summary["transform"] = transform
    summary["step_size"] = step_size
    out_json = args.output / f"{tag}_{transform}_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
