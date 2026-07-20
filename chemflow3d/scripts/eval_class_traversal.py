from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.flows import EnergyField, TraversalConfig, rollout
from chemflow3d.flows.traversal import normalize_direction, random_direction
from chemflow3d.metrics.trajectory import decoded_path_length, latent_path_length
from chemflow3d.models import PointNetAE, PointNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate class-target point-cloud latent traversal.")
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--classifier-ckpt", type=Path, required=True)
    parser.add_argument("--flow-ckpt", type=Path)
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e2_eval"))
    parser.add_argument("--method", choices=["random", "class_gradient", "pde"], default="pde")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--target-class", type=int, default=8)
    parser.add_argument("--source-class", type=int, default=None, help="Evaluate only this source class id.")
    parser.add_argument(
        "--include-target-samples",
        action="store_true",
        help="Include samples already belonging to target class. Disabled by default to avoid inflated success.",
    )
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def target_margin(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
    mask[target_class] = True
    other_max = logits.masked_fill(mask[None, :], -torch.inf).max(dim=1).values
    return logits[:, target_class] - other_max


def target_probability(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    return logits.softmax(dim=1)[:, target_class]


def class_gradient_direction(
    decoder,
    classifier: PointNetClassifier,
    z: torch.Tensor,
    target_class: int,
    normalize: bool = True,
) -> torch.Tensor:
    z_req = z.detach().requires_grad_(True)
    logits = classifier(decoder(z_req))
    score = logits[:, target_class].sum()
    (v,) = torch.autograd.grad(score, z_req)
    return normalize_direction(v) if normalize else v


def load_ae(args: argparse.Namespace, device: torch.device) -> PointNetAE:
    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device, weights_only=False)["model"])
    ae.eval()
    return ae


def load_classifier(args: argparse.Namespace, device: torch.device) -> PointNetClassifier:
    classifier = PointNetClassifier(num_classes=10).to(device)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device, weights_only=False)["model"])
    classifier.eval()
    return classifier


def load_energy(args: argparse.Namespace, device: torch.device) -> EnergyField:
    if args.flow_ckpt is None:
        raise ValueError("--flow-ckpt is required for --method pde")
    ckpt = torch.load(args.flow_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    latent_dim = int(ckpt_args.get("latent_dim", args.latent_dim))
    hidden_dim = int(ckpt_args.get("hidden_dim", args.hidden_dim))
    energy = EnergyField(latent_dim, num_flows=1, hidden_dim=hidden_dim).to(device)
    energy.load_state_dict(ckpt["model"])
    energy.eval()
    return energy


def filter_batch(
    points: torch.Tensor,
    labels: torch.Tensor,
    object_ids,
    class_names,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    if args.source_class is not None:
        keep = labels == args.source_class
    elif args.include_target_samples:
        keep = torch.ones_like(labels, dtype=torch.bool)
    else:
        keep = labels != args.target_class

    keep_idx = keep.nonzero(as_tuple=False).flatten()
    object_ids_out = [object_ids[int(i)] for i in keep_idx.cpu()]
    class_names_out = [class_names[int(i)] for i in keep_idx.cpu()]
    return points[keep], labels[keep], object_ids_out, class_names_out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ModelNet10PointClouds(args.cache_root, split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    ae = load_ae(args, device)
    classifier = load_classifier(args, device)
    energy = load_energy(args, device) if args.method == "pde" else None

    cfg = TraversalConfig(steps=args.steps, step_size=args.step_size, normalize=not args.no_normalize)
    rows = []

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        points_all = batch["points"].to(device)
        labels_all = torch.as_tensor(batch["class_id"], device=device, dtype=torch.long)
        points, labels, object_ids, class_names = filter_batch(
            points_all,
            labels_all,
            batch["object_id"],
            batch["class_name"],
            args,
        )
        if points.numel() == 0:
            continue

        with torch.no_grad():
            z0 = ae.encode(points)
            x0 = ae.decode(z0)
            logits0 = classifier(x0)

        if args.method == "random":
            velocity_fn = lambda z, t: random_direction(z, normalize=True)
        elif args.method == "class_gradient":
            velocity_fn = lambda z, t: class_gradient_direction(
                ae.decode,
                classifier,
                z,
                args.target_class,
                normalize=True,
            )
        else:
            velocity_fn = lambda z, t: energy.velocity(0, z, t, create_graph=False)

        zs, xs = rollout(velocity_fn, ae.decode, z0, cfg)
        with torch.no_grad():
            logits_t = classifier(xs[-1])
            pred0 = logits0.argmax(dim=1)
            pred_t = logits_t.argmax(dim=1)
            prob0 = target_probability(logits0, args.target_class)
            prob_t = target_probability(logits_t, args.target_class)
            margin0 = target_margin(logits0, args.target_class)
            margin_t = target_margin(logits_t, args.target_class)
            lat_len = latent_path_length(zs)
            dec_len = decoded_path_length(xs)

        for i in range(points.shape[0]):
            rows.append(
                {
                    "object_id": object_ids[i],
                    "source_class_id": int(labels[i].item()),
                    "source_class_name": class_names[i],
                    "method": args.method,
                    "target_class": args.target_class,
                    "initial_pred": int(pred0[i].item()),
                    "final_pred": int(pred_t[i].item()),
                    "initial_target_success": int(pred0[i].item() == args.target_class),
                    "target_success": int(pred_t[i].item() == args.target_class),
                    "target_prob0": float(prob0[i].cpu()),
                    "target_probT": float(prob_t[i].cpu()),
                    "target_prob_delta": float((prob_t[i] - prob0[i]).cpu()),
                    "margin0": float(margin0[i].cpu()),
                    "marginT": float(margin_t[i].cpu()),
                    "margin_delta": float((margin_t[i] - margin0[i]).cpu()),
                    "latent_path_length": float(lat_len[i].cpu()),
                    "decoded_path_length": float(dec_len[i].cpu()),
                }
            )

    if not rows:
        raise RuntimeError("No samples were evaluated. Check --source-class/--target-class filters.")

    tag = args.tag or args.method
    source_part = f"_source_{args.source_class}" if args.source_class is not None else ""
    out_csv = args.output / f"{tag}_target_{args.target_class}{source_part}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    success = torch.tensor([row["target_success"] for row in rows], dtype=torch.float32)
    initial_success = torch.tensor([row["initial_target_success"] for row in rows], dtype=torch.float32)
    margin_delta = torch.tensor([row["margin_delta"] for row in rows], dtype=torch.float32)
    prob_delta = torch.tensor([row["target_prob_delta"] for row in rows], dtype=torch.float32)
    decoded_lengths = torch.tensor([row["decoded_path_length"] for row in rows], dtype=torch.float32)
    summary = {
        "tag": tag,
        "method": args.method,
        "target_class": args.target_class,
        "source_class": args.source_class,
        "include_target_samples": args.include_target_samples,
        "num_samples": len(rows),
        "initial_target_success_rate": float(initial_success.mean()),
        "target_success_rate": float(success.mean()),
        "new_success_rate": float(((success == 1) & (initial_success == 0)).float().mean()),
        "mean_margin_delta": float(margin_delta.mean()),
        "median_margin_delta": float(margin_delta.median()),
        "positive_margin_delta_rate": float((margin_delta > 0).float().mean()),
        "mean_target_prob_delta": float(prob_delta.mean()),
        "positive_target_prob_delta_rate": float((prob_delta > 0).float().mean()),
        "mean_decoded_path_length": float(decoded_lengths.mean()),
    }
    out_json = args.output / f"{tag}_target_{args.target_class}{source_part}_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
