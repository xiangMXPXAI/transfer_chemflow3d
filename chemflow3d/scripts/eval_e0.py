from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.data.modelnet10 import MODELNET10_CLASSES
from chemflow3d.losses.chamfer import chamfer_l2_per_sample
from chemflow3d.metrics.geometry import bbox_stats
from chemflow3d.models import PointNetAE, PointNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E0 AE/classifier baselines and render reconstructions.")
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--ae-ckpt", type=Path, default=Path("chemflow3d_runs/e0_ae/best.pt"))
    parser.add_argument("--classifier-ckpt", type=Path, default=Path("chemflow3d_runs/e0_classifier/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e0_eval"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--viz-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[PointNetAE, PointNetClassifier]:
    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ae_ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()

    classifier = PointNetClassifier(num_classes=10).to(device)
    cl_ckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    classifier.load_state_dict(cl_ckpt["model"])
    classifier.eval()
    return ae, classifier


def render_recon_grid(samples: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not samples:
        return
    rows = len(samples)
    fig = plt.figure(figsize=(8, 3 * rows))
    for row, sample in enumerate(samples):
        for col, key in enumerate(["points", "recon"]):
            ax = fig.add_subplot(rows, 2, row * 2 + col + 1, projection="3d")
            pts = sample[key].detach().cpu()
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
            ax.set_title(f"{sample['class_name']} {key} CD={sample['chamfer']:.5f}")
            ax.set_axis_off()
            ax.view_init(elev=20, azim=35)
            lim = 1.1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ModelNet10PointClouds(args.cache_root, split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ae, classifier = load_models(args, device)

    total = 0
    correct_raw = 0
    correct_recon = 0
    chamfers = []
    latent_codes = []
    per_class = {
        name: {"count": 0, "chamfer_sum": 0.0, "raw_correct": 0, "recon_correct": 0}
        for name in MODELNET10_CLASSES
    }
    viz_samples = []
    class_viz_count = {name: 0 for name in MODELNET10_CLASSES}

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            labels = torch.as_tensor(batch["class_id"], device=device, dtype=torch.long)
            recon, z = ae(points)
            cd = chamfer_l2_per_sample(recon, points)
            raw_pred = classifier(points).argmax(dim=1)
            recon_pred = classifier(recon).argmax(dim=1)

            chamfers.append(cd.cpu())
            latent_codes.append(z.cpu())
            correct_raw += (raw_pred == labels).sum().item()
            correct_recon += (recon_pred == labels).sum().item()
            total += labels.numel()

            for i in range(points.shape[0]):
                class_name = batch["class_name"][i]
                per_class[class_name]["count"] += 1
                per_class[class_name]["chamfer_sum"] += float(cd[i].cpu())
                per_class[class_name]["raw_correct"] += int(raw_pred[i].item() == labels[i].item())
                per_class[class_name]["recon_correct"] += int(recon_pred[i].item() == labels[i].item())
                if class_viz_count[class_name] < args.viz_per_class:
                    viz_samples.append(
                        {
                            "class_name": class_name,
                            "points": points[i].cpu(),
                            "recon": recon[i].cpu(),
                            "chamfer": float(cd[i].cpu()),
                        }
                    )
                    class_viz_count[class_name] += 1

    chamfer_all = torch.cat(chamfers)
    z_all = torch.cat(latent_codes)
    z_norm = z_all.norm(dim=1)
    stats = bbox_stats(z_all[:, None, :])

    per_class_out = {}
    for name, item in per_class.items():
        count = max(item["count"], 1)
        per_class_out[name] = {
            "count": item["count"],
            "mean_chamfer": item["chamfer_sum"] / count,
            "raw_acc": item["raw_correct"] / count,
            "recon_acc": item["recon_correct"] / count,
        }

    metrics = {
        "num_test_samples": total,
        "mean_chamfer": float(chamfer_all.mean()),
        "median_chamfer": float(chamfer_all.median()),
        "p90_chamfer": float(chamfer_all.quantile(0.9)),
        "raw_classifier_acc": correct_raw / max(total, 1),
        "recon_classifier_acc": correct_recon / max(total, 1),
        "latent_norm_mean": float(z_norm.mean()),
        "latent_norm_std": float(z_norm.std(unbiased=False)),
        "latent_norm_min": float(z_norm.min()),
        "latent_norm_max": float(z_norm.max()),
        "latent_dim_min": float(stats["mins"].min()),
        "latent_dim_max": float(stats["maxs"].max()),
        "per_class": per_class_out,
    }

    metrics_path = args.output / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    render_recon_grid(viz_samples, args.output / "recon_grid.png")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {args.output / 'recon_grid.png'}")


if __name__ == "__main__":
    main()
