from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.data.modelnet10 import MODELNET10_CLASSES
from chemflow3d.flows import EnergyField, TraversalConfig, rollout
from chemflow3d.flows.traversal import normalize_direction, random_direction
from chemflow3d.models import PointNetAE, PointNetClassifier
from chemflow3d.scripts.eval_class_traversal import target_margin, target_probability


METHODS = ["random", "class_gradient", "wave", "hj"]
METHOD_LABELS = {
    "random": "Random",
    "class_gradient": "Classifier grad.",
    "wave": "Wave PDE",
    "hj": "HJ PDE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize E2 class-target latent-flow results.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--classifier-ckpt", type=Path, required=True)
    parser.add_argument("--wave-ckpt", type=Path, required=True)
    parser.add_argument("--hj-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trajectories", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_summaries(eval_root: Path, target_class: int) -> dict[str, dict]:
    out = {}
    missing = []
    for method in METHODS:
        path = eval_root / f"{method}_target_{target_class}_summary.json"
        if not path.exists():
            missing.append(str(path))
            continue
        out[method] = read_json(path)
    if missing:
        raise FileNotFoundError("Missing E2 summary files:\n" + "\n".join(missing))
    return out


def load_rows(eval_root: Path, target_class: int) -> dict[str, list[dict[str, str]]]:
    out = {}
    missing = []
    for method in METHODS:
        path = eval_root / f"{method}_target_{target_class}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        out[method] = read_csv(path)
    if missing:
        raise FileNotFoundError("Missing E2 csv files:\n" + "\n".join(missing))
    return out


def annotate_bars(ax, bars, fmt="{:.1f}%"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_summary_bars(summaries: dict[str, dict], output: Path, target_class: int) -> None:
    labels = [METHOD_LABELS[m] for m in METHODS]
    success = [100.0 * summaries[m]["target_success_rate"] for m in METHODS]
    new_success = [100.0 * summaries[m]["new_success_rate"] for m in METHODS]
    margin = [summaries[m]["mean_margin_delta"] for m in METHODS]
    prob = [summaries[m]["mean_target_prob_delta"] for m in METHODS]
    path_len = [summaries[m]["mean_decoded_path_length"] for m in METHODS]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        f"E2 target={target_class} ({MODELNET10_CLASSES[target_class]}): global traversal metrics",
        fontsize=14,
    )

    bars = axes[0, 0].bar(labels, success, color=["0.75", "#4C78A8", "#F58518", "#54A24B"])
    axes[0, 0].plot(labels, new_success, color="black", marker="o", linewidth=1.2, label="new success")
    axes[0, 0].set_ylabel("final target success (%)")
    axes[0, 0].set_ylim(0, max(100.0, max(success) * 1.18))
    axes[0, 0].legend(frameon=False)
    annotate_bars(axes[0, 0], bars)

    bars = axes[0, 1].bar(labels, margin, color=["0.75", "#4C78A8", "#F58518", "#54A24B"])
    axes[0, 1].axhline(0, color="0.2", linewidth=0.8)
    axes[0, 1].set_ylabel("mean target margin delta")
    annotate_bars(axes[0, 1], bars, fmt="{:.2f}")

    bars = axes[1, 0].bar(labels, prob, color=["0.75", "#4C78A8", "#F58518", "#54A24B"])
    axes[1, 0].axhline(0, color="0.2", linewidth=0.8)
    axes[1, 0].set_ylabel("mean target probability delta")
    annotate_bars(axes[1, 0], bars, fmt="{:.2f}")

    bars = axes[1, 1].bar(labels, path_len, color=["0.75", "#4C78A8", "#F58518", "#54A24B"])
    axes[1, 1].set_ylabel("mean decoded path length")
    annotate_bars(axes[1, 1], bars, fmt="{:.4f}")

    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=18)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / f"target_{target_class}_summary_bars.png", dpi=180)
    plt.close(fig)


def plot_source_success(rows: dict[str, list[dict[str, str]]], output: Path, target_class: int) -> None:
    source_ids = sorted({int(row["source_class_id"]) for method_rows in rows.values() for row in method_rows})
    x = torch.arange(len(source_ids), dtype=torch.float32)
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    for offset, method in enumerate(METHODS):
        values = []
        for source_id in source_ids:
            subset = [r for r in rows[method] if int(r["source_class_id"]) == source_id]
            values.append(100.0 * sum(int(r["target_success"]) for r in subset) / max(len(subset), 1))
        ax.bar(
            (x + (offset - 1.5) * width).numpy(),
            values,
            width=width,
            label=METHOD_LABELS[method],
        )
    ax.set_title(f"E2 target={target_class} ({MODELNET10_CLASSES[target_class]}): success by source class")
    ax.set_ylabel("final target success (%)")
    ax.set_xticks(x.numpy())
    ax.set_xticklabels([f"{i}:{MODELNET10_CLASSES[i]}" for i in source_ids], rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False)
    fig.savefig(output / f"target_{target_class}_source_success.png", dpi=180)
    plt.close(fig)


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


def load_energy(path: Path, args: argparse.Namespace, device: torch.device) -> EnergyField:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    latent_dim = int(ckpt_args.get("latent_dim", args.latent_dim))
    hidden_dim = int(ckpt_args.get("hidden_dim", args.hidden_dim))
    energy = EnergyField(latent_dim, num_flows=1, hidden_dim=hidden_dim).to(device)
    energy.load_state_dict(ckpt["model"])
    energy.eval()
    return energy


def class_gradient_direction(
    decoder,
    classifier: PointNetClassifier,
    z: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    z_req = z.detach().requires_grad_(True)
    logits = classifier(decoder(z_req))
    score = logits[:, target_class].sum()
    (v,) = torch.autograd.grad(score, z_req)
    return normalize_direction(v)


def make_velocity(method: str, ae: PointNetAE, classifier: PointNetClassifier, target_class: int, energies: dict[str, EnergyField]):
    if method == "random":
        return lambda z, t: random_direction(z, normalize=True)
    if method == "class_gradient":
        return lambda z, t: class_gradient_direction(ae.decode, classifier, z, target_class)
    return lambda z, t: energies[method].velocity(0, z, t, create_graph=False)


def choose_representatives(rows: dict[str, list[dict[str, str]]], target_class: int, n: int) -> list[tuple[str, int, str]]:
    """Prefer samples where both PDE methods succeed, spanning different source classes."""
    by_object = defaultdict(dict)
    meta = {}
    for method, method_rows in rows.items():
        for r in method_rows:
            object_id = r["object_id"]
            by_object[object_id][method] = int(r["target_success"])
            meta[object_id] = (int(r["source_class_id"]), r["source_class_name"])

    picked = []
    used_sources = set()
    for object_id, result in by_object.items():
        if result.get("wave") == 1 and result.get("hj") == 1:
            source_id, source_name = meta[object_id]
            if source_id == target_class or source_id in used_sources:
                continue
            picked.append((object_id, source_id, source_name))
            used_sources.add(source_id)
        if len(picked) >= n:
            return picked

    for object_id in by_object:
        source_id, source_name = meta[object_id]
        if source_id != target_class and source_id not in used_sources:
            picked.append((object_id, source_id, source_name))
            used_sources.add(source_id)
        if len(picked) >= n:
            break
    return picked


def find_dataset_item(ds: ModelNet10PointClouds, object_id: str) -> dict:
    for i in range(len(ds)):
        item = ds[i]
        if item["object_id"] == object_id:
            return item
    raise KeyError(f"object_id not found in dataset: {object_id}")


def set_equal_axes(ax, points: torch.Tensor) -> None:
    arr = points.detach().cpu()
    mins = arr.min(dim=0).values
    maxs = arr.max(dim=0).values
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max().item() / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))


def plot_point_cloud_trajectory(
    xs: torch.Tensor,
    logits: torch.Tensor,
    target_class: int,
    method: str,
    object_id: str,
    source_name: str,
    output: Path,
) -> None:
    steps = [0, xs.shape[0] // 2, xs.shape[0] - 1]
    probs = target_probability(logits, target_class).detach().cpu()
    margins = target_margin(logits, target_class).detach().cpu()
    preds = logits.argmax(dim=1).detach().cpu().tolist()

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    fig.suptitle(
        f"{METHOD_LABELS[method]} | {source_name} → {MODELNET10_CLASSES[target_class]} | {object_id}",
        fontsize=12,
    )

    merged_points = xs[steps].reshape(-1, 3)
    for col, step in enumerate(steps):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        pts = xs[step].detach().cpu()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.75)
        set_equal_axes(ax, merged_points)
        ax.set_title(
            f"t={step} pred={MODELNET10_CLASSES[preds[step]]}\n"
            f"p={probs[step]:.3f}, margin={margins[step]:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=20, azim=35)

    ts = list(range(xs.shape[0]))
    ax_prob = fig.add_subplot(gs[1, 0:2])
    ax_prob.plot(ts, probs.numpy(), marker="o", label="target probability")
    ax_prob.set_xlabel("rollout step")
    ax_prob.set_ylabel("target probability")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.grid(alpha=0.25)
    ax_prob.legend(frameon=False)

    ax_margin = fig.add_subplot(gs[1, 2])
    ax_margin.plot(ts, margins.numpy(), marker="o", color="#F58518", label="target margin")
    ax_margin.axhline(0.0, color="0.25", linewidth=0.8)
    ax_margin.set_xlabel("rollout step")
    ax_margin.set_ylabel("target margin")
    ax_margin.grid(alpha=0.25)
    ax_margin.legend(frameon=False)

    safe_object = object_id.replace("/", "_").replace("\\", "_")
    fig.savefig(output / f"target_{target_class}_{safe_object}_{method}_trajectory.png", dpi=180)
    plt.close(fig)


def plot_representative_trajectories(
    args: argparse.Namespace,
    rows: dict[str, list[dict[str, str]]],
    output: Path,
) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ModelNet10PointClouds(args.cache_root, split="test")
    ae = load_ae(args, device)
    classifier = load_classifier(args, device)
    energies = {
        "wave": load_energy(args.wave_ckpt, args, device),
        "hj": load_energy(args.hj_ckpt, args, device),
    }
    cfg = TraversalConfig(steps=args.steps, step_size=args.step_size, normalize=True)

    reps = choose_representatives(rows, args.target_class, args.num_trajectories)
    rep_info = []
    for object_id, source_id, source_name in reps:
        item = find_dataset_item(ds, object_id)
        points = item["points"].unsqueeze(0).to(device)
        with torch.no_grad():
            z0 = ae.encode(points)
        rep_info.append({"object_id": object_id, "source_class_id": source_id, "source_class_name": source_name})

        for method in METHODS:
            velocity_fn = make_velocity(method, ae, classifier, args.target_class, energies)
            zs, xs = rollout(velocity_fn, ae.decode, z0, cfg)
            with torch.no_grad():
                logits = classifier(xs[:, 0])
            plot_point_cloud_trajectory(
                xs[:, 0],
                logits,
                args.target_class,
                method,
                object_id,
                source_name,
                output,
            )

    with (output / f"target_{args.target_class}_representatives.json").open("w", encoding="utf-8") as f:
        json.dump(rep_info, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(args.eval_root, args.target_class)
    rows = load_rows(args.eval_root, args.target_class)
    plot_summary_bars(summaries, args.output, args.target_class)
    plot_source_success(rows, args.output, args.target_class)
    plot_representative_trajectories(args, rows, args.output)
    print(f"Wrote E2 visualizations to {args.output}")


if __name__ == "__main__":
    main()
