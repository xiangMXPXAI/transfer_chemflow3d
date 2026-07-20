from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.flows import EnergyField, TraversalConfig, rollout
from chemflow3d.flows.traversal import property_gradient, random_direction
from chemflow3d.metrics.geometry import bbox_stats, geometric_properties
from chemflow3d.models import PointNetAE


PROP_INDEX = {"width": 0, "height": 1, "depth": 2, "volume": 3, "compactness": 4}
METHODS = ["random", "property_gradient", "wave", "hj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize E1 traversal with axis projections, bbox overlays and curves.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--ae-ckpt", type=Path, default=Path("chemflow3d_runs/e0_ae/best.pt"))
    parser.add_argument("--wave-ckpt", type=Path, required=True)
    parser.add_argument("--hj-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--property", choices=list(PROP_INDEX), required=True)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary(eval_root: Path, prop: str) -> dict[str, dict[str, float]]:
    out = {}
    for tag in METHODS:
        with (eval_root / f"{tag}_{prop}_summary.json").open("r", encoding="utf-8") as f:
            out[tag] = json.load(f)
    return out


def plot_summary(summary: dict[str, dict[str, float]], prop: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = [summary[t]["mean_delta"] for t in METHODS]
    decoded = [summary[t]["mean_decoded_path_length"] for t in METHODS]
    rates = [summary[t]["positive_delta_rate"] for t in METHODS]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].bar(METHODS, means)
    axes[0].set_title(f"Mean {prop} delta")
    axes[1].bar(METHODS, decoded)
    axes[1].set_title("Decoded path length")
    axes[2].bar(METHODS, rates)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Positive delta rate")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "summary_bars.png", dpi=180)
    plt.close(fig)


def plot_delta_hist(eval_root: Path, prop: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for tag in METHODS:
        rows = read_csv(eval_root / f"{tag}_{prop}.csv")
        deltas = [float(r["delta"]) for r in rows]
        ax.hist(deltas, bins=40, alpha=0.45, density=True, label=tag)
    ax.set_title(f"{prop} delta distribution")
    ax.set_xlabel("delta")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "delta_hist.png", dpi=180)
    plt.close(fig)


def load_ae(args: argparse.Namespace, device: torch.device) -> PointNetAE:
    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device, weights_only=False)["model"])
    ae.eval()
    return ae


def load_energy(path: Path, args: argparse.Namespace, device: torch.device) -> EnergyField:
    energy = EnergyField(args.latent_dim, num_flows=1, hidden_dim=args.hidden_dim).to(device)
    energy.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    energy.eval()
    return energy


def select_diverse_samples(ds: ModelNet10PointClouds, num_samples: int) -> list[dict]:
    selected = []
    seen = set()
    preferred = ["chair", "table", "sofa", "toilet", "bed", "monitor", "desk", "bathtub"]
    for target in preferred:
        for idx in range(len(ds)):
            item = ds[idx]
            if item["class_name"] == target and target not in seen:
                selected.append(item)
                seen.add(target)
                break
        if len(selected) >= num_samples:
            return selected
    for idx in range(len(ds)):
        item = ds[idx]
        if item["class_name"] not in seen:
            selected.append(item)
            seen.add(item["class_name"])
        if len(selected) >= num_samples:
            break
    return selected


def make_velocity(method: str, ae: PointNetAE, prop_idx: int, energy: EnergyField | None):
    if method == "random":
        return lambda z, t: random_direction(z, normalize=True)
    if method == "property_gradient":
        def prop_fn(z):
            return geometric_properties(ae.decode(z))[:, prop_idx : prop_idx + 1]
        return lambda z, t: property_gradient(prop_fn, z, normalize=True)
    if energy is None:
        raise ValueError("PDE method requires an energy model.")
    return lambda z, t: energy.velocity(0, z, t, create_graph=False)


def projection_limits(trajectories: dict[str, torch.Tensor]) -> dict[str, tuple[float, float]]:
    all_points = torch.cat([xs[:, 0].reshape(-1, 3).detach().cpu() for xs in trajectories.values()], dim=0)
    mins = all_points.min(dim=0).values
    maxs = all_points.max(dim=0).values
    center = (mins + maxs) / 2
    span = (maxs - mins).max().item()
    span = max(span, 1e-3) * 0.58
    return {
        "x": (float(center[0] - span), float(center[0] + span)),
        "y": (float(center[1] - span), float(center[1] + span)),
        "z": (float(center[2] - span), float(center[2] + span)),
    }


def draw_projection(ax, cloud: torch.Tensor, dims: tuple[int, int], limits: dict[str, tuple[float, float]], title: str) -> None:
    import matplotlib.patches as patches

    labels = ["x", "y", "z"]
    a, b = dims
    pts = cloud.detach().cpu()
    ax.scatter(pts[:, a], pts[:, b], s=2, alpha=0.7)
    mins = pts.min(dim=0).values
    maxs = pts.max(dim=0).values
    rect = patches.Rectangle(
        (float(mins[a]), float(mins[b])),
        float(maxs[a] - mins[a]),
        float(maxs[b] - mins[b]),
        fill=False,
        linewidth=1.4,
        edgecolor="crimson",
    )
    ax.add_patch(rect)
    ax.set_xlim(*limits[labels[a]])
    ax.set_ylim(*limits[labels[b]])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels[a])
    ax.set_ylabel(labels[b])
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)


def trajectory_props(xs: torch.Tensor) -> torch.Tensor:
    return geometric_properties(xs[:, 0]).detach().cpu()


def render_projection_trajectories(args: argparse.Namespace, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ModelNet10PointClouds(args.cache_root, split="test")
    selected = select_diverse_samples(ds, args.num_samples)
    points = torch.stack([item["points"] for item in selected], dim=0).to(device)
    class_names = [item["class_name"] for item in selected]
    object_ids = [item["object_id"] for item in selected]

    ae = load_ae(args, device)
    wave = load_energy(args.wave_ckpt, args, device)
    hj = load_energy(args.hj_ckpt, args, device)
    prop_idx = PROP_INDEX[args.property]
    cfg = TraversalConfig(steps=args.steps, step_size=args.step_size)

    with torch.no_grad():
        z0_all = ae.encode(points)

    model_map = {"wave": wave, "hj": hj}
    methods = ["random", "property_gradient", "wave", "hj"]
    time_ids = [0, args.steps // 2, args.steps]
    diagnostics = []

    for sample_idx in range(points.shape[0]):
        trajectories = {}
        for method in methods:
            velocity_fn = make_velocity(method, ae, prop_idx, model_map.get(method))
            _, xs = rollout(velocity_fn, ae.decode, z0_all[sample_idx : sample_idx + 1], cfg)
            trajectories[method] = xs.detach().cpu()

        limits = projection_limits(trajectories)
        fig = plt.figure(figsize=(16, 3.2 * len(methods)))
        for row, method in enumerate(methods):
            xs = trajectories[method]
            props = trajectory_props(xs)
            for col, t_id in enumerate(time_ids):
                ax = fig.add_subplot(len(methods), 5, row * 5 + col + 1)
                title = (
                    f"{method} XY t={t_id}\n"
                    f"w={props[t_id,0]:.2f}, h={props[t_id,1]:.2f}, V={props[t_id,3]:.2f}"
                )
                draw_projection(ax, xs[t_id, 0], (0, 1), limits, title)

            ax_xz = fig.add_subplot(len(methods), 5, row * 5 + 4)
            draw_projection(ax_xz, xs[-1, 0], (0, 2), limits, f"{method} XZ final\ndepth={props[-1,2]:.2f}")

            ax_curve = fig.add_subplot(len(methods), 5, row * 5 + 5)
            steps = list(range(xs.shape[0]))
            ax_curve.plot(steps, props[:, 0], label="width")
            ax_curve.plot(steps, props[:, 1], label="height")
            ax_curve.plot(steps, props[:, 2], label="depth")
            ax_curve.plot(steps, props[:, 3], label="volume")
            ax_curve.set_title(f"{method} property curves", fontsize=9)
            ax_curve.set_xlabel("step")
            ax_curve.grid(alpha=0.25)
            if row == 0:
                ax_curve.legend(fontsize=7)

            diagnostics.append(
                {
                    "sample_idx": sample_idx,
                    "object_id": object_ids[sample_idx],
                    "class_name": class_names[sample_idx],
                    "method": method,
                    "width_0": float(props[0, 0]),
                    "width_T": float(props[-1, 0]),
                    "height_0": float(props[0, 1]),
                    "height_T": float(props[-1, 1]),
                    "depth_0": float(props[0, 2]),
                    "depth_T": float(props[-1, 2]),
                    "volume_0": float(props[0, 3]),
                    "volume_T": float(props[-1, 3]),
                    "target_delta": float(props[-1, prop_idx] - props[0, prop_idx]),
                }
            )

        fig.suptitle(
            f"E1 {args.property} diagnostic: sample {sample_idx} "
            f"{class_names[sample_idx]} / {object_ids[sample_idx]}",
            y=0.995,
        )
        fig.tight_layout()
        fig.savefig(output / f"diagnostic_{sample_idx}_{class_names[sample_idx]}_{args.property}.png", dpi=180)
        plt.close(fig)

    with (output / "trajectory_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.eval_root, args.property)
    plot_summary(summary, args.property, args.output)
    plot_delta_hist(args.eval_root, args.property, args.output)
    render_projection_trajectories(args, args.output)
    with (args.output / "summary_used.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote E1 visualizations to {args.output}")


if __name__ == "__main__":
    main()
