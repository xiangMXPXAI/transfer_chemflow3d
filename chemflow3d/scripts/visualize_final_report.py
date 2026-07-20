from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chemflow3d.data import SyntheticSequenceDataset
from chemflow3d.flows import EnergyField
from chemflow3d.models import PointNetAE
from chemflow3d.scripts.eval_synthetic_flow import rollout_energy
from chemflow3d.scripts.train_synthetic_flow import encode_sequence


METHOD_COLORS = {
    "random": "#9CA3AF",
    "classifier_gradient": "#4C78A8",
    "wave": "#F58518",
    "hj": "#54A24B",
    "none": "#4C78A8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create polished final figures for ChemFlow3D reports.")
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/final_figures"))
    parser.add_argument("--ae-ckpt", type=Path, default=Path("chemflow3d_runs/e0_ae/best.pt"))
    parser.add_argument("--sequence-root", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--representative-index", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def style_axes(ax, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    if grid:
        ax.grid(axis="y", color="#CBD5E1", alpha=0.35, linewidth=0.8)


def annotate(ax, bars, fmt="{:.2f}", dy=3) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )


def plot_framework(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 4.8), constrained_layout=True)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    nodes = [
        ("Point cloud\nx ∈ R^{N×3}", 0.06, "#DBEAFE"),
        ("Encoder\nz = E(x)", 0.22, "#E0F2FE"),
        ("Energy field\nuθ(z,t)", 0.38, "#FEF3C7"),
        ("Potential velocity\nvθ = ∇z uθ", 0.54, "#FFEDD5"),
        ("Euler rollout\nz_{t+1}=z_t+ηvθ", 0.70, "#DCFCE7"),
        ("Decoder\nx̂ = D(z)", 0.86, "#EDE9FE"),
    ]
    y = 0.58
    for label, x, color in nodes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.16", fc=color, ec="#94A3B8", lw=1.2),
            transform=ax.transAxes,
        )
    for i in range(len(nodes) - 1):
        x0 = nodes[i][1] + 0.065
        x1 = nodes[i + 1][1] - 0.065
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"),
            xycoords=ax.transAxes,
        )
    ax.text(
        0.5,
        0.20,
        "Wave / Hamilton-Jacobi / no-PDE regularization  ·  property, class, and synthetic ground-truth guidance",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
        transform=ax.transAxes,
    )
    fig.savefig(output / "fig_01_framework.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_dashboard(output: Path) -> None:
    e0 = read_json(Path("chemflow3d_runs/e0_eval/metrics.json"))
    e1 = read_json(Path("chemflow3d_runs/e1_visuals_all/combined_e1_summary.json"))
    e2_t2 = {m: read_json(Path(f"chemflow3d_runs/e2_eval_target_2/{m}_target_2_summary.json")) for m in ["random", "wave", "hj"]}
    e2_t8 = {m: read_json(Path(f"chemflow3d_runs/e2_eval_target_8/{m}_target_8_summary.json")) for m in ["random", "wave", "hj"]}
    e3_yaw = {m: read_json(Path(f"chemflow3d_runs/e3_eval_yaw/{m}_yaw_summary.json")) for m in ["wave", "hj", "none"]}
    e3_scale = {
        m: read_json(Path(f"chemflow3d_runs/e3_eval_scale_x/{m}_scale_x_summary.json"))
        for m in ["wave", "hj", "none"]
    }

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle("ChemFlow3D E0–E3 Experimental Summary", fontsize=18, color="#0F172A", fontweight="bold")

    ax = axes[0, 0]
    bars = ax.bar(
        ["Raw cls.", "Recon cls."],
        [100 * e0["raw_classifier_acc"], 100 * e0["recon_classifier_acc"]],
        color=["#2563EB", "#60A5FA"],
        width=0.58,
    )
    ax.set_title("E0 · Representation quality", fontsize=12, color="#0F172A")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    annotate(ax, bars, "{:.1f}%")
    ax.text(0.5, 12, f"mean CD={e0['mean_chamfer']:.4f}", ha="center", color="#475569")
    style_axes(ax)

    ax = axes[0, 1]
    props = ["height", "width", "volume"]
    x = np.arange(len(props))
    width = 0.26
    wave_vals = [e1[p]["wave"]["mean_delta"] for p in props]
    hj_vals = [e1[p]["hj"]["mean_delta"] for p in props]
    random_vals = [e1[p]["random"]["mean_delta"] for p in props]
    ax.bar(x - width, random_vals, width, label="Random", color=METHOD_COLORS["random"])
    ax.bar(x, wave_vals, width, label="Wave", color=METHOD_COLORS["wave"])
    ax.bar(x + width, hj_vals, width, label="HJ", color=METHOD_COLORS["hj"])
    ax.set_xticks(x)
    ax.set_xticklabels(props)
    ax.set_title("E1 · Geometry attribute increase", fontsize=12, color="#0F172A")
    ax.set_ylabel("Mean property delta")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)

    ax = axes[0, 2]
    labels = ["chair\nrandom", "chair\nwave", "chair\nHJ", "table\nrandom", "table\nwave", "table\nHJ"]
    vals = [
        100 * e2_t2["random"]["target_success_rate"],
        100 * e2_t2["wave"]["target_success_rate"],
        100 * e2_t2["hj"]["target_success_rate"],
        100 * e2_t8["random"]["target_success_rate"],
        100 * e2_t8["wave"]["target_success_rate"],
        100 * e2_t8["hj"]["target_success_rate"],
    ]
    colors = [METHOD_COLORS["random"], METHOD_COLORS["wave"], METHOD_COLORS["hj"]] * 2
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title("E2 · Class-target traversal", fontsize=12, color="#0F172A")
    ax.set_ylabel("Final target success (%)")
    ax.set_ylim(0, 100)
    annotate(ax, bars, "{:.1f}%")
    style_axes(ax)

    ax = axes[1, 0]
    labels = ["yaw\nWave", "yaw\nHJ", "yaw\nNo PDE", "scale_x\nWave", "scale_x\nHJ", "scale_x\nNo PDE"]
    vals = [
        e3_yaw["wave"]["mean_mean_velocity_cosine"],
        e3_yaw["hj"]["mean_mean_velocity_cosine"],
        e3_yaw["none"]["mean_mean_velocity_cosine"],
        e3_scale["wave"]["mean_mean_velocity_cosine"],
        e3_scale["hj"]["mean_mean_velocity_cosine"],
        e3_scale["none"]["mean_mean_velocity_cosine"],
    ]
    colors = [METHOD_COLORS["wave"], METHOD_COLORS["hj"], METHOD_COLORS["none"]] * 2
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title("E3 · Ground-truth direction alignment", fontsize=12, color="#0F172A")
    ax.set_ylabel("Velocity cosine")
    ax.set_ylim(0, 1.05)
    annotate(ax, bars, "{:.3f}")
    style_axes(ax)

    ax = axes[1, 1]
    vals = [
        e3_yaw["wave"]["mean_trajectory_chamfer"],
        e3_yaw["hj"]["mean_trajectory_chamfer"],
        e3_yaw["none"]["mean_trajectory_chamfer"],
        e3_scale["wave"]["mean_trajectory_chamfer"],
        e3_scale["hj"]["mean_trajectory_chamfer"],
        e3_scale["none"]["mean_trajectory_chamfer"],
    ]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title("E3 · Decoded trajectory error", fontsize=12, color="#0F172A")
    ax.set_ylabel("Mean trajectory Chamfer")
    annotate(ax, bars, "{:.3f}")
    style_axes(ax)

    ax = axes[1, 2]
    ax.axis("off")
    bullets = [
        "E0: AE/classifier baseline is reliable.",
        "E1: property traversal is stable for height, width, volume.",
        "E2: chair target is strong; table target is source-class dependent.",
        "E3: scale_x direction is much cleaner than yaw.",
        "Key limitation: scalar potential fields struggle with rotation-like flows.",
    ]
    ax.text(
        0.03,
        0.92,
        "Main takeaways",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        transform=ax.transAxes,
    )
    for i, b in enumerate(bullets):
        ax.text(0.04, 0.78 - i * 0.14, f"• {b}", fontsize=11, color="#334155", transform=ax.transAxes)

    fig.savefig(output / "fig_02_e0_e3_dashboard.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_ae(args: argparse.Namespace, device: torch.device) -> PointNetAE:
    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def load_energy(path: Path, args: argparse.Namespace, device: torch.device) -> tuple[EnergyField, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    latent_dim = int(ckpt_args.get("latent_dim", args.latent_dim))
    hidden_dim = int(ckpt_args.get("hidden_dim", args.hidden_dim))
    step_size = float(ckpt_args.get("step_size", 1.0))
    energy = EnergyField(latent_dim, num_flows=1, hidden_dim=hidden_dim).to(device)
    energy.load_state_dict(ckpt["model"])
    energy.eval()
    return energy, step_size


def set_equal_3d(ax, points: torch.Tensor) -> None:
    pts = points.detach().cpu()
    mins = pts.min(dim=0).values
    maxs = pts.max(dim=0).values
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max().item() / 2)
    radius = max(radius, 1e-3)
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))


def scatter_3d(ax, pts: torch.Tensor, title: str, all_pts: torch.Tensor) -> None:
    p = pts.detach().cpu()
    color = p[:, 2].numpy()
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=color, cmap="viridis", s=4, alpha=0.92, linewidths=0)
    set_equal_3d(ax, all_pts)
    ax.view_init(elev=21, azim=38)
    ax.set_title(title, fontsize=10, color="#0F172A")
    ax.set_xlabel("x", labelpad=-8)
    ax.set_ylabel("y", labelpad=-8)
    ax.set_zlabel("z", labelpad=-8)
    ax.tick_params(labelsize=7, pad=-2, colors="#64748B")
    ax.xaxis.pane.set_facecolor((0.97, 0.98, 1.0, 0.2))
    ax.yaxis.pane.set_facecolor((0.97, 0.98, 1.0, 0.2))
    ax.zaxis.pane.set_facecolor((0.97, 0.98, 1.0, 0.2))


def plot_e3_pointclouds(args: argparse.Namespace, output: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_ae(args, device)
    transforms = [
        ("yaw", "HJ", Path("chemflow3d_runs/e3_hj_yaw/last.pt")),
        ("scale_x", "HJ", Path("chemflow3d_runs/e3_hj_scale_x/last.pt")),
    ]
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("#F8FAFC")
    outer = fig.add_gridspec(2, 6)

    for row, (transform, method, ckpt) in enumerate(transforms):
        ds = SyntheticSequenceDataset(args.sequence_root, args.sequence_root / f"index_test_{transform}.json")
        item = ds[min(args.representative_index, len(ds) - 1)]
        seq = item["sequence"].unsqueeze(0).to(device)
        steps = seq.shape[1] - 1
        frames = [0, steps // 2, steps]
        z_seq = encode_sequence(ae, seq)
        energy, step_size = load_energy(ckpt, args, device)
        _, pred = rollout_energy(energy, ae.decode, z_seq[:, 0], steps, step_size)
        gt = seq[0].detach().cpu()
        pr = pred[0].detach().cpu()
        all_pts = torch.cat([gt[frames].reshape(-1, 3), pr[frames].reshape(-1, 3)], dim=0)

        for col, frame in enumerate(frames):
            ax = fig.add_subplot(outer[row, col], projection="3d")
            scatter_3d(ax, gt[frame], f"{transform} GT · t={frame}", all_pts)
        for col, frame in enumerate(frames):
            ax = fig.add_subplot(outer[row, col + 3], projection="3d")
            scatter_3d(ax, pr[frame], f"{transform} {method} flow · t={frame}", all_pts)

    fig.suptitle("E3 Synthetic Geometry: Ground Truth vs Learned Potential Flow", fontsize=16, fontweight="bold", color="#0F172A")
    fig.savefig(output / "fig_03_e3_3d_pointcloud_rollouts.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_framework(args.output)
    plot_dashboard(args.output)
    plot_e3_pointclouds(args, args.output)
    print(f"Wrote final report figures to {args.output}")


if __name__ == "__main__":
    main()
