from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from chemflow3d.data import SyntheticSequenceDataset
from chemflow3d.flows import EnergyField
from chemflow3d.models import PointNetAE
from chemflow3d.scripts.eval_synthetic_flow import rollout_energy
from chemflow3d.scripts.train_synthetic_flow import encode_sequence


METHODS = ["wave", "hj", "none"]
LABELS = {"wave": "Wave PDE", "hj": "HJ PDE", "none": "No PDE"}
COLORS = {"wave": "#F58518", "hj": "#54A24B", "none": "#4C78A8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize E3 synthetic ground-truth flow results.")
    parser.add_argument("--eval-root", type=Path, default=Path("chemflow3d_runs/e3_eval_yaw"))
    parser.add_argument("--run-root", type=Path, default=Path("chemflow3d_runs"))
    parser.add_argument("--sequence-root", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences"))
    parser.add_argument("--index", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences/index_test_yaw.json"))
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e3_visuals_yaw"))
    parser.add_argument("--transform", type=str, default="yaw")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--representative-index", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_history(args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for method in METHODS:
        hist_path = args.run_root / f"e3_{method}_{args.transform}" / "history.json"
        hist = read_json(hist_path)
        epochs = [row["epoch"] for row in hist]
        axes[0, 0].plot(epochs, [row["loss"] for row in hist], label=LABELS[method], color=COLORS[method])
        axes[0, 1].plot(epochs, [row["velocity"] for row in hist], label=LABELS[method], color=COLORS[method])
        axes[1, 0].plot(epochs, [row["cosine"] for row in hist], label=LABELS[method], color=COLORS[method])
        axes[1, 1].plot(epochs, [row["pde"] for row in hist], label=LABELS[method], color=COLORS[method])

    axes[0, 0].set_title("Training loss")
    axes[0, 1].set_title("Velocity MSE")
    axes[1, 0].set_title("Velocity cosine")
    axes[1, 1].set_title("PDE residual loss")
    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(args.output / f"e3_{args.transform}_history.png", dpi=180)
    plt.close(fig)


def annotate_bars(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_summary(args: argparse.Namespace) -> None:
    summaries = {
        method: read_json(args.eval_root / f"{method}_{args.transform}_summary.json")
        for method in METHODS
    }
    labels = [LABELS[m] for m in METHODS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    endpoint = [summaries[m]["mean_endpoint_chamfer"] for m in METHODS]
    identity = summaries["wave"]["mean_identity_endpoint_chamfer"]
    oracle = summaries["wave"]["mean_oracle_endpoint_chamfer"]
    bars = axes[0, 0].bar(labels, endpoint, color=[COLORS[m] for m in METHODS])
    axes[0, 0].axhline(identity, color="0.2", linestyle="--", label="identity")
    axes[0, 0].axhline(oracle, color="0.45", linestyle=":", label="AE oracle")
    axes[0, 0].set_ylabel("mean endpoint Chamfer")
    axes[0, 0].set_title("Endpoint geometry error")
    axes[0, 0].legend(frameon=False)
    annotate_bars(axes[0, 0], bars)

    traj = [summaries[m]["mean_trajectory_chamfer"] for m in METHODS]
    bars = axes[0, 1].bar(labels, traj, color=[COLORS[m] for m in METHODS])
    axes[0, 1].set_ylabel("mean trajectory Chamfer")
    axes[0, 1].set_title("Full trajectory error")
    annotate_bars(axes[0, 1], bars)

    cosine = [summaries[m]["mean_mean_velocity_cosine"] for m in METHODS]
    bars = axes[1, 0].bar(labels, cosine, color=[COLORS[m] for m in METHODS])
    axes[1, 0].axhline(0.0, color="0.2", linewidth=0.8)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_ylabel("mean velocity cosine")
    axes[1, 0].set_title("Direction alignment")
    annotate_bars(axes[1, 0], bars)

    ratio = [summaries[m]["mean_identity_to_flow_ratio"] for m in METHODS]
    bars = axes[1, 1].bar(labels, ratio, color=[COLORS[m] for m in METHODS])
    axes[1, 1].axhline(1.0, color="0.2", linewidth=0.8)
    axes[1, 1].set_ylabel("identity / flow endpoint CD")
    axes[1, 1].set_title("Improvement over identity")
    annotate_bars(axes[1, 1], bars)

    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(args.output / f"e3_{args.transform}_summary.png", dpi=180)
    plt.close(fig)


def plot_distributions(args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    metrics = [
        ("endpoint_chamfer", "Endpoint Chamfer"),
        ("mean_velocity_cosine", "Velocity cosine"),
        ("latent_endpoint_l2", "Latent endpoint L2"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for method in METHODS:
            rows = read_csv(args.eval_root / f"{method}_{args.transform}.csv")
            vals = [float(row[key]) for row in rows]
            ax.hist(vals, bins=40, alpha=0.45, density=True, label=LABELS[method], color=COLORS[method])
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(args.output / f"e3_{args.transform}_distributions.png", dpi=180)
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


def set_equal_2d(ax, x, y):
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    r = max((xmax - xmin), (ymax - ymin), 1e-3) / 2
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal", adjustable="box")


def plot_representative(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SyntheticSequenceDataset(args.sequence_root, args.index)
    item = ds[min(args.representative_index, len(ds) - 1)]
    sequence = item["sequence"].unsqueeze(0).to(device)
    steps = sequence.shape[1] - 1
    ae = load_ae(args, device)
    z_seq = encode_sequence(ae, sequence)

    methods_with_gt = ["gt", *METHODS]
    frames = [0, steps // 2, steps]
    fig, axes = plt.subplots(len(methods_with_gt), len(frames), figsize=(12, 12), constrained_layout=True)
    fig.suptitle(f"E3 representative rollout: {item['object_id']} ({item['class_name']}, {args.transform})", fontsize=13)

    all_points = sequence[:, frames].reshape(-1, 3).detach().cpu()
    for method_idx, method in enumerate(methods_with_gt):
        if method == "gt":
            xs = sequence[0].detach().cpu()
            row_label = "Ground truth"
        else:
            flow_path = args.run_root / f"e3_{method}_{args.transform}" / "last.pt"
            energy, step_size = load_energy(flow_path, args, device)
            _, xs_pred = rollout_energy(energy, ae.decode, z_seq[:, 0], steps, step_size)
            xs = xs_pred[0].detach().cpu()
            row_label = LABELS[method]
        for col, frame in enumerate(frames):
            ax = axes[method_idx, col]
            pts = xs[frame]
            ax.scatter(pts[:, 0], pts[:, 2], s=2, alpha=0.7)
            set_equal_2d(ax, all_points[:, 0], all_points[:, 2])
            ax.set_title(f"{row_label} | t={frame}", fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.grid(alpha=0.2)
    fig.savefig(args.output / f"e3_{args.transform}_representative_rollout.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_history(args)
    plot_summary(args)
    plot_distributions(args)
    plot_representative(args)
    print(f"Wrote E3 visualizations to {args.output}")


if __name__ == "__main__":
    main()
