from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chemflow3d.data import SyntheticSequenceDataset
from chemflow3d.flows import EnergyField, hamilton_jacobi_residual, wave_residual
from chemflow3d.flows.traversal import normalize_direction
from chemflow3d.models import PointNetAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ChemFlow3D energy field on synthetic ground-truth geometric sequences."
    )
    parser.add_argument("--sequence-root", type=Path, default=Path("chemflow3d_cache/modelnet10_sequences"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e3_flow"))
    parser.add_argument("--pde", choices=["wave", "hj", "none"], default="wave")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--lambda-velocity", type=float, default=1.0)
    parser.add_argument("--lambda-direction", type=float, default=0.1)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--laplacian", choices=["hutchinson", "exact"], default="hutchinson")
    parser.add_argument("--hutchinson-samples", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ae(args: argparse.Namespace, device: torch.device) -> PointNetAE:
    ae = PointNetAE(args.num_points, args.latent_dim).to(device)
    ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def encode_sequence(ae: PointNetAE, sequence: torch.Tensor) -> torch.Tensor:
    """Encode B x (T+1) x N x 3 sequence into B x (T+1) x D latent tensor."""
    bsz, frames, npts, dims = sequence.shape
    flat = sequence.reshape(bsz * frames, npts, dims)
    with torch.no_grad():
        z = ae.encode(flat)
    return z.reshape(bsz, frames, -1)


def adjacent_training_pairs(z_seq: torch.Tensor, step_size: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened z_t, time index t, and ground-truth velocity.

    The training target is finite-difference velocity in latent space:

        v_gt(z_t,t) = (z_{t+1} - z_t) / step_size

    so that an Euler rollout z_{t+1}=z_t+step_size*v(z_t,t) matches the
    encoded synthetic sequence when v=v_gt.
    """
    bsz, frames, latent_dim = z_seq.shape
    steps = frames - 1
    z_t = z_seq[:, :-1].reshape(bsz * steps, latent_dim)
    z_next = z_seq[:, 1:].reshape(bsz * steps, latent_dim)
    t = torch.arange(steps, device=z_seq.device, dtype=z_seq.dtype)[None, :].expand(bsz, steps).reshape(-1)
    v_gt = (z_next - z_t) / float(step_size)
    return z_t, t, v_gt


def energy_velocity(energy: EnergyField, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z_req = z.detach().requires_grad_(True)
    u = energy.energy(0, z_req, t)
    (v,) = torch.autograd.grad(u.sum(), z_req, create_graph=True)
    return v


def direction_loss(v_pred: torch.Tensor, v_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    valid = v_gt.norm(dim=1) > eps
    if not bool(valid.any()):
        return v_pred.new_tensor(0.0)
    pred = normalize_direction(v_pred[valid], eps=eps)
    target = normalize_direction(v_gt[valid], eps=eps)
    return 1.0 - (pred * target).sum(dim=1).mean()


def pde_loss_for_batch(
    args: argparse.Namespace,
    energy: EnergyField,
    z: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    if args.pde == "none" or args.lambda_pde <= 0.0:
        return z.new_tensor(0.0)

    def efn(zz, tt):
        return energy.energy(0, zz, tt)

    if args.pde == "wave":
        residual = wave_residual(
            efn,
            z,
            t,
            speed=energy.wave_speed[0],
            laplacian=args.laplacian,
            hutchinson_samples=args.hutchinson_samples,
        )
    elif args.pde == "hj":
        residual = hamilton_jacobi_residual(efn, z, t)
    else:
        raise ValueError(args.pde)
    return residual.square().mean()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SyntheticSequenceDataset(args.sequence_root, args.index)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    ae = load_ae(args, device)
    energy = EnergyField(args.latent_dim, num_flows=1, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(energy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    history = []
    for epoch in range(1, args.epochs + 1):
        energy.train()
        totals = {"loss": 0.0, "velocity": 0.0, "direction": 0.0, "pde": 0.0, "cosine": 0.0}
        seen = 0
        iterator = tqdm(loader, desc=f"train synthetic flow epoch {epoch}")
        for batch_idx, batch in enumerate(iterator):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            sequence = batch["sequence"].to(device)
            z_seq = encode_sequence(ae, sequence)
            z_t, t, v_gt = adjacent_training_pairs(z_seq, args.step_size)
            v_pred = energy_velocity(energy, z_t, t)

            velocity_loss = torch.nn.functional.mse_loss(v_pred, v_gt)
            dir_loss = direction_loss(v_pred, v_gt)
            pde = pde_loss_for_batch(args, energy, z_t, t)
            loss = (
                args.lambda_velocity * velocity_loss
                + args.lambda_direction * dir_loss
                + args.lambda_pde * pde
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(energy.parameters(), max_norm=args.max_grad_norm)
            opt.step()

            with torch.no_grad():
                valid = v_gt.norm(dim=1) > 1e-8
                cosine = (
                    (normalize_direction(v_pred[valid]) * normalize_direction(v_gt[valid])).sum(dim=1).mean()
                    if bool(valid.any())
                    else v_gt.new_tensor(0.0)
                )

            n = z_t.shape[0]
            seen += n
            totals["loss"] += float(loss.detach().cpu()) * n
            totals["velocity"] += float(velocity_loss.detach().cpu()) * n
            totals["direction"] += float(dir_loss.detach().cpu()) * n
            totals["pde"] += float(pde.detach().cpu()) * n
            totals["cosine"] += float(cosine.detach().cpu()) * n
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                vel=f"{velocity_loss.item():.4f}",
                cos=f"{cosine.item():.3f}",
            )

        scheduler.step()
        denom = max(seen, 1)
        epoch_stats = {k: v / denom for k, v in totals.items()}
        epoch_stats["epoch"] = epoch
        epoch_stats["lr"] = scheduler.get_last_lr()[0]
        history.append(epoch_stats)
        print(
            f"epoch={epoch} "
            f"loss={epoch_stats['loss']:.6f} "
            f"velocity={epoch_stats['velocity']:.6f} "
            f"direction={epoch_stats['direction']:.6f} "
            f"pde={epoch_stats['pde']:.6f} "
            f"cosine={epoch_stats['cosine']:.4f} "
            f"lr={epoch_stats['lr']:.2e}"
        )
        torch.save({"model": energy.state_dict(), "args": vars(args), "epoch": epoch}, args.output / "last.pt")
        with (args.output / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
