from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.losses import chamfer_l2
from chemflow3d.models import PointNetAE, PointNetVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet AE/VAE on cached ModelNet10 point clouds.")
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e0_ae"))
    parser.add_argument("--model", choices=["ae", "vae"], default="ae")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def batch_points(batch: dict, device: torch.device) -> torch.Tensor:
    return batch["points"].to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ModelNet10PointClouds(args.cache_root, split="train")
    test_ds = ModelNet10PointClouds(args.cache_root, split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PointNetAE(args.num_points, args.latent_dim) if args.model == "ae" else PointNetVAE(args.num_points, args.latent_dim)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train ae epoch {epoch}"):
            points = batch_points(batch, device)
            opt.zero_grad(set_to_none=True)
            if args.model == "ae":
                recon, _ = model(points)
                loss = chamfer_l2(recon, points)
            else:
                recon, mu, logvar, _ = model(points)
                rec = chamfer_l2(recon, points)
                kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
                loss = rec + args.beta * kl
            loss.backward()
            opt.step()
            train_loss += loss.item() * points.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                points = batch_points(batch, device)
                if args.model == "ae":
                    recon, _ = model(points)
                else:
                    recon, _, _, _ = model(points)
                loss = chamfer_l2(recon, points)
                val_loss += loss.item() * points.shape[0]
        val_loss /= len(test_ds)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_chamfer={val_loss:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_chamfer": val_loss,
        }
        torch.save(ckpt, args.output / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.output / "best.pt")


if __name__ == "__main__":
    main()
