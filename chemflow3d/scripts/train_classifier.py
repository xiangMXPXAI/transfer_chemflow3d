from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.models import PointNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet classifier on cached ModelNet10 point clouds.")
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/e0_classifier"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ModelNet10PointClouds(args.cache_root, split="train")
    test_ds = ModelNet10PointClouds(args.cache_root, split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PointNetClassifier(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train classifier epoch {epoch}"):
            points = batch["points"].to(device)
            labels = torch.as_tensor(batch["class_id"], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = model(points)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * points.shape[0]
        total_loss /= len(train_ds)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                points = batch["points"].to(device)
                labels = torch.as_tensor(batch["class_id"], device=device, dtype=torch.long)
                pred = model(points).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        acc = correct / max(total, 1)
        print(f"epoch={epoch} train_loss={total_loss:.6f} test_acc={acc:.4f}")

        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "test_acc": acc}
        torch.save(ckpt, args.output / "last.pt")
        if acc > best_acc:
            best_acc = acc
            torch.save(ckpt, args.output / "best.pt")


if __name__ == "__main__":
    main()
