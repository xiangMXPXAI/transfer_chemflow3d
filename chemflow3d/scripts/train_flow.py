from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chemflow3d.data import ModelNet10PointClouds
from chemflow3d.flows import EnergyField, hamilton_jacobi_residual, wave_residual
from chemflow3d.flows.traversal import normalize_direction
from chemflow3d.losses.chamfer import chamfer_l2
from chemflow3d.metrics.geometry import geometric_properties
from chemflow3d.models import PointNetAE, PointNetClassifier


class GeometryProperty(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return geometric_properties(points)[:, self.index : self.index + 1]


class DecoderProperty(nn.Module):
    def __init__(self, decoder: nn.Module, prop: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.prop = prop

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.prop(self.decoder(z))


class DecoderClassLogit(nn.Module):
    def __init__(self, decoder: nn.Module, classifier: nn.Module, target_class: int):
        super().__init__()
        self.decoder = decoder
        self.classifier = classifier
        self.target_class = target_class

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.decoder(z))
        return logits[:, self.target_class : self.target_class + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChemFlow3D energy field with Wave/HJ PDE and JVP guidance.")
    parser.add_argument("--cache-root", type=Path, default=Path("chemflow3d_cache/modelnet10_1024"))
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--classifier-ckpt", type=Path)
    parser.add_argument("--output", type=Path, default=Path("chemflow3d_runs/flow"))
    parser.add_argument("--pde", choices=["wave", "hj", "none"], default="wave")
    parser.add_argument(
        "--guidance",
        choices=["height", "width", "depth", "volume", "compactness", "class"],
        default="height",
    )
    parser.add_argument("--target-class", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--lambda-pde", type=float, default=1.0)
    parser.add_argument("--lambda-guide", type=float, default=1.0)
    parser.add_argument("--lambda-ic", type=float, default=0.1)
    parser.add_argument("--lambda-decoded-step", type=float, default=0.0)
    parser.add_argument("--laplacian", choices=["hutchinson", "exact"], default="hutchinson")
    parser.add_argument("--hutchinson-samples", type=int, default=1)
    parser.add_argument("--normalize-guide", action="store_true", dest="normalize_guide", default=True)
    parser.add_argument("--no-normalize-guide", action="store_false", dest="normalize_guide")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ae(args: argparse.Namespace, device: torch.device) -> PointNetAE:
    model = PointNetAE(args.num_points, args.latent_dim).to(device)
    ckpt = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_guidance(args: argparse.Namespace, ae: PointNetAE, device: torch.device) -> nn.Module:
    prop_map = {"width": 0, "height": 1, "depth": 2, "volume": 3, "compactness": 4}
    if args.guidance != "class":
        return DecoderProperty(ae.decode, GeometryProperty(prop_map[args.guidance])).to(device)

    if args.classifier_ckpt is None:
        raise ValueError("--classifier-ckpt is required when --guidance class")

    classifier = PointNetClassifier(num_classes=10).to(device)
    ckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt["model"])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)
    return DecoderClassLogit(ae.decode, classifier, args.target_class).to(device)


def energy_velocity(energy: EnergyField, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_req = z.detach().requires_grad_(True)
    u = energy.energy(0, z_req, t)
    (v,) = torch.autograd.grad(u.sum(), z_req, create_graph=True)
    return z_req, v


def directional_guidance(guidance: nn.Module, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ChemFlow-style scalar JVP: d/d eps guidance(z + eps v)|eps=0."""
    prop = guidance(z)
    if prop.ndim == 1:
        prop = prop[:, None]
    (grad_prop,) = torch.autograd.grad(prop.sum(), z, create_graph=True)
    return (grad_prop * v).sum(dim=1).mean()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ModelNet10PointClouds(args.cache_root, split="train")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    ae = load_ae(args, device)
    guidance = make_guidance(args, ae, device)
    energy = EnergyField(args.latent_dim, num_flows=1, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(energy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(1, args.epochs + 1):
        energy.train()
        running_loss = 0.0
        running_pde = 0.0
        running_guide = 0.0
        running_ic = 0.0
        running_step = 0.0
        seen = 0

        iterator = tqdm(loader, desc=f"train flow epoch {epoch}")
        for batch_idx, batch in enumerate(iterator):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            points = batch["points"].to(device)
            with torch.no_grad():
                z0 = ae.encode(points)

            t = torch.randint(
                low=0,
                high=max(args.time_steps, 1),
                size=(z0.shape[0],),
                device=device,
                dtype=z0.dtype,
            )

            def efn(zz, tt):
                return energy.energy(0, zz, tt)

            z_req, v_raw = energy_velocity(energy, z0, t)
            v_guide = normalize_direction(v_raw) if args.normalize_guide else v_raw

            if args.pde == "wave":
                pde_loss = wave_residual(
                    efn,
                    z0,
                    t,
                    speed=energy.wave_speed[0],
                    laplacian=args.laplacian,
                    hutchinson_samples=args.hutchinson_samples,
                ).square().mean()
            elif args.pde == "hj":
                pde_loss = hamilton_jacobi_residual(efn, z0, t).square().mean()
            else:
                pde_loss = z0.new_tensor(0.0)

            t0 = z0.new_zeros(z0.shape[0])
            _, v_ic = energy_velocity(energy, z0, t0)
            ic_loss = v_ic.square().mean()

            guide = directional_guidance(guidance, z_req, v_guide)

            decoded_step_loss = z0.new_tensor(0.0)
            if args.lambda_decoded_step > 0.0:
                z1 = z_req + args.step_size * v_guide
                decoded_step_loss = chamfer_l2(ae.decode(z1), ae.decode(z_req).detach())

            loss = (
                args.lambda_pde * pde_loss
                + args.lambda_ic * ic_loss
                - args.lambda_guide * guide
                + args.lambda_decoded_step * decoded_step_loss
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(energy.parameters(), max_norm=args.max_grad_norm)
            opt.step()

            n = points.shape[0]
            seen += n
            running_loss += loss.item() * n
            running_pde += pde_loss.item() * n
            running_guide += guide.item() * n
            running_ic += ic_loss.item() * n
            running_step += decoded_step_loss.item() * n
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                pde=f"{pde_loss.item():.4f}",
                guide=f"{guide.item():.4f}",
            )

        scheduler.step()
        denom = max(seen, 1)
        print(
            f"epoch={epoch} "
            f"loss={running_loss / denom:.6f} "
            f"pde={running_pde / denom:.6f} "
            f"guide={running_guide / denom:.6f} "
            f"ic={running_ic / denom:.6f} "
            f"decoded_step={running_step / denom:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        torch.save({"model": energy.state_dict(), "args": vars(args), "epoch": epoch}, args.output / "last.pt")


if __name__ == "__main__":
    main()
