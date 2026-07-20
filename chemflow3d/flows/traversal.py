from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TraversalConfig:
    steps: int = 10
    step_size: float = 0.1
    normalize: bool = True
    time_period: int = 10
    eps: float = 1e-8


def normalize_direction(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=1, keepdim=True) + eps)


def random_direction(z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    v = torch.randn_like(z)
    return normalize_direction(v) if normalize else v


def property_gradient(prop_fn, z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    z_req = z.detach().requires_grad_(True)
    prop = prop_fn(z_req)
    (v,) = torch.autograd.grad(prop.sum(), z_req)
    return normalize_direction(v) if normalize else v


@torch.no_grad()
def _decode(decoder, z: torch.Tensor) -> torch.Tensor:
    return decoder(z)


def rollout(velocity_fn, decoder, z0: torch.Tensor, cfg: TraversalConfig):
    """Roll out z_{t+1}=z_t+step*v(z_t,t), returning latent and decoded trajectories."""
    zs = [z0]
    xs = [_decode(decoder, z0)]
    z = z0
    for step in range(cfg.steps):
        t = z.new_full((z.shape[0],), float(step % max(cfg.time_period, 1)))
        v = velocity_fn(z, t)
        if cfg.normalize:
            v = normalize_direction(v)
        z = z + cfg.step_size * v
        zs.append(z)
        xs.append(_decode(decoder, z))
    return torch.stack(zs, dim=0), torch.stack(xs, dim=0)
