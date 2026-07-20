from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class EnergyMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256, time_dim: int = 32):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(z.shape[0])
        if t.ndim == 1 and t.shape[0] == 1:
            t = t.expand(z.shape[0])
        emb = self.time(t.to(dtype=z.dtype, device=z.device))
        return self.net(torch.cat([z, emb], dim=-1))


class EnergyField(nn.Module):
    """K independent scalar energy networks u_k(z,t)."""

    def __init__(self, latent_dim: int, num_flows: int = 1, hidden_dim: int = 256, time_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        self.mlps = nn.ModuleList(
            [EnergyMLP(latent_dim, hidden_dim=hidden_dim, time_dim=time_dim) for _ in range(num_flows)]
        )
        self.wave_speed = nn.Parameter(torch.ones(num_flows))

    def energy(self, flow_idx: int, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.mlps[flow_idx](z, t)

    def velocity(self, flow_idx: int, z: torch.Tensor, t: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        z_req = z.detach().requires_grad_(True) if not z.requires_grad else z
        u = self.energy(flow_idx, z_req, t)
        (v,) = torch.autograd.grad(u.sum(), z_req, create_graph=create_graph)
        return v
