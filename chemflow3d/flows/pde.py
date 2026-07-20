from __future__ import annotations

import torch


def _as_time_batch(t: torch.Tensor, batch_size: int, like: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = like.new_tensor(float(t))
    t = t.to(device=like.device, dtype=like.dtype)
    if t.ndim == 0:
        return t.expand(batch_size).clone().requires_grad_(True)
    if t.ndim == 1 and t.shape[0] == 1:
        return t.expand(batch_size).clone().requires_grad_(True)
    return t.clone().requires_grad_(True)


def energy_and_derivatives(energy_fn, z: torch.Tensor, t: torch.Tensor):
    z = z.clone().requires_grad_(True)
    t = _as_time_batch(t, z.shape[0], z)
    u = energy_fn(z, t)
    (u_z,) = torch.autograd.grad(u.sum(), z, create_graph=True)
    (u_t,) = torch.autograd.grad(u.sum(), t, create_graph=True)
    return u, u_z, u_t, z, t


def exact_laplacian(u_z: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Exact per-sample Hessian trace. O(B*D) autograd calls; use for tests/small D."""
    traces = []
    for dim in range(z.shape[1]):
        grad_dim = u_z[:, dim].sum()
        (second,) = torch.autograd.grad(grad_dim, z, create_graph=True, retain_graph=True)
        traces.append(second[:, dim])
    return torch.stack(traces, dim=1).sum(dim=1)


def hutchinson_laplacian(u_z: torch.Tensor, z: torch.Tensor, samples: int = 1) -> torch.Tensor:
    estimates = []
    for _ in range(samples):
        eps = torch.empty_like(z).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        dot = (u_z * eps).sum()
        (hvp,) = torch.autograd.grad(dot, z, create_graph=True, retain_graph=True)
        estimates.append((hvp * eps).sum(dim=1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def wave_residual(
    energy_fn,
    z: torch.Tensor,
    t: torch.Tensor,
    speed: torch.Tensor | float = 1.0,
    laplacian: str = "hutchinson",
    hutchinson_samples: int = 1,
) -> torch.Tensor:
    """Per-sample Wave PDE residual u_tt - c^2 Delta u."""
    _, u_z, u_t, z_req, t_req = energy_and_derivatives(energy_fn, z, t)
    (u_tt,) = torch.autograd.grad(u_t.sum(), t_req, create_graph=True)
    if laplacian == "exact":
        lap = exact_laplacian(u_z, z_req)
    elif laplacian == "hutchinson":
        lap = hutchinson_laplacian(u_z, z_req, samples=hutchinson_samples)
    else:
        raise ValueError(f"Unknown laplacian mode: {laplacian}")
    return u_tt - torch.as_tensor(speed, device=z.device, dtype=z.dtype).square() * lap


def hamilton_jacobi_residual(energy_fn, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-sample HJ residual u_t + 0.5 ||grad_z u||^2."""
    _, u_z, u_t, _, _ = energy_and_derivatives(energy_fn, z, t)
    return u_t + 0.5 * u_z.square().sum(dim=1)
