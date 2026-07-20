import torch
from torch import nn

from chemflow3d.flows import EnergyField, hamilton_jacobi_residual, wave_residual
from chemflow3d.flows.traversal import normalize_direction
from chemflow3d.scripts.train_flow import directional_guidance, energy_velocity


class LinearGuidance(nn.Module):
    def forward(self, z):
        return z[:, :1]


def _grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().abs().sum())
    return total


def test_directional_guidance_backpropagates_to_energy():
    torch.manual_seed(0)
    energy = EnergyField(latent_dim=4, hidden_dim=16)
    z0 = torch.randn(3, 4)
    t = torch.zeros(3)
    z_req, v = energy_velocity(energy, z0, t)
    guide = directional_guidance(LinearGuidance(), z_req, normalize_direction(v))
    loss = -guide
    loss.backward()
    assert _grad_norm(energy.parameters()) > 0.0


def test_pde_losses_backpropagate_to_energy():
    torch.manual_seed(0)
    energy = EnergyField(latent_dim=3, hidden_dim=16)
    z0 = torch.randn(2, 3)
    t = torch.ones(2)

    def efn(z, tt):
        return energy.energy(0, z, tt)

    loss = hamilton_jacobi_residual(efn, z0, t).square().mean()
    loss = loss + wave_residual(efn, z0, t, laplacian="exact").square().mean()
    loss.backward()
    assert _grad_norm(energy.parameters()) > 0.0
