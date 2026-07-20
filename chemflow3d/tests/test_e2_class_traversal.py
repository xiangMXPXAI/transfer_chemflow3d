import torch
from torch import nn

from chemflow3d.flows import EnergyField
from chemflow3d.flows.traversal import normalize_direction
from chemflow3d.scripts.eval_class_traversal import class_gradient_direction, target_margin, target_probability
from chemflow3d.scripts.train_flow import DecoderClassLogit, directional_guidance, energy_velocity


class TinyDecoder(nn.Module):
    def forward(self, z):
        return z[:, None, :3].repeat(1, 8, 1)


class TinyClassifier(nn.Module):
    def forward(self, points):
        pooled = points.mean(dim=1)
        return torch.stack([pooled[:, 0], pooled[:, 1], pooled[:, 2]], dim=1)


def _grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().abs().sum())
    return total


def test_target_margin_excludes_target_class():
    logits = torch.tensor([[1.0, 5.0, 3.0], [2.0, -1.0, 4.0]])
    margin = target_margin(logits, target_class=2)
    assert torch.allclose(margin, torch.tensor([-2.0, 2.0]))


def test_target_probability_shape():
    logits = torch.randn(4, 3)
    prob = target_probability(logits, target_class=1)
    assert prob.shape == (4,)
    assert torch.all((prob >= 0) & (prob <= 1))


def test_class_gradient_direction_shape_and_finite():
    z = torch.randn(5, 4)
    v = class_gradient_direction(TinyDecoder(), TinyClassifier(), z, target_class=1)
    assert v.shape == z.shape
    assert torch.isfinite(v).all()
    assert torch.allclose(v.norm(dim=1), torch.ones(5), atol=1e-6)


def test_class_guidance_backpropagates_to_energy():
    torch.manual_seed(0)
    energy = EnergyField(latent_dim=4, hidden_dim=16)
    z0 = torch.randn(3, 4)
    t = torch.zeros(3)
    z_req, v = energy_velocity(energy, z0, t)
    guidance = DecoderClassLogit(TinyDecoder(), TinyClassifier(), target_class=1)
    guide = directional_guidance(guidance, z_req, normalize_direction(v))
    (-guide).backward()
    assert _grad_norm(energy.parameters()) > 0.0
