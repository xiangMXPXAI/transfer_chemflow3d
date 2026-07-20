import torch

from chemflow3d.data.transforms import anisotropic_scale, translate, yaw_rotation
from chemflow3d.scripts.eval_synthetic_flow import velocity_alignment
from chemflow3d.scripts.train_synthetic_flow import adjacent_training_pairs, direction_loss


class ConstantVelocityEnergy(torch.nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.v_const = velocity

    def velocity(self, flow_idx, z, t, create_graph=False):
        return self.v_const.to(device=z.device, dtype=z.dtype).expand_as(z)


def test_synthetic_transforms_are_deterministic():
    points = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]])
    translated = translate(points, (0.5, 0.0, 0.0))
    scaled = anisotropic_scale(points, (2.0, 1.0, 1.0))
    rotated_zero = yaw_rotation(points, 0.0)

    assert torch.allclose(translated[..., 0], points[..., 0] + 0.5)
    assert torch.allclose(scaled[..., 0], points[..., 0] * 2.0)
    assert torch.allclose(rotated_zero, points)


def test_adjacent_training_pairs_match_euler_step():
    z_seq = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
            ]
        ]
    )
    z_t, t, v_gt = adjacent_training_pairs(z_seq, step_size=0.5)

    assert z_t.shape == (2, 2)
    assert torch.allclose(t, torch.tensor([0.0, 1.0]))
    assert torch.allclose(v_gt, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
    assert torch.allclose(z_t + 0.5 * v_gt, z_seq[:, 1:].reshape(2, 2))


def test_direction_loss_zero_for_aligned_vectors():
    v = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert torch.allclose(direction_loss(v, v), torch.tensor(0.0), atol=1e-6)


def test_velocity_alignment_detects_correct_constant_direction():
    z_seq = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            [
                [5.0, 1.0],
                [6.0, 1.0],
                [7.0, 1.0],
            ],
        ]
    )
    energy = ConstantVelocityEnergy(torch.tensor([1.0, 0.0]))
    mean_cos, pos_rate, mse = velocity_alignment(energy, z_seq, step_size=1.0)

    assert torch.allclose(mean_cos, torch.ones(2), atol=1e-6)
    assert torch.allclose(pos_rate, torch.ones(2), atol=1e-6)
    assert torch.allclose(mse, torch.zeros(2), atol=1e-6)
