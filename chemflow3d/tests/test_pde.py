import torch

from chemflow3d.flows.pde import exact_laplacian, hamilton_jacobi_residual, wave_residual


def test_exact_laplacian_quadratic():
    z = torch.randn(5, 7, requires_grad=True)
    u = 0.5 * z.square().sum(dim=1, keepdim=True)
    (u_z,) = torch.autograd.grad(u.sum(), z, create_graph=True)
    lap = exact_laplacian(u_z, z)
    assert torch.allclose(lap, torch.full((5,), 7.0), atol=1e-5)


def test_hj_residual_per_sample_shape():
    def energy(z, t):
        return (z.square().sum(dim=1, keepdim=True) + t[:, None])

    z = torch.randn(3, 4)
    t = torch.zeros(3)
    residual = hamilton_jacobi_residual(energy, z, t)
    assert residual.shape == (3,)
    assert torch.isfinite(residual).all()


def test_wave_residual_shape_with_exact_laplacian():
    def energy(z, t):
        return 0.5 * z.square().sum(dim=1, keepdim=True) + 0.5 * t[:, None].square()

    z = torch.randn(3, 4)
    t = torch.zeros(3)
    residual = wave_residual(energy, z, t, speed=1.0, laplacian="exact")
    assert residual.shape == (3,)
