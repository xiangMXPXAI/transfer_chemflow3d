import torch

from chemflow3d.losses import chamfer_l2
from chemflow3d.metrics.geometry import bbox_stats, geometric_properties


def test_chamfer_zero_for_identical_clouds():
    x = torch.randn(2, 16, 3)
    assert torch.allclose(chamfer_l2(x, x), torch.tensor(0.0), atol=1e-6)


def test_geometric_properties_shape():
    x = torch.randn(4, 32, 3)
    props = geometric_properties(x)
    assert props.shape == (4, 5)
    assert torch.isfinite(props).all()


def test_bbox_properties_are_axis_aligned_extents():
    points = torch.tensor(
        [
            [
                [-1.0, -2.0, -3.0],
                [2.0, 4.0, 5.0],
                [0.5, 1.0, -1.0],
            ]
        ]
    )
    stats = bbox_stats(points)
    props = geometric_properties(points)

    assert torch.allclose(stats["width"], torch.tensor([3.0]))
    assert torch.allclose(stats["height"], torch.tensor([6.0]))
    assert torch.allclose(stats["depth"], torch.tensor([8.0]))
    assert torch.allclose(stats["volume"], torch.tensor([144.0]))
    assert torch.allclose(props[:, :4], torch.tensor([[3.0, 6.0, 8.0, 144.0]]))
