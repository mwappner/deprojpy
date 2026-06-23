import numpy as np

from deprojpy.geometry import fit_plane, polygon_metrics
from deprojpy.heightmap import compute_curvatures


def test_square_metrics():
    square = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], float)
    area, perimeter, area2d, perimeter2d = polygon_metrics(square)
    assert np.isclose(area, 4)
    assert np.isclose(perimeter, 8)
    assert np.isclose(area2d, 4)
    assert np.isclose(perimeter2d, 8)
    _, angles = fit_plane(square)
    assert np.isfinite(angles).all()


def test_flat_curvature_is_zero():
    fields = compute_curvatures(np.ones((40, 50)), object_scale=None, pixel_size=0.2)
    for field in fields:
        assert np.allclose(field, 0)


def test_tilted_plane_curvature_is_zero():
    y, x = np.mgrid[:40, :50]
    fields = compute_curvatures(2 * x + 3 * y, object_scale=None)
    for field in fields:
        assert np.allclose(field, 0, atol=1e-12)
