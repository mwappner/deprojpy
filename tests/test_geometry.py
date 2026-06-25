import numpy as np

from deprojpy.geometry import fit_plane, polygon_metrics, simplify_boundary_pixels
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


def test_simplify_boundary_pixels_keeps_square_usable():
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    simplified = simplify_boundary_pixels(square)
    assert simplified.shape[1] == 2
    assert len(simplified) >= 4
    assert np.isfinite(simplified).all()
    assert not np.allclose(simplified[0], simplified[-1])


def test_simplify_boundary_pixels_reduces_oversampled_rectangle():
    top = np.column_stack([np.arange(0, 11), np.zeros(11)])
    right = np.column_stack([np.full(5, 10), np.arange(1, 6)])
    bottom = np.column_stack([np.arange(9, -1, -1), np.full(10, 5)])
    left = np.column_stack([np.zeros(4), np.arange(4, 0, -1)])
    rectangle = np.vstack([top, right, bottom, left]).astype(float)

    simplified = simplify_boundary_pixels(rectangle, tolerance_px=0.5)

    assert len(simplified) < len(rectangle)
    assert np.isclose(simplified[:, 0].min(), 0)
    assert np.isclose(simplified[:, 0].max(), 10)
    assert np.isclose(simplified[:, 1].min(), 0)
    assert np.isclose(simplified[:, 1].max(), 5)


def test_simplify_boundary_pixels_tiny_boundary_does_not_collapse():
    triangle = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    simplified = simplify_boundary_pixels(triangle, tolerance_px=10.0)
    assert np.array_equal(simplified, triangle)
    assert len(simplified) == 3


def test_simplify_boundary_pixels_tolerance_is_monotonic():
    wave = np.column_stack(
        [
            np.linspace(0, 12, 60),
            np.sin(np.linspace(0, 4 * np.pi, 60)),
        ]
    )
    boundary = np.vstack([wave, [12, 3], [0, 3]])

    loose = simplify_boundary_pixels(boundary, tolerance_px=1.0)
    tight = simplify_boundary_pixels(boundary, tolerance_px=0.1)

    assert len(loose) <= len(tight)
