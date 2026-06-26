from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deprojpy.models import Epicell
from deprojpy.surface_distance import (
    SurfaceDistanceCalculator,
    SurfaceGraph,
    _bilinear_sample,
    cell_centers_xy_pixels,
    choose_surface_graph_step,
    sample_height_at_xy,
    surface_straight_distance,
)


def test_flat_surface_straight_distance_equals_2d_distance():
    heightmap = np.zeros((10, 10), dtype=float)
    distance = surface_straight_distance(heightmap, np.array([1, 2]), np.array([4, 6]))
    assert np.isclose(distance, 5.0)


def test_straight_distance_on_planar_slope_matches_analytic_distance():
    y, x = np.mgrid[:20, :20]
    heightmap = 0.25 * x + 0.5 * y
    p0 = np.array([2.0, 3.0])
    p1 = np.array([10.0, 7.0])
    dz = (0.25 * (p1[0] - p0[0])) + (0.5 * (p1[1] - p0[1]))
    expected = np.sqrt(np.sum((p1 - p0) ** 2) + dz**2)
    measured = surface_straight_distance(heightmap, p0, p1, n_samples=64)
    assert np.isclose(measured, expected, rtol=1e-3)


def test_numba_bilinear_sampler_matches_map_coordinates():
    y, x = np.mgrid[:5, :6]
    heightmap = (10 * y + x).astype(float)
    points = np.array(
        [
            [0.0, 0.0],
            [2.25, 1.5],
            [5.0, 4.0],
            [-0.2, 3.7],
            [5.4, -1.0],
        ],
        dtype=float,
    )
    expected = sample_height_at_xy(heightmap, points, order=1, mode="nearest")
    measured = np.array([_bilinear_sample(heightmap, x, y) for x, y in points])
    assert np.allclose(measured, expected)


def test_pairwise_distance_matrix_matches_individual_calls():
    y, x = np.mgrid[:12, :12]
    calc = SurfaceDistanceCalculator((0.1 * x + 0.2 * y).astype(float), pixel_size=0.5)
    points = np.array([[1, 1], [4, 1], [4, 5], [8, 6]], dtype=float)

    matrix = calc.straight_pairwise_distances(points, n_samples=32)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0)
    for i, j in [(0, 1), (0, 3), (2, 3)]:
        expected = calc.straight_distance(points[i], points[j], n_samples=32)
        assert np.isclose(matrix[i, j], expected)


def test_selected_pair_distances_match_individual_calls():
    heightmap = np.zeros((8, 8), dtype=float)
    calc = SurfaceDistanceCalculator(heightmap, pixel_size=1.0)
    points = np.array([[0, 0], [3, 4], [7, 7]], dtype=float)
    pairs = np.array([[0, 1], [1, 2]])

    distances = calc.straight_distances_for_pairs(points, pairs, n_samples=16)

    assert np.allclose(
        distances,
        [
            calc.straight_distance(points[0], points[1], n_samples=16),
            calc.straight_distance(points[1], points[2], n_samples=16),
        ],
    )


def test_surface_graph_flat_8_connectivity_is_exact_for_diagonal():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((6, 6), dtype=float),
        pixel_size=1.0,
        connectivity="8",
        step=1,
    )
    distance = graph.distance(np.array([0, 0]), np.array([5, 5]))
    assert np.isclose(distance, 5 * np.sqrt(2))


def test_surface_graph_flat_4_connectivity_is_manhattan_like():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((6, 6), dtype=float),
        pixel_size=1.0,
        connectivity="4",
        step=1,
    )
    distance = graph.distance(np.array([0, 0]), np.array([5, 3]))
    assert np.isclose(distance, 8.0)


def test_surface_graph_16_connectivity_reduces_grid_bias():
    heightmap = np.zeros((10, 10), dtype=float)
    p0 = np.array([0, 0])
    p1 = np.array([7, 3])
    d4 = SurfaceGraph.from_heightmap(
        heightmap,
        pixel_size=1.0,
        connectivity="4",
        step=1,
    ).distance(p0, p1)
    d8 = SurfaceGraph.from_heightmap(
        heightmap,
        pixel_size=1.0,
        connectivity="8",
        step=1,
    ).distance(p0, p1)
    d16 = SurfaceGraph.from_heightmap(
        heightmap,
        pixel_size=1.0,
        connectivity="16",
        step=1,
    ).distance(p0, p1)
    assert d16 < d8 < d4


def test_choose_surface_graph_step_small_and_large_images():
    assert choose_surface_graph_step((20, 20), target_nodes=80_000) == 1
    assert choose_surface_graph_step((1000, 1000), target_nodes=10_000) > 1


def test_surface_graph_distance_return_path():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((8, 8), dtype=float),
        pixel_size=1.0,
        connectivity="8",
        step=1,
    )
    distance, path = graph.distance(np.array([0, 0]), np.array([5, 5]), return_path=True)
    assert np.isfinite(distance)
    assert path.shape[1] == 2
    assert np.allclose(path[0], [0, 0])
    assert np.allclose(path[-1], [5, 5])


def test_surface_graph_distances_from_source_matches_individual_calls():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((8, 8), dtype=float),
        pixel_size=1.0,
        connectivity="8",
        step=1,
    )
    source = np.array([0, 0])
    targets = np.array([[3, 0], [3, 3], [5, 2]], dtype=float)
    distances = graph.distances_from_source(source, targets)
    expected = np.array([graph.distance(source, target) for target in targets])
    assert np.allclose(distances, expected)


def _cell(center):
    return Epicell(
        id=1,
        source_label=None,
        boundary=np.zeros((4, 3)),
        center=np.asarray(center, dtype=float),
        junction_ids=np.array([], dtype=int),
        n_neighbors=0,
        area=1.0,
        perimeter=1.0,
        euler_angles=np.zeros(3),
        curvatures=np.zeros(4),
        ellipse_fit=np.zeros(6),
        eccentricity=0.0,
        proj_direction=0.0,
        uncorrected_area=1.0,
        uncorrected_perimeter=1.0,
        area_error=0.0,
        perimeter_error=0.0,
    )


@dataclass
class FakeResult:
    epicells: list[Epicell]
    pixel_size: float = 0.5
    voxel_depth: float = 2.0
    units: str = "µm"
    prepared_heightmap: np.ndarray | None = None


def test_cell_centers_xy_pixels_divides_by_pixel_size():
    result = FakeResult(epicells=[_cell([1.0, 2.0, 0.0]), _cell([3.0, 4.0, 0.0])])
    centers = cell_centers_xy_pixels(result)
    assert np.allclose(centers, [[2.0, 4.0], [6.0, 8.0]])


def test_surface_distance_calculator_from_result_infers_metadata():
    result = FakeResult(
        epicells=[],
        pixel_size=0.25,
        voxel_depth=3.0,
        units="µm",
        prepared_heightmap=np.ones((4, 4)),
    )
    calc = SurfaceDistanceCalculator.from_result(result)
    assert calc.pixel_size == 0.25
    assert calc.voxel_depth == 1.0
    assert calc.units == "µm"
    assert calc.prepared
