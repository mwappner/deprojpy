from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deprojpy.models import Epicell
from deprojpy.surface_distance import (
    SurfaceDistanceCalculator,
    SurfaceGraph,
    cell_centers_xy_pixels,
    choose_surface_graph_step,
    heightmap_from_cell_boundaries,
    sample_height_at_xy,
    surface_straight_distance,
)
from deprojpy.surface_distance._boundary_surface import _average_duplicate_xy
from deprojpy.surface_distance._helpers import _as_pixel_xy
from deprojpy.surface_distance._straight import _bilinear_sample


def test_as_pixel_xy_converts_physical_coordinates_and_rejects_invalid_units():
    xy = np.array([[2.0, 4.0], [6.0, 8.0]])
    assert np.allclose(_as_pixel_xy(xy, pixel_size=2.0, input_units="pixel"), xy)
    assert np.allclose(_as_pixel_xy(xy, pixel_size=2.0, input_units="physical"), xy / 2.0)
    with np.testing.assert_raises_regex(ValueError, "input_units must be 'pixel' or 'physical'"):
        _as_pixel_xy(xy, pixel_size=2.0, input_units="micron")  # type: ignore[arg-type]


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


def test_bilinear_sampler_matches_map_coordinates():
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


def test_straight_distance_accepts_pixel_and_physical_inputs():
    y, x = np.mgrid[:20, :20]
    calc = SurfaceDistanceCalculator((0.1 * x + 0.2 * y).astype(float), pixel_size=0.5)
    p0_px = np.array([2.0, 3.0])
    p1_px = np.array([12.0, 9.0])
    p0_phys = p0_px * calc.pixel_size
    p1_phys = p1_px * calc.pixel_size

    pixel_distance = calc.straight_distance(p0_px, p1_px, input_units="pixel", n_samples=48)
    physical_distance = calc.straight_distance(
        p0_phys,
        p1_phys,
        input_units="physical",
        n_samples=48,
    )

    assert np.isclose(pixel_distance, physical_distance)


def test_pairwise_distance_accepts_pixel_and_physical_inputs():
    y, x = np.mgrid[:20, :20]
    calc = SurfaceDistanceCalculator((0.1 * x + 0.2 * y).astype(float), pixel_size=0.25)
    points_px = np.array([[1, 1], [4, 1], [4, 5], [8, 6]], dtype=float)
    points_phys = points_px * calc.pixel_size

    pixel_matrix = calc.straight_pairwise_distances(points_px, input_units="pixel", n_samples=32)
    physical_matrix = calc.straight_pairwise_distances(
        points_phys,
        input_units="physical",
        n_samples=32,
    )

    assert np.allclose(pixel_matrix, physical_matrix)


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


def test_selected_pair_distances_accept_pixel_and_physical_inputs():
    heightmap = np.zeros((10, 10), dtype=float)
    calc = SurfaceDistanceCalculator(heightmap, pixel_size=0.5)
    points_px = np.array([[0, 0], [3, 4], [7, 7]], dtype=float)
    points_phys = points_px * calc.pixel_size
    pairs = np.array([[0, 1], [1, 2]])

    pixel_distances = calc.straight_distances_for_pairs(
        points_px,
        pairs,
        input_units="pixel",
        n_samples=16,
    )
    physical_distances = calc.straight_distances_for_pairs(
        points_phys,
        pairs,
        input_units="physical",
        n_samples=16,
    )

    assert np.allclose(pixel_distances, physical_distances)


def test_surface_graph_flat_8_connectivity_is_exact_for_diagonal():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((6, 6), dtype=float),
        pixel_size=1.0,
        connectivity="8",
        step=1,
    )
    distance = graph.distance(np.array([0, 0]), np.array([5, 5]))
    assert np.isclose(distance, 5 * np.sqrt(2))


def test_surface_graph_distance_accepts_pixel_and_physical_inputs():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((8, 8), dtype=float),
        pixel_size=0.5,
        connectivity="8",
        step=1,
    )
    p0_px = np.array([0.0, 0.0])
    p1_px = np.array([5.0, 5.0])
    p0_phys = p0_px * graph.pixel_size
    p1_phys = p1_px * graph.pixel_size

    pixel_distance = graph.distance(p0_px, p1_px, input_units="pixel")
    physical_distance = graph.distance(p0_phys, p1_phys, input_units="physical")

    assert np.isclose(pixel_distance, physical_distance)


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


def test_surface_graph_path_remains_pixel_units_for_physical_input():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((8, 8), dtype=float),
        pixel_size=0.5,
        connectivity="8",
        step=1,
    )
    p0_px = np.array([0.0, 0.0])
    p1_px = np.array([5.0, 5.0])
    p0_phys = p0_px * graph.pixel_size
    p1_phys = p1_px * graph.pixel_size

    distance, path = graph.distance(
        p0_phys,
        p1_phys,
        input_units="physical",
        return_path=True,
    )

    assert np.isfinite(distance)
    assert np.allclose(path[0], p0_px)
    assert np.allclose(path[-1], p1_px)
    assert not np.allclose(path[-1], p1_phys)


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


def test_surface_graph_distances_from_source_accepts_pixel_and_physical_inputs():
    graph = SurfaceGraph.from_heightmap(
        np.zeros((8, 8), dtype=float),
        pixel_size=0.5,
        connectivity="8",
        step=1,
    )
    source_px = np.array([0.0, 0.0])
    targets_px = np.array([[3.0, 0.0], [3.0, 3.0], [5.0, 2.0]])
    source_phys = source_px * graph.pixel_size
    targets_phys = targets_px * graph.pixel_size

    pixel_distances = graph.distances_from_source(source_px, targets_px, input_units="pixel")
    physical_distances = graph.distances_from_source(
        source_phys,
        targets_phys,
        input_units="physical",
    )

    assert np.allclose(pixel_distances, physical_distances)


def _cell(center, boundary=None):
    return Epicell(
        id=1,
        source_label=None,
        boundary=np.zeros((4, 3)) if boundary is None else np.asarray(boundary, dtype=float),
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


def _plane_z(x, y):
    return 2.0 * x + 3.0 * y + 5.0


def _plane_boundary(points_xy):
    points = np.asarray(points_xy, dtype=float)
    z = _plane_z(points[:, 0], points[:, 1])
    return np.column_stack([points, z])


def _plane_result(*, shape=(20, 30), pixel_size=1.0, units="µm"):
    outer = _plane_boundary(
        [
            [2, 2],
            [17, 2],
            [17, 13],
            [2, 13],
        ]
    )
    inner = _plane_boundary(
        [
            [8, 8],
            [24, 8],
            [24, 17],
            [8, 17],
        ]
    )
    return FakeResult(
        epicells=[_cell([0, 0, 0], outer), _cell([0, 0, 0], inner)],
        pixel_size=pixel_size,
        units=units,
        prepared_heightmap=np.zeros(shape, dtype=float),
    )


def test_heightmap_from_cell_boundaries_uses_prepared_heightmap_shape():
    result = _plane_result(shape=(20, 30))
    surface = heightmap_from_cell_boundaries(result)
    assert surface.shape == result.prepared_heightmap.shape
    assert np.isfinite(surface).all()


def test_heightmap_from_cell_boundaries_recovers_planar_surface():
    result = _plane_result(shape=(10, 12))
    surface = heightmap_from_cell_boundaries(
        result,
        method="linear",
        extrapolation="linear",
    )
    for y, x in [(0, 0), (3, 4), (8, 10)]:
        assert np.isclose(surface[y, x], _plane_z(x, y), atol=1e-10)


def test_surface_distance_calculator_from_cell_boundaries_metadata():
    result = _plane_result(pixel_size=0.5, units="µm")
    calc = SurfaceDistanceCalculator.from_cell_boundaries(result)
    assert calc.result is result
    assert calc.pixel_size == result.pixel_size
    assert calc.voxel_depth == 1.0
    assert calc.units == result.units
    assert calc.prepared is True


def test_surface_distance_calculator_samples_boundary_surface():
    result = _plane_result(shape=(12, 12))
    calc = SurfaceDistanceCalculator.from_cell_boundaries(result)
    xy = np.array([[4.0, 5.0], [10.0, 9.0]])
    assert np.allclose(calc.sample_height(xy), _plane_z(xy[:, 0], xy[:, 1]))


def test_average_duplicate_xy_averages_z_values():
    xy = np.array([[0, 0], [1, 1], [0, 0], [2, 0]], dtype=float)
    z = np.array([1.0, 5.0, 3.0, 7.0])
    unique_xy, averaged_z = _average_duplicate_xy(xy, z)
    values = {tuple(point): value for point, value in zip(unique_xy, averaged_z, strict=True)}
    assert values[(0.0, 0.0)] == 2.0
    assert values[(1.0, 1.0)] == 5.0
    assert values[(2.0, 0.0)] == 7.0


def test_heightmap_from_cell_boundaries_extrapolation_modes():
    boundary = _plane_boundary([[2, 2], [5, 2], [5, 5], [2, 5]])
    result = FakeResult(
        epicells=[_cell([0, 0, 0], boundary)],
        pixel_size=1.0,
        units="µm",
        prepared_heightmap=np.zeros((8, 8)),
    )

    none = heightmap_from_cell_boundaries(result, extrapolation="none")
    nearest = heightmap_from_cell_boundaries(result, extrapolation="nearest")
    constant = heightmap_from_cell_boundaries(
        result,
        extrapolation="constant",
        fill_value=-1.0,
    )
    linear = heightmap_from_cell_boundaries(result, extrapolation="linear")

    assert np.isnan(none[0, 0])
    assert np.isfinite(nearest).all()
    assert constant[0, 0] == -1.0
    assert np.isclose(linear[0, 0], _plane_z(0, 0), atol=1e-10)


def test_heightmap_from_cell_boundaries_rejects_invalid_options():
    result = _plane_result()
    with np.testing.assert_raises(ValueError):
        heightmap_from_cell_boundaries(result, method="cubic")
    with np.testing.assert_raises(ValueError):
        heightmap_from_cell_boundaries(result, extrapolation="mystery")
