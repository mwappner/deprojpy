from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.ndimage import map_coordinates
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

from .heightmap import prepare_heightmap


def _validate_xy_array(xy: np.ndarray, *, name: str = "xy") -> np.ndarray:
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2) in (x, y) order")
    return points


def _validate_xy_point(point_xy: np.ndarray, *, name: str = "point_xy") -> np.ndarray:
    point = np.asarray(point_xy, dtype=float)
    if point.shape != (2,):
        raise ValueError(f"{name} must have shape (2,) in (x, y) order")
    return point


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def sample_height_at_xy(
    heightmap: np.ndarray,
    xy: np.ndarray,
    *,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Sample a height map at ``(x, y)`` coordinates.

    Parameters
    ----------
    heightmap : np.ndarray
        2-D height map indexed as ``heightmap[row=y, column=x]``.
    xy : np.ndarray
        Point coordinates with shape ``(n, 2)`` in geometric ``(x, y)`` order.
    order : int, optional
        Interpolation order passed to :func:`scipy.ndimage.map_coordinates`.
        Default is bilinear interpolation, ``order=1``.
    mode : str, optional
        Boundary mode passed to :func:`scipy.ndimage.map_coordinates`. Default
        is ``"nearest"``.

    Returns
    -------
    np.ndarray
        Interpolated height values, one per input point.
    """
    h = np.asarray(heightmap, dtype=float)
    if h.ndim != 2:
        raise ValueError("heightmap must be a 2-D array")
    points = _validate_xy_array(xy)
    coords_yx = np.vstack([points[:, 1], points[:, 0]])
    return np.asarray(map_coordinates(h, coords_yx, order=order, mode=mode), dtype=float)


@njit(cache=True)
def _bilinear_sample(heightmap, x, y):  # pragma: no cover - exercised through wrappers
    h, w = heightmap.shape

    # Clamp before interpolation to match map_coordinates(..., mode="nearest")
    # for order-1 sampling in the common use cases here.
    if x < 0.0:
        x = 0.0
    elif x > w - 1:
        x = w - 1.0
    if y < 0.0:
        y = 0.0
    elif y > h - 1:
        y = h - 1.0

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = x - x0
    ty = y - y0

    z00 = float(heightmap[y0, x0])
    z10 = float(heightmap[y0, x1])
    z01 = float(heightmap[y1, x0])
    z11 = float(heightmap[y1, x1])
    z0 = z00 * (1.0 - tx) + z10 * tx
    z1 = z01 * (1.0 - tx) + z11 * tx
    return z0 * (1.0 - ty) + z1 * ty


@njit(cache=True)
def _surface_straight_distance_numba(
    heightmap,
    x0,
    y0,
    x1,
    y1,
    pixel_size,
    voxel_depth,
    n_samples,
):  # pragma: no cover
    dx = x1 - x0
    dy = y1 - y0
    if n_samples < 2:
        n_samples = 2

    prev_x = x0
    prev_y = y0
    prev_z = _bilinear_sample(heightmap, prev_x, prev_y) * voxel_depth
    total = 0.0

    for sample in range(1, n_samples):
        t = sample / (n_samples - 1)
        x = x0 + t * dx
        y = y0 + t * dy
        z = _bilinear_sample(heightmap, x, y) * voxel_depth
        ddx = (x - prev_x) * pixel_size
        ddy = (y - prev_y) * pixel_size
        ddz = z - prev_z
        total += math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
        prev_x = x
        prev_y = y
        prev_z = z
    return total


@njit(cache=True, parallel=True)
def _pairwise_surface_straight_distances_numba(
    heightmap,
    points_xy,
    pixel_size,
    voxel_depth,
    samples_per_pixel,
    min_samples,
    max_samples,
):  # pragma: no cover
    n_points = points_xy.shape[0]
    out = np.zeros((n_points, n_points), dtype=np.float64)
    for i in prange(n_points):
        xi = points_xy[i, 0]
        yi = points_xy[i, 1]
        for j in range(i + 1, n_points):
            dx = points_xy[j, 0] - xi
            dy = points_xy[j, 1] - yi
            xy_dist_px = math.sqrt(dx * dx + dy * dy)
            n_samples = int(math.ceil(xy_dist_px * samples_per_pixel)) + 1
            if n_samples < min_samples:
                n_samples = min_samples
            elif n_samples > max_samples:
                n_samples = max_samples
            distance = _surface_straight_distance_numba(
                heightmap,
                xi,
                yi,
                points_xy[j, 0],
                points_xy[j, 1],
                pixel_size,
                voxel_depth,
                n_samples,
            )
            out[i, j] = distance
            out[j, i] = distance
    return out


@njit(cache=True, parallel=True)
def _selected_surface_straight_distances_numba(
    heightmap,
    points_xy,
    pairs,
    pixel_size,
    voxel_depth,
    samples_per_pixel,
    min_samples,
    max_samples,
):  # pragma: no cover
    n_pairs = pairs.shape[0]
    out = np.empty(n_pairs, dtype=np.float64)
    for k in prange(n_pairs):
        i = pairs[k, 0]
        j = pairs[k, 1]
        x0 = points_xy[i, 0]
        y0 = points_xy[i, 1]
        x1 = points_xy[j, 0]
        y1 = points_xy[j, 1]
        dx = x1 - x0
        dy = y1 - y0
        xy_dist_px = math.sqrt(dx * dx + dy * dy)
        n_samples = int(math.ceil(xy_dist_px * samples_per_pixel)) + 1
        if n_samples < min_samples:
            n_samples = min_samples
        elif n_samples > max_samples:
            n_samples = max_samples
        out[k] = _surface_straight_distance_numba(
            heightmap,
            x0,
            y0,
            x1,
            y1,
            pixel_size,
            voxel_depth,
            n_samples,
        )
    return out


def _auto_n_samples(
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    *,
    samples_per_pixel: float,
    min_samples: int,
    max_samples: int,
) -> int:
    xy_dist_px = float(np.linalg.norm(p1_xy - p0_xy))
    n_samples = int(np.ceil(xy_dist_px * samples_per_pixel)) + 1
    return int(np.clip(n_samples, int(min_samples), int(max_samples)))


def surface_straight_distance(
    heightmap: np.ndarray,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    *,
    pixel_size: float = 1.0,
    voxel_depth: float = 1.0,
    n_samples: int | None = None,
    samples_per_pixel: float = 2.0,
    min_samples: int = 16,
    max_samples: int = 2048,
    order: int = 1,
) -> float:
    """
    Approximate surface distance along the straight segment in ``xy``.

    Parameters
    ----------
    heightmap : np.ndarray
        2-D height map. Values are multiplied by ``voxel_depth`` before distance
        accumulation.
    p0_xy, p1_xy : np.ndarray
        Endpoints in ``(x, y)`` pixel coordinates.
    pixel_size : float, optional
        Physical size of one pixel in the ``x`` and ``y`` directions.
    voxel_depth : float, optional
        Physical scale factor for height-map values.
    n_samples : int, optional
        Number of points sampled along the segment. If omitted, it is chosen
        from the 2-D pixel distance.
    samples_per_pixel : float, optional
        Sampling density used when ``n_samples`` is omitted.
    min_samples, max_samples : int, optional
        Bounds for automatically chosen sample counts.
    order : int, optional
        Interpolation order. The optimized path supports ``order=1``.

    Returns
    -------
    float
        3-D polyline length in physical units.
    """
    h = np.asarray(heightmap, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("heightmap must be a 2-D array")
    p0 = _validate_xy_point(p0_xy, name="p0_xy")
    p1 = _validate_xy_point(p1_xy, name="p1_xy")
    pixel_size = _validate_positive(pixel_size, "pixel_size")
    voxel_depth = _validate_positive(voxel_depth, "voxel_depth")
    if n_samples is None:
        n_samples = _auto_n_samples(
            p0,
            p1,
            samples_per_pixel=samples_per_pixel,
            min_samples=min_samples,
            max_samples=max_samples,
        )
    if int(n_samples) < 2:
        raise ValueError("n_samples must be at least 2")

    if order == 1:
        return float(
            _surface_straight_distance_numba(
                h,
                float(p0[0]),
                float(p0[1]),
                float(p1[0]),
                float(p1[1]),
                pixel_size,
                voxel_depth,
                int(n_samples),
            )
        )

    # Keep uncommon interpolation orders available through SciPy for one-off use.
    xy = np.column_stack(
        [np.linspace(p0[0], p1[0], int(n_samples)), np.linspace(p0[1], p1[1], int(n_samples))]
    )
    z = sample_height_at_xy(h, xy, order=order, mode="nearest") * voxel_depth
    xyz = np.column_stack([xy[:, 0] * pixel_size, xy[:, 1] * pixel_size, z])
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def cell_centers_xy_pixels(result) -> np.ndarray:
    """
    Return deprojected cell centers in ``(x, y)`` pixel coordinates.

    Parameters
    ----------
    result
        DeProjPy result-like object with ``epicells`` and ``pixel_size``.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_cells, 2)`` in ``(x, y)`` pixel coordinates.
    """
    pixel_size = _validate_positive(getattr(result, "pixel_size"), "result.pixel_size")
    centers = np.asarray([cell.center[:2] for cell in result.epicells], dtype=float)
    return centers.reshape(-1, 2) / pixel_size


def _average_duplicate_xy(points_xy: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average z-values for exact duplicate xy coordinates."""
    points = _validate_xy_array(points_xy, name="points_xy")
    values = np.asarray(z, dtype=float)
    if values.ndim != 1 or len(values) != len(points):
        raise ValueError("z must be a 1-D array with one value per xy point")
    unique_xy, inverse = np.unique(points, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique_xy, sums / counts


def _boundary_interpolator(method: str, points_xy: np.ndarray, z: np.ndarray):
    if method == "linear":
        return LinearNDInterpolator(points_xy, z, fill_value=np.nan)
    if method in {"clough", "clough_tocher"}:
        return CloughTocher2DInterpolator(points_xy, z, fill_value=np.nan)
    if method == "nearest":
        return NearestNDInterpolator(points_xy, z)
    raise ValueError("method must be one of 'linear', 'clough', 'clough_tocher', or 'nearest'")


def _linear_plane_extrapolate(
    points_xy: np.ndarray,
    z: np.ndarray,
    grid_xy: np.ndarray,
) -> np.ndarray:
    """Evaluate the least-squares plane fitted to scattered xyz samples."""
    design = np.column_stack([points_xy[:, 0], points_xy[:, 1], np.ones(len(points_xy))])
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    grid_design = np.column_stack([grid_xy[:, 0], grid_xy[:, 1], np.ones(len(grid_xy))])
    return grid_design @ coeffs


def heightmap_from_cell_boundaries(
    result,
    *,
    shape: tuple[int, ...] | None = None,
    method: str = "linear",
    extrapolation: str = "linear",
    fill_value: float | None = None,
) -> np.ndarray:
    """
    Interpolate a raster height map from deprojected cell-boundary points.

    Parameters
    ----------
    result
        DeProjPy result-like object with ``epicells`` and ``pixel_size``.
        Each cell must provide ``boundary`` coordinates in physical
        ``(x, y, z)`` order.
    shape : tuple[int, int], optional
        Output shape in array ``(rows, columns)`` order. If omitted,
        ``result.prepared_heightmap.shape`` is used when available. Otherwise a
        minimal shape is inferred from the maximum boundary pixel coordinates.
    method : {"linear", "clough", "clough_tocher", "nearest"}, optional
        Scattered interpolation method used inside the convex hull of boundary
        points. ``"linear"`` is the default.
    extrapolation : {"linear", "nearest", "constant", "none"}, optional
        How to fill grid points not covered by the scattered interpolator.
        ``"linear"`` fits a least-squares plane to all boundary points,
        ``"nearest"`` uses nearest-neighbor interpolation, ``"constant"`` uses
        ``fill_value``, and ``"none"`` leaves missing values as ``NaN``.
    fill_value : float, optional
        Constant value used when ``extrapolation="constant"``. If omitted,
        missing values are filled with ``NaN``.

    Returns
    -------
    np.ndarray
        Interpolated height map in physical z units.
    """
    if extrapolation not in {"linear", "nearest", "constant", "none"}:
        raise ValueError("extrapolation must be one of 'linear', 'nearest', 'constant', or 'none'")
    pixel_size = _validate_positive(getattr(result, "pixel_size"), "result.pixel_size")
    boundaries = []
    for cell in getattr(result, "epicells", []):
        boundary = np.asarray(cell.boundary, dtype=float)
        if boundary.ndim != 2 or boundary.shape[1] < 3:
            raise ValueError("each cell boundary must have shape (n, >=3)")
        finite = np.all(np.isfinite(boundary[:, :3]), axis=1)
        if np.any(finite):
            boundaries.append(boundary[finite, :3])
    if not boundaries:
        raise ValueError("result contains no finite cell-boundary xyz points")

    points_xyz = np.vstack(boundaries)
    points_xy_px = points_xyz[:, :2] / pixel_size
    points_z = points_xyz[:, 2]

    # Adjacent cells can contribute the same xy boundary point with slightly
    # different z-values. Average exact duplicates before scattered interpolation.
    points_xy_px, points_z = _average_duplicate_xy(points_xy_px, points_z)

    if shape is None:
        prepared = getattr(result, "prepared_heightmap", None)
        if prepared is not None:
            shape = np.asarray(prepared).shape
        else:
            max_x = float(np.nanmax(points_xy_px[:, 0]))
            max_y = float(np.nanmax(points_xy_px[:, 1]))
            shape = (int(np.ceil(max_y)) + 1, int(np.ceil(max_x)) + 1)
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError("shape must be a positive (rows, columns) tuple")
    height, width = int(shape[0]), int(shape[1])

    yy, xx = np.mgrid[:height, :width]
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

    interpolator = _boundary_interpolator(method, points_xy_px, points_z)
    surface = np.asarray(interpolator(grid_xy), dtype=float).reshape(height, width)
    missing = ~np.isfinite(surface)
    if not np.any(missing):
        return surface

    if extrapolation == "none":
        return surface
    if extrapolation == "nearest":
        nearest = NearestNDInterpolator(points_xy_px, points_z)
        surface[missing] = nearest(grid_xy[missing.ravel()])
        return surface
    if extrapolation == "constant":
        surface[missing] = np.nan if fill_value is None else float(fill_value)
        return surface
    if extrapolation == "linear":
        plane_values = _linear_plane_extrapolate(points_xy_px, points_z, grid_xy)
        surface[missing] = plane_values.reshape(height, width)[missing]
        return surface
    raise AssertionError("unreachable extrapolation branch")


@dataclass
class SurfaceDistanceCalculator:
    """
    Reusable calculator for distances on a height-map surface.

    This class stores the height map and physical scale factors needed to
    evaluate distances on the height-field surface
    ``r(x, y) = (x, y, h(x, y))``. It is intended to be constructed once and
    reused for many distance queries, especially all-pairs distances between
    cell centers.

    Parameters
    ----------
    heightmap : np.ndarray
        Two-dimensional height map. Arrays are indexed in image order as
        ``heightmap[row=y, column=x]``. Public point inputs to calculator
        methods are always geometric ``(x, y)`` pixel coordinates.
    pixel_size : float
        Physical size of one pixel in the ``x`` and ``y`` directions. If
        ``units="µm"``, for example, distances returned by the calculator are
        in micrometers.
    voxel_depth : float, optional
        Physical scale factor applied to height-map values. Use ``1.0`` when
        ``heightmap`` is already prepared in physical z units. When using
        :meth:`from_heightmap` or :meth:`from_result` with ``prepared=False``,
        the returned calculator stores a prepared physical-height map and sets
        ``voxel_depth`` to ``1.0`` internally.
    units : str, optional
        Text label for the physical distance units. This does not change
        calculations; it is stored for display and examples.
    prepared : bool, optional
        Whether the stored ``heightmap`` is already the array used directly for
        distance calculations. Direct construction does not call
        :func:`deprojpy.heightmap.prepare_heightmap`; use
        :meth:`from_heightmap` or :meth:`from_result` to prepare raw maps.
    result : object, optional
        Optional DeProjPy result-like object. When present, methods such as
        :meth:`straight_pairwise_distances` can use its cell centers if explicit
        ``points_xy`` are not supplied.

    Coordinate and unit conventions
    -------------------------------
    All point arguments accepted by public methods are ``(x, y)`` pixel
    coordinates, not physical coordinates and not array ``(row, column)``
    indices. Cell centers stored in :class:`deprojpy.models.Epicell` are already
    physical ``(x, y, z)`` coordinates; use :func:`cell_centers_xy_pixels` or
    :meth:`from_result` to convert them consistently.

    Methods return physical distances. In the ``x`` and ``y`` directions,
    pixel-coordinate differences are multiplied by ``pixel_size``. In the
    ``z`` direction, sampled height-map differences are multiplied by
    ``voxel_depth`` unless the map was already prepared into physical units.

    Main methods
    ------------
    ``sample_height(xy)``
        Bilinearly sample the calculator's height map at ``(x, y)`` pixel
        coordinates and return physical z values.
    ``straight_distance(p0_xy, p1_xy, ...)``
        Follow the straight segment between two ``xy`` points, lift it onto the
        surface, and return the 3-D polyline length. This is useful and fast,
        but it is not a shortest-path geodesic.
    ``straight_pairwise_distances(points_xy=None, result=None, ...)``
        Return a symmetric ``N × N`` matrix of straight-line surface distances.
        If ``points_xy`` is omitted, cell centers are taken from the stored
        result or from the ``result`` argument.
    ``straight_distances_for_pairs(points_xy, pairs, ...)``
        Compute distances only for selected index pairs, where ``pairs`` has
        shape ``(n_pairs, 2)`` and indexes rows of ``points_xy``.

    Construction helpers
    --------------------
    Use :meth:`from_result` when a DeProjPy result is available; it infers
    ``pixel_size``, ``voxel_depth``, and ``units`` from the result while still
    building the surface from a height map. Use :meth:`from_cell_boundaries`
    when distances should follow an interpolated surface reconstructed from the
    deprojected cell-boundary xyz points. Use :meth:`from_heightmap` when
    working from an external height map or exported cell-center coordinates.
    """

    heightmap: np.ndarray
    pixel_size: float
    voxel_depth: float = 1.0
    units: str = "a.u."
    prepared: bool = True
    result: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        h = np.asarray(self.heightmap, dtype=np.float64)
        if h.ndim != 2:
            raise ValueError("heightmap must be a 2-D array")
        self.heightmap = h
        self.pixel_size = _validate_positive(self.pixel_size, "pixel_size")
        self.voxel_depth = _validate_positive(self.voxel_depth, "voxel_depth")

    @classmethod
    def from_heightmap(
        cls,
        heightmap,
        *,
        pixel_size,
        voxel_depth=1.0,
        prepared: bool = True,
        units: str = "a.u.",
        smooth_scale: float = -1.0,
        invert_z: bool = False,
        inpaint_zeros: bool = True,
        prune_zeros: bool = True,
    ) -> "SurfaceDistanceCalculator":
        """
        Build a calculator from a raw or already prepared height map.

        Parameters
        ----------
        heightmap : np.ndarray
            Two-dimensional height map indexed as ``heightmap[row=y, column=x]``.
            If ``prepared=True``, values are used directly. If
            ``prepared=False``, the map is passed through
            :func:`deprojpy.heightmap.prepare_heightmap`.
        pixel_size : float
            Physical size of one pixel in the ``x`` and ``y`` directions.
            Public point inputs remain ``(x, y)`` pixel coordinates; returned
            distances are physical lengths.
        voxel_depth : float, optional
            Physical scale factor for raw height-map values. When
            ``prepared=True``, this factor is applied during distance
            calculations. When ``prepared=False``, it is consumed by
            ``prepare_heightmap`` and the returned calculator stores
            ``voxel_depth=1.0`` because its height map is already in physical
            z units.
        prepared : bool, optional
            If ``True`` (default), assume ``heightmap`` is already the surface
            used for distance calculations. If ``False``, run DeProjPy height-map
            preprocessing first.
        units : str, optional
            Label for the physical distance units.
        smooth_scale : float, optional
            Gaussian smoothing scale passed to ``prepare_heightmap`` when
            ``prepared=False``. Values ``<= 0`` disable smoothing.
        invert_z : bool, optional
            Whether to invert height values during preparation.
        inpaint_zeros : bool, optional
            Whether zero-valued regions are inpainted during preparation.
        prune_zeros : bool, optional
            Whether remaining zero-valued pixels become ``NaN`` during
            preparation.

        Returns
        -------
        SurfaceDistanceCalculator
            Calculator whose stored ``heightmap`` is the actual surface used for
            distance calculations.
        """
        pixel_size = _validate_positive(pixel_size, "pixel_size")
        voxel_depth = _validate_positive(voxel_depth, "voxel_depth")
        if prepared:
            return cls(
                heightmap=np.asarray(heightmap, dtype=float),
                pixel_size=pixel_size,
                voxel_depth=voxel_depth,
                units=units,
                prepared=True,
            )
        h = prepare_heightmap(
            heightmap,
            voxel_depth=voxel_depth,
            smooth_scale=smooth_scale,
            invert_z=invert_z,
            inpaint_zeros=inpaint_zeros,
            prune_zeros=prune_zeros,
        )
        return cls(h, pixel_size=pixel_size, voxel_depth=1.0, units=units, prepared=True)

    @classmethod
    def from_result(
        cls,
        result,
        heightmap=None,
        *,
        prepared: bool = False,
        pixel_size: float | None = None,
        voxel_depth: float | None = None,
        units: str | None = None,
        invert_z: bool | None = None,
        smooth_scale: float = -1.0,
        inpaint_zeros: bool = True,
        prune_zeros: bool = True,
    ) -> "SurfaceDistanceCalculator":
        """
        Build a height-map-based calculator using metadata from a result.

        This constructor still builds the surface from a height map. It does
        not use deprojected cell-boundary points; use
        :meth:`from_cell_boundaries` for that surface.

        Parameters
        ----------
        result
            DeProjPy result-like object. ``pixel_size``, ``voxel_depth``, and
            ``units`` are inferred from this object unless explicitly supplied.
            The result is also stored so pairwise methods can use cell centers
            when ``points_xy`` is omitted.
        heightmap : np.ndarray, optional
            Raw or prepared height map. If omitted, ``result.prepared_heightmap``
            is used and treated as prepared.
        prepared : bool, optional
            If ``True``, use ``heightmap`` directly. If ``False`` (default),
            prepare the provided raw height map with DeProjPy preprocessing.
        pixel_size : float, optional
            Override ``result.pixel_size``.
        voxel_depth : float, optional
            Override ``result.voxel_depth``. Ignored after construction when
            ``prepared=False`` because the stored height map is converted to
            physical z units.
        units : str, optional
            Override ``result.units``.
        invert_z : bool, optional
            Whether to invert height values during preparation. If omitted,
            defaults to ``False`` because current result objects do not record
            the original inversion setting.
        smooth_scale, inpaint_zeros, prune_zeros
            Passed to ``prepare_heightmap`` when ``prepared=False``.

        Returns
        -------
        SurfaceDistanceCalculator
            Calculator with scale metadata inferred from ``result`` and a stored
            height map ready for repeated distance calls.
        """
        if pixel_size is None:
            pixel_size = getattr(result, "pixel_size")
        if voxel_depth is None:
            voxel_depth = getattr(result, "voxel_depth", 1.0)
        if units is None:
            units = getattr(result, "units", "a.u.")
        if heightmap is None:
            prepared_heightmap = getattr(result, "prepared_heightmap", None)
            if prepared_heightmap is None:
                raise ValueError("heightmap is required when result has no prepared_heightmap")
            heightmap = prepared_heightmap
            prepared = True
        if invert_z is None:
            invert_z = False

        calc = cls.from_heightmap(
            heightmap,
            pixel_size=pixel_size,
            voxel_depth=1.0 if prepared else voxel_depth, # type: ignore
            prepared=prepared,
            units=units, # type: ignore
            smooth_scale=smooth_scale,
            invert_z=invert_z,
            inpaint_zeros=inpaint_zeros,
            prune_zeros=prune_zeros,
        )
        calc.result = result
        return calc

    @classmethod
    def from_cell_boundaries(
        cls,
        result,
        *,
        shape: tuple[int, int] | None = None,
        method: str = "linear",
        extrapolation: str = "nearest",
        fill_value: float | None = None,
        units: str | None = None,
    ) -> "SurfaceDistanceCalculator":
        """
        Build a calculator from deprojected cell-boundary xyz points.

        This constructor interpolates the physical z-values stored in
        ``cell.boundary`` arrays back onto an image-like raster. It is useful
        when distance calculations or path visualizations should follow the
        fitted DeProj boundary surface rather than the original height map.

        Parameters
        ----------
        result
            DeProjPy result-like object with ``epicells`` and ``pixel_size``.
            Each cell boundary is expected to have physical ``(x, y, z)``
            columns. ``pixel_size`` converts boundary ``x/y`` coordinates back
            to pixel coordinates before interpolation.
        shape : tuple[int, int], optional
            Output raster shape in array ``(rows, columns)`` order. If omitted,
            ``result.prepared_heightmap.shape`` is preferred. If that is not
            available, a minimal shape is inferred from boundary coordinates.
        method : {"linear", "clough", "clough_tocher", "nearest"}, optional
            Scattered interpolation method used inside the convex hull of
            boundary points.
        extrapolation : {"linear", "nearest", "constant", "none"}, optional
            How to fill grid points outside the interpolator support. ``"linear"``
            fits a least-squares plane to boundary points, ``"nearest"`` uses
            nearest-neighbor values, ``"constant"`` uses ``fill_value``, and
            ``"none"`` leaves missing values as ``NaN``.
        fill_value : float, optional
            Fill value used when ``extrapolation="constant"``.
        units : str, optional
            Override ``result.units`` for display. Calculations are unaffected.

        Returns
        -------
        SurfaceDistanceCalculator
            Calculator with ``pixel_size`` inferred from ``result``,
            ``voxel_depth=1.0``, and a prepared raster surface in physical
            z units.
        """
        pixel_size = _validate_positive(getattr(result, "pixel_size"), "result.pixel_size")
        if units is None:
            units = getattr(result, "units", "a.u.")
        surface = heightmap_from_cell_boundaries(
            result,
            shape=shape,
            method=method,
            extrapolation=extrapolation,
            fill_value=fill_value,
        )
        return cls(
            heightmap=surface,
            pixel_size=pixel_size,
            voxel_depth=1.0,
            units=units, # type: ignore (units is validated above)
            prepared=True,
            result=result,
        )

    def sample_height(self, xy: np.ndarray) -> np.ndarray:
        """Sample this calculator's height map at ``(x, y)`` pixel coordinates."""
        return sample_height_at_xy(self.heightmap, xy, order=1, mode="nearest") * self.voxel_depth

    def straight_distance(self, p0_xy, p1_xy, **kwargs) -> float:
        """Return straight-line surface distance between two ``(x, y)`` points."""
        return surface_straight_distance(
            self.heightmap,
            p0_xy,
            p1_xy,
            pixel_size=self.pixel_size,
            voxel_depth=self.voxel_depth,
            **kwargs,
        )

    def _resolve_points(self, points_xy: np.ndarray | None, result=None) -> np.ndarray:
        if points_xy is not None:
            return _validate_xy_array(points_xy, name="points_xy")
        result = result if result is not None else self.result
        if result is None:
            raise ValueError("points_xy is required unless a result is available")
        return cell_centers_xy_pixels(result)

    def straight_pairwise_distances(
        self,
        points_xy: np.ndarray | None = None,
        *,
        result=None,
        n_samples: int | None = None,
        samples_per_pixel: float = 2.0,
        min_samples: int = 16,
        max_samples: int = 2048,
        block_size: int | None = None,
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """
        Return an ``N × N`` matrix of straight-line surface distances.

        The implementation computes only the upper triangle and mirrors it to
        the lower triangle.
        """
        del block_size, show_progress
        points = self._resolve_points(points_xy, result=result)
        if len(points) > 2000:
            warnings.warn(
                "straight_pairwise_distances computes an N x N matrix and may "
                "take substantial time and memory for N > 2000",
                RuntimeWarning,
                stacklevel=2,
            )
        if n_samples is not None:
            min_samples = max_samples = int(n_samples)
        return _pairwise_surface_straight_distances_numba(
            self.heightmap,
            points.astype(np.float64),
            float(self.pixel_size),
            float(self.voxel_depth),
            float(samples_per_pixel),
            int(min_samples),
            int(max_samples),
        )

    def straight_distances_for_pairs(
        self,
        points_xy: np.ndarray,
        pairs: np.ndarray,
        *,
        n_samples: int | None = None,
        samples_per_pixel: float = 2.0,
        min_samples: int = 16,
        max_samples: int = 2048,
    ) -> np.ndarray:
        """Compute straight-line surface distances for selected point pairs."""
        points = _validate_xy_array(points_xy, name="points_xy")
        pair_array = np.asarray(pairs, dtype=np.int64)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2)")
        if pair_array.size and (pair_array.min() < 0 or pair_array.max() >= len(points)):
            raise IndexError("pairs contain point indices outside points_xy")
        if n_samples is not None:
            min_samples = max_samples = int(n_samples)
        return _selected_surface_straight_distances_numba(
            self.heightmap,
            points.astype(np.float64),
            pair_array,
            float(self.pixel_size),
            float(self.voxel_depth),
            float(samples_per_pixel),
            int(min_samples),
            int(max_samples),
        )


def choose_surface_graph_step(
    shape: tuple[int, ...],
    *,
    target_nodes: int = 80_000,
    min_step: int = 1,
    max_step: int = 16,
) -> int:
    """Choose a grid step that keeps a surface graph near ``target_nodes``."""
    if len(shape) != 2:
        raise ValueError("shape must be a (height, width) tuple")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")
    target_nodes = int(target_nodes)
    if target_nodes <= 0:
        raise ValueError("target_nodes must be positive")
    raw_step = math.sqrt((height * width) / target_nodes)
    step = int(math.ceil(raw_step))
    return int(np.clip(step, int(min_step), int(max_step)))


def _sampled_axis(length: int, step: int) -> np.ndarray:
    values = np.arange(0, length, step, dtype=np.int64)
    if values.size == 0 or values[-1] != length - 1:
        values = np.r_[values, length - 1]
    return values


def _connectivity_offsets(connectivity: str) -> list[tuple[int, int]]:
    offsets = {
        "4": [(1, 0), (0, 1)],
        "8": [(1, 0), (0, 1), (1, 1), (1, -1)],
        "16": [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
            (2, 1),
            (2, -1),
            (1, 2),
            (-1, 2),
        ],
    }
    if connectivity not in offsets:
        raise ValueError("connectivity must be one of '4', '8', or '16'")
    return offsets[connectivity]


@dataclass
class SurfaceGraph:
    """
    Sparse graph approximation to geodesic distances on a height-map surface.

    ``SurfaceGraph`` represents a sampled version of the height-field surface
    ``r(x, y) = (x, y, h(x, y))`` as a sparse weighted graph. Each graph node is
    an ``(x, y)`` pixel location lifted to the surface, and each edge weight is
    the local 3-D distance between two sampled surface points. Dijkstra shortest
    paths on this graph approximate surface geodesic distances.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Sparse adjacency matrix. Entry ``graph[i, j]`` is the physical 3-D
        distance between sampled surface nodes ``i`` and ``j``. The graph is
        treated as undirected by distance methods.
    xy : np.ndarray
        ``(n_nodes, 2)`` array of node coordinates in geometric ``(x, y)`` pixel
        units. These are the coordinates returned in shortest paths.
    z : np.ndarray
        Height values for each node. Values are interpreted together with
        ``voxel_depth`` when edge weights are constructed.
    shape : tuple[int, int]
        Shape of the original height map in array order ``(rows, columns)``.
    step : int
        Grid sampling step in pixels. ``step=1`` uses every pixel. Larger values
        build smaller, faster, more approximate graphs. If constructed with
        ``step="auto"``, the step is selected by :func:`choose_surface_graph_step`.
    pixel_size : float
        Physical size of one pixel in ``x`` and ``y``. Graph distances are
        returned in these physical units.
    voxel_depth : float
        Physical scale factor for height values used during graph construction.
        For prepared physical height maps this is typically ``1.0``.
    connectivity : {"4", "8", "16"}
        Neighborhood stencil used to add graph edges:

        - ``"4"`` connects horizontal and vertical neighbors. It is fastest but
          strongly grid-biased and gives Manhattan-like distances on flat
          surfaces.
        - ``"8"`` also connects diagonal neighbors. It is a good default for
          many uses.
        - ``"16"`` adds longer oblique moves such as ``(2, 1)`` and ``(1, 2)``.
          It reduces directional bias at higher memory and compute cost.

    Coordinate and unit conventions
    -------------------------------
    Public methods accept ``(x, y)`` pixel coordinates. Input points are snapped
    to their nearest graph nodes before Dijkstra searches. Returned distances
    are physical lengths using ``pixel_size`` for ``x/y`` and ``voxel_depth`` for
    height differences as encoded in the graph edge weights. Returned paths are
    arrays of ``(x, y)`` pixel coordinates along graph nodes, not physical
    coordinates.

    Construction helpers
    --------------------
    :meth:`from_calculator` builds a graph from a
    :class:`SurfaceDistanceCalculator`, reusing its prepared height map and
    scale factors. :meth:`from_heightmap` builds directly from a height map when
    scale factors are supplied manually. The lower-level :func:`build_surface_graph`
    function performs the same construction.

    Notes
    -----
    Graph distances are approximate. They depend on ``step`` and
    ``connectivity`` and should not be described as exact continuous geodesics.
    For one source against many targets, prefer :meth:`distances_from_source`;
    it runs one Dijkstra search and indexes the requested targets.
    """

    graph: sparse.csr_matrix
    xy: np.ndarray
    z: np.ndarray
    shape: tuple[int, ...]
    step: int
    pixel_size: float
    voxel_depth: float
    connectivity: str
    _tree: KDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tree = KDTree(np.asarray(self.xy, dtype=float))

    @classmethod
    def from_calculator(
        cls,
        calculator: SurfaceDistanceCalculator,
        *,
        step: int | str = "auto",
        target_nodes: int = 80_000,
        connectivity: str = "8",
    ) -> "SurfaceGraph":
        """
        Build a sparse surface graph from a distance calculator.

        Parameters
        ----------
        calculator : SurfaceDistanceCalculator
            Calculator whose prepared height map and scale factors define the
            surface. The graph uses ``calculator.heightmap``,
            ``calculator.pixel_size``, and ``calculator.voxel_depth``.
        step : int or "auto", optional
            Grid sampling step in pixels. ``1`` includes every pixel.
            ``"auto"`` chooses a step with :func:`choose_surface_graph_step`
            using ``target_nodes``.
        target_nodes : int, optional
            Approximate maximum node count used when ``step="auto"``.
        connectivity : {"4", "8", "16"}, optional
            Edge stencil. ``"4"`` is most grid-biased, ``"8"`` is the default,
            and ``"16"`` reduces directional bias at higher cost.

        Returns
        -------
        SurfaceGraph
            Sparse graph whose distances are in ``calculator.units``-compatible
            physical units.
        """
        return build_surface_graph(
            calculator.heightmap,
            pixel_size=calculator.pixel_size,
            voxel_depth=calculator.voxel_depth,
            step=step,
            target_nodes=target_nodes,
            connectivity=connectivity,
        )

    @classmethod
    def from_heightmap(
        cls,
        heightmap,
        *,
        pixel_size,
        voxel_depth=1.0,
        step: int | str = "auto",
        target_nodes: int = 80_000,
        connectivity: str = "8",
    ) -> "SurfaceGraph":
        """
        Build a sparse surface graph directly from a height map.

        Parameters
        ----------
        heightmap : np.ndarray
            Two-dimensional surface raster indexed as
            ``heightmap[row=y, column=x]``. Values are multiplied by
            ``voxel_depth`` when edge weights are constructed.
        pixel_size : float
            Physical size of one pixel in the ``x`` and ``y`` directions.
        voxel_depth : float, optional
            Physical scale factor for height values. Use ``1.0`` for height maps
            already in physical z units.
        step : int or "auto", optional
            Grid sampling step in pixels. Larger values create coarser, faster,
            more approximate graphs. ``"auto"`` uses ``target_nodes`` to choose
            a step.
        target_nodes : int, optional
            Approximate maximum node count used when ``step="auto"``.
        connectivity : {"4", "8", "16"}, optional
            Edge stencil used to connect sampled grid nodes.

        Returns
        -------
        SurfaceGraph
            Sparse graph suitable for approximate geodesic distance queries.
        """
        return build_surface_graph(
            heightmap,
            pixel_size=pixel_size,
            voxel_depth=voxel_depth,
            step=step,
            target_nodes=target_nodes,
            connectivity=connectivity,
        )

    def nearest_nodes(self, points_xy: np.ndarray) -> np.ndarray:
        """Return nearest graph node indices for ``(x, y)`` points."""
        points = _validate_xy_array(points_xy, name="points_xy")
        _, indices = self._tree.query(points)
        return np.asarray(indices, dtype=np.int64)

    def distance(self, p0_xy, p1_xy, *, return_path: bool = False):
        """Return approximate graph-geodesic distance between two points."""
        endpoints = np.vstack(
            [
                _validate_xy_point(p0_xy, name="p0_xy"),
                _validate_xy_point(p1_xy, name="p1_xy"),
            ]
        )
        source, target = self.nearest_nodes(endpoints)
        if return_path:
            dist, predecessors = dijkstra(
                self.graph,
                directed=False,
                indices=int(source),
                return_predecessors=True,
            )
            distance_value = float(dist[int(target)])
            path = self._path_from_predecessors(predecessors, int(source), int(target))
            return distance_value, path
        dist = dijkstra(self.graph, directed=False, indices=int(source))
        return float(dist[int(target)])

    def distances_from_source(
        self,
        source_xy_px,
        target_xy_px: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return graph-geodesic distances from one source to targets. 

        xy coordinates are in pixel units.

        If ``target_xy_px`` is omitted, distances to every graph node are returned.
        Otherwise, targets are snapped to their nearest graph nodes.
        """
        source = int(self.nearest_nodes(np.asarray([_validate_xy_point(source_xy_px)]))[0])
        dist = np.asarray(dijkstra(self.graph, directed=False, indices=source), dtype=float)
        if target_xy_px is None:
            return dist
        targets = self.nearest_nodes(target_xy_px)
        return dist[targets]

    def _path_from_predecessors(
        self,
        predecessors: np.ndarray,
        source: int,
        target: int,
    ) -> np.ndarray:
        if source == target:
            return self.xy[[source]].copy()
        if predecessors[target] < 0:
            return np.empty((0, 2), dtype=float)
        nodes = [target]
        current = target
        while current != source:
            current = int(predecessors[current])
            if current < 0:
                return np.empty((0, 2), dtype=float)
            nodes.append(current)
        nodes.reverse()
        return self.xy[np.asarray(nodes, dtype=np.int64)].copy()


def build_surface_graph(
    heightmap,
    *,
    pixel_size,
    voxel_depth=1.0,
    step: int | str = "auto",
    target_nodes: int = 80_000,
    connectivity: str = "8",
) -> SurfaceGraph:
    """
    Build a sparse graph approximation of a height-map surface.

    The graph samples the image grid every ``step`` pixels. Edges connect nearby
    sampled grid points and are weighted by 3-D Euclidean distances.
    """
    h = np.asarray(heightmap, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("heightmap must be a 2-D array")
    pixel_size = _validate_positive(pixel_size, "pixel_size")
    voxel_depth = _validate_positive(voxel_depth, "voxel_depth")
    if step == "auto":
        step = choose_surface_graph_step(h.shape, target_nodes=target_nodes)
        if step > 1:
            warnings.warn(
                f"Building an approximate surface graph with step={step}; "
                "increase target_nodes or pass a smaller step for higher accuracy.",
                RuntimeWarning,
                stacklevel=2,
            )
    step = int(step)
    if step <= 0:
        raise ValueError("step must be a positive integer or 'auto'")

    ys = _sampled_axis(h.shape[0], step)
    xs = _sampled_axis(h.shape[1], step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    z = h[yy.ravel(), xx.ravel()].astype(float)
    n_rows, n_cols = len(ys), len(xs)
    n_nodes = n_rows * n_cols
    if n_nodes > 200_000:
        warnings.warn(
            "surface graph has more than 200,000 nodes; construction and "
            "Dijkstra searches may be slow. Increase step or reduce target_nodes.",
            RuntimeWarning,
            stacklevel=2,
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    offsets = _connectivity_offsets(str(connectivity))

    # Use one direction per stencil edge, then add both orientations to create
    # an undirected sparse graph without relying on implicit symmetry.
    for row in range(n_rows):
        for col in range(n_cols):
            source = row * n_cols + col
            for drow, dcol in offsets:
                rr = row + drow
                cc = col + dcol
                if rr < 0 or rr >= n_rows or cc < 0 or cc >= n_cols:
                    continue
                target = rr * n_cols + cc
                dx = (xy[target, 0] - xy[source, 0]) * pixel_size
                dy = (xy[target, 1] - xy[source, 1]) * pixel_size
                dz = (z[target] - z[source]) * voxel_depth
                weight = math.sqrt(dx * dx + dy * dy + dz * dz)
                rows.extend([source, target])
                cols.extend([target, source])
                data.extend([weight, weight])

    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return SurfaceGraph(
        graph=graph, # type: ignore (type casting not well documented)
        xy=xy,
        z=z,
        shape=h.shape,
        step=step,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        connectivity=str(connectivity),
    )


def surface_straight_cell_distance(
    result,
    heightmap,
    i: int,
    j: int,
    *,
    prepared: bool = True,
    **kwargs,
) -> float:
    """Return straight-line surface distance between two result cell centers."""
    calc = SurfaceDistanceCalculator.from_result(result, heightmap, prepared=prepared)
    centers = cell_centers_xy_pixels(result)
    return calc.straight_distance(centers[int(i)], centers[int(j)], **kwargs)


def surface_graph_cell_distance(
    result,
    surface_graph: SurfaceGraph,
    i: int,
    j: int,
    *,
    return_path=False,
):
    """Return graph-geodesic surface distance between two result cell centers."""
    centers = cell_centers_xy_pixels(result)
    return surface_graph.distance(centers[int(i)], centers[int(j)], return_path=return_path)
