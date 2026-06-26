from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numba import njit, prange

from ._helpers import (
    _as_pixel_xy,
    _validate_positive,
    _validate_xy_point,
    sample_height_at_xy,
)


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
    input_units: Literal["pixel", "physical"] = "pixel",
) -> float:
    """
    Approximate surface distance along the straight segment in ``xy``.

    ``p0_xy`` and ``p1_xy`` are interpreted according to ``input_units``.
    Distances are always returned in physical units.
    """
    h = np.asarray(heightmap, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("heightmap must be a 2-D array")
    pixel_size = _validate_positive(pixel_size, "pixel_size")
    voxel_depth = _validate_positive(voxel_depth, "voxel_depth")
    p0 = _validate_xy_point(
        _as_pixel_xy(p0_xy, pixel_size=pixel_size, input_units=input_units),
        name="p0_xy",
    )
    p1 = _validate_xy_point(
        _as_pixel_xy(p1_xy, pixel_size=pixel_size, input_units=input_units),
        name="p1_xy",
    )
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
