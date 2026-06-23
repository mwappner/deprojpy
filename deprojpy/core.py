from __future__ import annotations

import numpy as np

from .geometry import fit_ellipse_3d, fit_plane, polygon_metrics, reduce_boundary
from .heightmap import compute_curvatures, get_z, prepare_heightmap
from .mask import mask_to_objects
from .models import DeprojResult, Epicell


def from_heightmap(
    mask: np.ndarray,
    heightmap: np.ndarray,
    pixel_size: float = 1.0,
    voxel_depth: float = 1.0,
    units: str = "pixels",
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
) -> DeprojResult:
    """Run the behavioral Python port of DeProj's ``from_heightmap``."""
    mask = np.asarray(mask)
    heightmap = np.asarray(heightmap)
    if mask.shape != heightmap.shape or mask.ndim != 2:
        raise ValueError("mask and heightmap must be 2-D arrays with identical shapes")
    if pixel_size <= 0 or voxel_depth <= 0:
        raise ValueError("pixel_size and voxel_depth must be positive")

    objects, graph = mask_to_objects(mask)
    median_area = np.median(
        [polygon_metrics(np.c_[obj.boundary, np.zeros(len(obj.boundary))])[2] for obj in objects]
    )
    smooth_scale = 2.0 * np.sqrt(median_area) / np.pi
    prepared = prepare_heightmap(
        heightmap, voxel_depth, smooth_scale, invert_z, inpaint_zeros, prune_zeros
    )

    bad_junctions: set[int] = set()
    for node_id, data in graph.nodes(data=True):
        xy = data["centroid"] * pixel_size
        z = get_z(xy[None, :], prepared, pixel_size)[0]
        data["centroid"] = np.array([xy[0], xy[1], z])
        if prune_zeros and (not np.isfinite(z) or z == 0):
            bad_junctions.add(node_id)

    curvatures = compute_curvatures(prepared, smooth_scale, pixel_size)
    cells: list[Epicell] = []
    for obj in objects:
        if bad_junctions.intersection(map(int, obj.junction_ids)):
            continue
        xy = obj.boundary * pixel_size
        z = get_z(xy, prepared, pixel_size)
        if prune_zeros and (not np.all(np.isfinite(z)) or np.any(z == 0)):
            continue
        boundary = np.c_[xy, z]
        reduced = reduce_boundary(boundary)
        area, perimeter, area2d, perimeter2d = polygon_metrics(reduced)
        rotation, euler = fit_plane(boundary)
        ellipse, eccentricity, direction = fit_ellipse_3d(boundary, rotation)
        center = boundary.mean(axis=0)
        px = int(np.clip(np.rint(center[0] / pixel_size), 0, mask.shape[1] - 1))
        py = int(np.clip(np.rint(center[1] / pixel_size), 0, mask.shape[0] - 1))
        local_curvature = np.array([field[py, px] for field in curvatures])
        cells.append(
            Epicell(
                id=len(cells) + 1,
                boundary=reduced,
                center=center,
                junction_ids=obj.junction_ids.copy(),
                n_neighbors=len(obj.junction_ids),
                area=area,
                perimeter=perimeter,
                euler_angles=euler,
                curvatures=local_curvature,
                ellipse_fit=ellipse,
                eccentricity=eccentricity,
                proj_direction=direction,
                uncorrected_area=area2d,
                uncorrected_perimeter=perimeter2d,
                area_error=area - area2d,
                perimeter_error=perimeter - perimeter2d,
            )
        )
    graph.remove_nodes_from(bad_junctions)
    return DeprojResult(
        cells,
        graph,
        units=units,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        source_shape=mask.shape,
        prepared_heightmap=prepared,
    )

