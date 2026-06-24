from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from .geometry import fit_ellipse_3d, fit_plane, polygon_metrics, reduce_boundary
from .heightmap import compute_curvatures, get_z, prepare_heightmap
from .labels import labels_to_objects
from .mask import mask_to_objects
from .models import DeprojResult, Epicell
from .objects import MaskObject


def _validate_common_inputs(
    segmentation: np.ndarray,
    heightmap: np.ndarray,
    *,
    segmentation_name: str,
    pixel_size: float,
    voxel_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    segmentation = np.asarray(segmentation)
    heightmap = np.asarray(heightmap)
    if segmentation.ndim != 2 or heightmap.ndim != 2:
        raise ValueError(
            f"{segmentation_name} and heightmap must both be 2-D arrays; "
            f"received shapes {segmentation.shape} and {heightmap.shape}"
        )
    if segmentation.shape != heightmap.shape:
        raise ValueError(
            f"{segmentation_name} and heightmap must have identical shapes; "
            f"got {segmentation.shape} and {heightmap.shape}"
        )
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be a positive finite number")
    if not np.isfinite(voxel_depth) or voxel_depth <= 0:
        raise ValueError("voxel_depth must be a positive finite number")
    return segmentation, heightmap


def _from_objects_and_graph(
    objects: list[MaskObject],
    graph: nx.Graph,
    heightmap: np.ndarray,
    *,
    source_shape: tuple[int, int],
    pixel_size: float = 1.0,
    voxel_depth: float = 1.0,
    units: str = "pixels",
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
    cell_id_policy: Literal["sequential", "source_label"] = "sequential",
) -> DeprojResult:
    """Run shared height-map deprojection from extracted 2-D objects and graph."""
    if not objects:
        raise ValueError("object parsing found no retained cells")
    median_area = np.median(
        [polygon_metrics(np.c_[obj.boundary, np.zeros(len(obj.boundary))])[2] for obj in objects]
    )
    smooth_scale = 2.0 * np.sqrt(median_area) / np.pi
    prepared = prepare_heightmap(
        heightmap, voxel_depth, smooth_scale, invert_z, inpaint_zeros, prune_zeros
    )
    if not prune_zeros and not np.all(np.isfinite(prepared)):
        raise ValueError(
            "heightmap contains NaN or infinite values after preprocessing; "
            "enable prune_zeros or provide a finite height map"
        )

    graph = graph.copy()
    bad_junctions: set[int] = set()
    for node_id, data in graph.nodes(data=True):
        xy = np.asarray(data["centroid"], dtype=float) * pixel_size
        z = get_z(xy[None, :], prepared, pixel_size)[0]
        data["centroid"] = np.array([xy[0], xy[1], z])
        if prune_zeros and (not np.isfinite(z) or z == 0):
            bad_junctions.add(int(node_id))

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
        px = int(np.clip(np.rint(center[0] / pixel_size), 0, source_shape[1] - 1))
        py = int(np.clip(np.rint(center[1] / pixel_size), 0, source_shape[0] - 1))
        local_curvature = np.array([field[py, px] for field in curvatures])
        source_label = obj.source_label
        cell_id = (
            int(source_label)
            if cell_id_policy == "source_label" and source_label is not None
            else len(cells) + 1
        )
        cells.append(
            Epicell(
                id=cell_id,
                source_label=source_label,
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
        source_shape=source_shape,
        prepared_heightmap=prepared,
    )


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
    """Deproject binary-ridge segmented cell contours onto a height-map surface."""
    mask, heightmap = _validate_common_inputs(
        mask,
        heightmap,
        segmentation_name="mask",
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
    )
    objects, graph = mask_to_objects(mask)
    return _from_objects_and_graph(
        objects,
        graph,
        heightmap,
        source_shape=mask.shape,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        units=units,
        invert_z=invert_z,
        inpaint_zeros=inpaint_zeros,
        prune_zeros=prune_zeros,
        cell_id_policy="sequential",
    )


def from_labels(
    labels: np.ndarray,
    heightmap: np.ndarray,
    pixel_size: float = 1.0,
    voxel_depth: float = 1.0,
    units: str = "pixels",
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
    *,
    background: int = 0,
    drop_border_cells: bool = True,
    min_junction_labels: int = 3,
    junction_background: int | None = None,
    junction_connectivity: int = 2,
    junction_merge_epsilon: float = 0.0,
) -> DeprojResult:
    """Deproject labeled-image cell contours onto a height-map surface."""
    labels, heightmap = _validate_common_inputs(
        labels,
        heightmap,
        segmentation_name="labels",
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
    )
    objects, graph = labels_to_objects(
        labels,
        background=background,
        drop_border_cells=drop_border_cells,
        min_junction_labels=min_junction_labels,
        junction_background=junction_background,
        junction_connectivity=junction_connectivity,
        junction_merge_epsilon=junction_merge_epsilon,
    )
    return _from_objects_and_graph(
        objects,
        graph,
        heightmap,
        source_shape=labels.shape,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        units=units,
        invert_z=invert_z,
        inpaint_zeros=inpaint_zeros,
        prune_zeros=prune_zeros,
        cell_id_policy="source_label",
    )
