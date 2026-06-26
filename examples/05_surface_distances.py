"""Compute straight-line and graph-geodesic distances on a height-map surface."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import deprojpy as dp  # noqa: E402
from deprojpy.surface_distance import (  # noqa: E402
    SurfaceDistanceCalculator,
    SurfaceGraph,
    cell_centers_xy_pixels,
)

LABELS_PATH = REPOSITORY / "samples" / "Labels-2.tif"
HEIGHTMAP_PATH = REPOSITORY / "samples" / "HeightMap-2.tif"
OUTPUT_DIR = REPOSITORY / "examples" / "output"

PIXEL_SIZE = 0.183
VOXEL_DEPTH = 1.0
UNITS = "µm"
INVERT_Z = True


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Edit LABELS_PATH and HEIGHTMAP_PATH near the top of "
            "examples/05_surface_distances.py to point at your TIFF files."
        )


def run_surface_distance_example(
    labels_path: Path = LABELS_PATH,
    heightmap_path: Path = HEIGHTMAP_PATH,
    output_dir: Path = OUTPUT_DIR,
    *,
    dpi: int = 150,
):
    """Run the surface-distance example and return computed distances and output paths."""
    _require_file(labels_path)
    _require_file(heightmap_path)
    labels, heightmap = dp.load_label_heightmap_pair(labels_path, heightmap_path)
    result = dp.from_labels(
        labels,
        heightmap,
        pixel_size=PIXEL_SIZE,
        voxel_depth=VOXEL_DEPTH,
        units=UNITS,
        invert_z=INVERT_Z,
        drop_border_cells=False,
    )

    calc = SurfaceDistanceCalculator.from_result(
        result,
        heightmap,
        prepared=False,
        invert_z=INVERT_Z,
        inpaint_zeros=True,
        prune_zeros=True,
    )
    centers = cell_centers_xy_pixels(result)

    i = 10 if len(centers) > 10 else 0
    j = min(len(centers) - 1, 200) if len(centers) else 0
    p0 = centers[i]
    p1 = centers[j]

    d_xy = float(np.linalg.norm((p1 - p0) * result.pixel_size))
    d_center = float(np.linalg.norm(result.epicells[j].center - result.epicells[i].center))
    d_straight = calc.straight_distance(p0, p1)

    graph = SurfaceGraph.from_calculator(
        calc,
        step="auto",
        target_nodes=80_000,
        connectivity="16",
    )
    d_graph, path = graph.distance(p0, p1, return_path=True) # type: ignore

    subset = centers[: min(100, len(centers))]
    pairwise = calc.straight_pairwise_distances(subset)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(calc.heightmap, cmap="viridis")
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", linewidth=1.2, label="straight xy")
    if len(path):
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=1.0, label="graph path")
    ax.scatter([p0[0], p1[0]], [p0[1], p1[1]], c=["cyan", "orange"], s=25)
    ax.set_title("Surface-distance paths")
    ax.set_xlabel("x / column (pixels)")
    ax.set_ylabel("y / row (pixels)")
    ax.legend(loc="best")
    path_plot = plot_dir / "surface_distance_paths.png"
    figure.savefig(path_plot, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"2D xy distance:                 {d_xy:.3f} {calc.units}")
    print(f"3D Euclidean center distance:   {d_center:.3f} {calc.units}")
    print(f"Straight-line surface distance: {d_straight:.3f} {calc.units}")
    print(f"Graph-geodesic distance:        {d_graph:.3f} {calc.units}")
    print(f"Pairwise matrix shape:          {pairwise.shape}")
    print(f"Wrote path plot:                {path_plot}")

    return {
        "result": result,
        "calculator": calc,
        "surface_graph": graph,
        "pairwise": pairwise,
        "distances": {
            "xy": d_xy,
            "center_3d": d_center,
            "straight_surface": d_straight,
            "graph_geodesic": d_graph,
        },
        "path_plot": path_plot,
    }


if __name__ == "__main__":
    run_surface_distance_example()
