"""Script-style gallery of deprojpy plotting helpers.

Edit the constants in the "User settings" block, then run this file from your
editor, terminal, or copy selected sections into a notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import deprojpy as dp  # noqa: E402
from deprojpy.plotting import (  # noqa: E402
    plot_3d_boundaries,
    plot_feature_map,
    plot_relative_error_map,
)

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

SAMPLES_DIR = REPOSITORY / "samples"
MASK_PATH = SAMPLES_DIR / "Segmentation-2.tif"
HEIGHTMAP_PATH = SAMPLES_DIR / "HeightMap-2.tif"

OUTPUT_DIR = REPOSITORY / "examples" / "output"
DPI = 150

PIXEL_SIZE = 0.183
VOXEL_DEPTH = 1.0
UNITS = "µm"
INVERT_Z = True
INPAINT_ZEROS = True
PRUNE_ZEROS = True


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Edit MASK_PATH and HEIGHTMAP_PATH near the top of "
            "examples/02_plot_gallery.py to point at your TIFF files."
        )


def analyze_sample(mask_path: Path = MASK_PATH, heightmap_path: Path = HEIGHTMAP_PATH):
    """Load the sample files and return ``(mask, heightmap, result, dataframe)``."""
    _require_file(mask_path)
    _require_file(heightmap_path)
    mask, heightmap = dp.load_tiff_pair(mask_path, heightmap_path)
    result = dp.from_heightmap(
        mask,
        heightmap,
        pixel_size=PIXEL_SIZE,
        voxel_depth=VOXEL_DEPTH,
        units=UNITS,
        invert_z=INVERT_Z,
        inpaint_zeros=INPAINT_ZEROS,
        prune_zeros=PRUNE_ZEROS,
    )
    return mask, heightmap, result, result.to_dataframe()


def make_gallery(
    mask_path: Path = MASK_PATH,
    heightmap_path: Path = HEIGHTMAP_PATH,
    output_dir: Path = OUTPUT_DIR,
    dpi: int = DPI,
) -> list[Path]:
    """Generate the gallery images and return their paths."""
    mask, heightmap, result, frame = analyze_sample(mask_path, heightmap_path)
    plot_dir = output_dir / "plots"

    paths = dp.save_plots(
        plot_dir,
        mask,
        heightmap,
        result,
        frame,
        features=("area", "eccentricity"),
        dpi=dpi,
    )

    figure, ax = plot_3d_boundaries(result, "area")
    ax.view_init(azim=115, elev=1)  # type: ignore[attr-defined]
    boundary_path = plot_dir / "boundaries_3d.png"
    figure.savefig(boundary_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    plot_feature_map(
        result,
        "eccentricity",
        ax=axes[0],
        title="Eccentricity",
        edgecolor="white",
        linewidth=0.1,
    )
    plot_feature_map(
        result,
        "n_neighbors",
        ax=axes[1],
        title="Neighbor count",
        cmap="magma",
        edgecolor="white",
        linewidth=0.1,
    )
    fig.suptitle("Custom feature-map subplots")
    fig.tight_layout()
    custom_path = plot_dir / "custom_feature_maps.png"
    fig.savefig(custom_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(custom_path)

    fig, ax = plot_feature_map(
        result,
        "n_neighbors",
        title="Neighbor count",
        cmap="magma",
        edgecolor="white",
        linewidth=0.1,
        colorbar_label="neighbors",
    )
    neighbor_path = plot_dir / "neighbor_map.png"
    fig.savefig(neighbor_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(neighbor_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    plot_feature_map(
        result,
        "area_error",
        ax=axes[0, 0],
        title="Absolute area error",
        colorbar_label=f"area error ({UNITS}²)",
    )
    plot_feature_map(
        result,
        "perimeter_error",
        ax=axes[0, 1],
        title="Absolute perimeter error",
        colorbar_label=f"perimeter error ({UNITS})",
    )
    plot_relative_error_map(
        result,
        "area",
        ax=axes[1, 0],
        title="Relative area error",
        colorbar_label="relative area error (%)",
    )
    plot_relative_error_map(
        result,
        "perimeter",
        ax=axes[1, 1],
        title="Relative perimeter error",
        colorbar_label="relative perimeter error (%)",
    )
    fig.suptitle("Absolute and relative deprojection correction maps")
    error_maps_path = plot_dir / "error_maps_2x2.png"
    fig.savefig(error_maps_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(error_maps_path)

    print(f"Input mask: {mask_path}")
    print(f"Input height map: {heightmap_path}")
    print(f"Retained cells: {len(result.epicells)}")
    print(f"Wrote plots under: {plot_dir}")
    for path in paths:
        print(f"  {path}")
    return paths


if __name__ == "__main__":
    make_gallery()
