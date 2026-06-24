"""Generate plot outputs for the labeled-image deprojpy workflow."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import deprojpy as dp  # noqa: E402
from deprojpy.plotting import plot_label_objects, save_plots  # noqa: E402

LABELS_PATH = REPOSITORY / "samples" / "Labels-2.tif"
HEIGHTMAP_PATH = REPOSITORY / "samples" / "HeightMap-2.tif"
OUTPUT_DIR = REPOSITORY / "examples" / "output" / "labeled_plots"

PIXEL_SIZE = 0.183
VOXEL_DEPTH = 1.0
UNITS = "µm"
BACKGROUND = 0
DPI = 150


def make_labeled_plots(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    labels, heightmap = dp.load_label_heightmap_pair(LABELS_PATH, HEIGHTMAP_PATH)
    result = dp.from_labels(
        labels,
        heightmap,
        pixel_size=PIXEL_SIZE,
        voxel_depth=VOXEL_DEPTH,
        units=UNITS,
        invert_z=True,
        background=BACKGROUND,
    )
    paths = save_plots(
        output_dir,
        None,
        heightmap,
        result,
        labels=labels,
        features=("area", "eccentricity"),
        dpi=DPI,
    )
    fig, ax = plot_label_objects(
        labels,
        result,
        background=BACKGROUND,
        title="Detailed label contours and DeProj boundaries",
    )
    custom_path = output_dir / "label_objects_custom.png"
    fig.savefig(custom_path, dpi=DPI, bbox_inches="tight")
    paths.append(custom_path)
    print(f"Wrote plots under: {output_dir}")
    for path in paths:
        print(f"  {path}")
    return paths


if __name__ == "__main__":
    make_labeled_plots()
