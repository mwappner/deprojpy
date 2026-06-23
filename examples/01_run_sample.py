"""Minimal script-style deprojpy example.

Edit the constants in the "User settings" block, then run this file from your
editor, terminal, or copy the cells into a notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import deprojpy as dp  # noqa: E402

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

SAMPLES_DIR = REPOSITORY / "samples"
MASK_PATH = SAMPLES_DIR / "Segmentation-2.tif"
HEIGHTMAP_PATH = SAMPLES_DIR / "HeightMap-2.tif"

OUTPUT_DIR = REPOSITORY / "examples" / "output"

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
            "examples/01_run_sample.py to point at your TIFF files."
        )


def run_sample(
    mask_path: Path = MASK_PATH,
    heightmap_path: Path = HEIGHTMAP_PATH,
    output_dir: Path = OUTPUT_DIR,
):
    """Run the sample workflow and return ``(result, dataframe, csv_path)``."""
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
    frame = result.to_dataframe()

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "measurements.csv"
    result.to_csv(csv_path)

    print(f"Input mask: {mask_path}")
    print(f"Input height map: {heightmap_path}")
    print(f"Image shape: {mask.shape}")
    print(f"Retained cells: {len(result.epicells)}")
    print(
        "Junction graph: "
        f"{result.junction_graph.number_of_nodes()} nodes, "
        f"{result.junction_graph.number_of_edges()} edges"
    )
    print(f"Wrote CSV: {csv_path}")
    print("\nFirst rows:")
    print(frame.head().to_string(index=False))
    print("\nCore measurement summary:")
    print(frame[["area", "perimeter", "eccentricity", "n_neighbors"]].describe().to_string())
    return result, frame, csv_path


if __name__ == "__main__":
    run_sample()
