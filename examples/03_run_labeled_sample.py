"""Run deprojpy from a labeled-cell image and height map."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import deprojpy as dp  # noqa: E402

LABELS_PATH = REPOSITORY / "samples" / "Labels-2.tif"
HEIGHTMAP_PATH = REPOSITORY / "samples" / "HeightMap-2.tif"
OUTPUT_DIR = REPOSITORY / "examples" / "output"

PIXEL_SIZE = 0.183
VOXEL_DEPTH = 1.0
UNITS = "µm"
BACKGROUND = 0


def run_labeled_sample(output_dir: Path = OUTPUT_DIR):
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
    frame = result.to_dataframe()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "labeled_measurements.csv"
    result.to_csv(csv_path)

    print(f"Input labels: {LABELS_PATH}")
    print(f"Input height map: {HEIGHTMAP_PATH}")
    print(f"Retained cells: {len(result.epicells)}")
    print(f"Junction graph nodes: {result.junction_graph.number_of_nodes()}")
    print(f"Junction graph edges: {result.junction_graph.number_of_edges()}")
    print(f"Wrote CSV: {csv_path}")
    print(frame[["id", "source_label", "area", "perimeter", "n_neighbors"]].head().to_string())
    return result, frame, csv_path


if __name__ == "__main__":
    run_labeled_sample()
