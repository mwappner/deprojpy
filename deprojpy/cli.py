from __future__ import annotations

import argparse
from pathlib import Path

from .core import from_heightmap
from .io import load_tiff_pair
from .plotting import save_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeProj-compatible mask and height-map analysis"
    )
    parser.add_argument("mask")
    parser.add_argument("heightmap")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--pixel-size", type=float, default=0.183)
    parser.add_argument("--voxel-depth", type=float, default=1.0)
    parser.add_argument("--units", default="µm")
    parser.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--plots",
        type=Path,
        metavar="DIR",
        help="save plot PNG files in DIR",
    )
    args = parser.parse_args()
    mask, heightmap = load_tiff_pair(args.mask, args.heightmap)
    result = from_heightmap(
        mask,
        heightmap,
        pixel_size=args.pixel_size,
        voxel_depth=args.voxel_depth,
        units=args.units,
        invert_z=args.invert_z,
    )
    frame = result.to_dataframe()
    print(f"Input shape: {mask.shape}")
    print(f"Retained cells: {len(result.epicells)}")
    print(
        "Junction graph: "
        f"{result.junction_graph.number_of_nodes()} nodes, "
        f"{result.junction_graph.number_of_edges()} edges"
    )
    print(f"DataFrame shape: {frame.shape}")
    summary_columns = ["area", "perimeter", "eccentricity", "n_neighbors"]
    print(frame[summary_columns].describe().to_string())
    print("Missing key values:")
    for column in summary_columns:
        missing = int(frame[column].isna().sum())
        fraction = missing / len(frame) if len(frame) else 0.0
        print(f"  {column}: {missing}/{len(frame)} ({fraction:.1%})")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.csv)
        print(f"Wrote CSV: {args.csv}")
    if args.plots:
        paths = save_plots(args.plots, mask, heightmap, result, frame)
        for path in paths:
            print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
