from __future__ import annotations

import argparse
from pathlib import Path

from .core import from_heightmap
from .io import load_tiff_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DeProj sample smoke test")
    parser.add_argument("mask")
    parser.add_argument("heightmap")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--pixel-size", type=float, default=0.183)
    parser.add_argument("--voxel-depth", type=float, default=1.0)
    parser.add_argument("--units", default="µm")
    parser.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True)
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
    print(
        f"shape={mask.shape}, cells={len(result.epicells)}, "
        f"junction_nodes={result.junction_graph.number_of_nodes()}, "
        f"junction_edges={result.junction_graph.number_of_edges()}"
    )
    print(frame[["area", "perimeter", "n_neighbors"]].describe())
    if args.csv:
        result.to_csv(args.csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
