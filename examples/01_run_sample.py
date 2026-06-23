from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import deprojpy as dp  # noqa: E402


def _sample_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, Path]:
    if args.mask or args.heightmap:
        if not (args.mask and args.heightmap):
            parser.error("--mask and --heightmap must be supplied together")
        mask = args.mask
        heightmap = args.heightmap
    else:
        samples = args.samples or REPOSITORY / "samples"
        mask = samples / "Segmentation-2.tif"
        heightmap = samples / "HeightMap-2.tif"

    missing = [path for path in (mask, heightmap) if not path.is_file()]
    if missing:
        parser.error(
            "sample file(s) not found: "
            + ", ".join(str(path) for path in missing)
            + "\nUse --samples DIR or pass --mask PATH --heightmap PATH."
        )
    return mask, heightmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bundled DeProj sample.")
    parser.add_argument("--samples", type=Path, help="directory containing Segmentation-2.tif")
    parser.add_argument("--mask", type=Path, help="explicit segmentation TIFF")
    parser.add_argument("--heightmap", type=Path, help="explicit height-map TIFF")
    parser.add_argument("--out", type=Path, default=REPOSITORY / "examples" / "output")
    parser.add_argument("--pixel-size", type=float, default=0.183)
    parser.add_argument("--voxel-depth", type=float, default=1.0)
    parser.add_argument("--units", default="µm")
    parser.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    mask_path, heightmap_path = _sample_paths(args, parser)
    mask, heightmap = dp.load_tiff_pair(mask_path, heightmap_path)
    result = dp.from_heightmap(
        mask,
        heightmap,
        pixel_size=args.pixel_size,
        voxel_depth=args.voxel_depth,
        units=args.units,
        invert_z=args.invert_z,
        inpaint_zeros=True,
        prune_zeros=True,
    )
    frame = result.to_dataframe()

    results_dir = args.out / "results"
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


if __name__ == "__main__":
    main()
