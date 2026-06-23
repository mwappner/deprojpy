from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Generate a small deprojpy plot gallery.")
    parser.add_argument("--samples", type=Path, help="directory containing Segmentation-2.tif")
    parser.add_argument("--mask", type=Path, help="explicit segmentation TIFF")
    parser.add_argument("--heightmap", type=Path, help="explicit height-map TIFF")
    parser.add_argument("--out", type=Path, default=REPOSITORY / "examples" / "output")
    parser.add_argument("--pixel-size", type=float, default=0.183)
    parser.add_argument("--voxel-depth", type=float, default=1.0)
    parser.add_argument("--units", default="µm")
    parser.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=150)
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

    plot_dir = args.out / "plots"
    paths = dp.save_plots(
        plot_dir,
        mask,
        heightmap,
        result,
        frame,
        features=("area", "eccentricity"),
        dpi=args.dpi,
    )

    figure, ax = plot_3d_boundaries(result, "area")
    ax.view_init(azim=115, elev=1) # type: ignore
    boundary_path = plot_dir / "boundaries_3d.png"
    figure.savefig(boundary_path, dpi=args.dpi, bbox_inches="tight")
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
    fig.savefig(custom_path, dpi=args.dpi, bbox_inches="tight")
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
    fig.savefig(neighbor_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(neighbor_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    plot_feature_map(
        result,
        "area_error",
        ax=axes[0, 0],
        title="Absolute area error",
        colorbar_label=f"area error ({args.units}²)",
    )
    plot_feature_map(
        result,
        "perimeter_error",
        ax=axes[0, 1],
        title="Absolute perimeter error",
        colorbar_label=f"perimeter error ({args.units})",
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
    fig.savefig(error_maps_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(error_maps_path)

    print(f"Input mask: {mask_path}")
    print(f"Input height map: {heightmap_path}")
    print(f"Retained cells: {len(result.epicells)}")
    print(f"Wrote plots under: {plot_dir}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
