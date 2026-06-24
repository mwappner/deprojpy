from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import from_heightmap, from_labels
from .io import load_label_heightmap_pair, load_tiff_pair
from .plotting import save_plots


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--pixel-size", type=float, default=0.183)
    parser.add_argument("--voxel-depth", type=float, default=1.0)
    parser.add_argument("--units", default="µm")
    parser.add_argument("--invert-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", type=Path, metavar="DIR", help="save plot PNG files in DIR")


def _print_summary(shape, result, frame) -> None:
    print(f"Input shape: {shape}")
    print(f"Retained cells: {len(result.epicells)}")
    print(
        "Junction graph: "
        f"{result.junction_graph.number_of_nodes()} nodes, "
        f"{result.junction_graph.number_of_edges()} edges"
    )
    print(f"DataFrame shape: {frame.shape}")
    summary_columns = ["area", "perimeter", "eccentricity", "n_neighbors"]
    print(frame[summary_columns].describe().to_string())


def _finish(args, segmentation, heightmap, result, *, labels=None) -> None:
    frame = result.to_dataframe()
    _print_summary(segmentation.shape, result, frame)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.csv)
        print(f"Wrote CSV: {args.csv}")
    if args.plots:
        paths = save_plots(
            args.plots,
            None if labels is not None else segmentation,
            heightmap,
            result,
            frame,
            labels=labels,
        )
        for path in paths:
            print(f"Wrote plot: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeProj-compatible mask/label and height-map analysis"
    )
    subparsers = parser.add_subparsers(dest="mode")

    binary = subparsers.add_parser("binary-mask", help="run the binary-ridge mask workflow")
    binary.add_argument("mask")
    binary.add_argument("heightmap")
    _add_common_options(binary)

    labels = subparsers.add_parser("labels", help="run the labeled-image workflow")
    labels.add_argument("labels")
    labels.add_argument("heightmap")
    _add_common_options(labels)
    labels.add_argument("--background", type=int, default=0)
    labels.add_argument("--drop-border-cells", action=argparse.BooleanOptionalAction, default=True)
    labels.add_argument("--junction-background", type=int)
    labels.add_argument("--min-junction-labels", type=int, default=3)
    labels.add_argument("--junction-merge-epsilon", type=float, default=0.0)
    labels.add_argument("--preprocess-labels", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in {"binary-mask", "labels", "-h", "--help"}:
        # Backward-compatible old form:
        # deprojpy-smoke Segmentation-2.tif HeightMap-2.tif [options]
        legacy = argparse.ArgumentParser(
            description="Run DeProj-compatible binary mask and height-map analysis"
        )
        legacy.add_argument("mask")
        legacy.add_argument("heightmap")
        _add_common_options(legacy)
        args = legacy.parse_args(argv)
        args.mode = "binary-mask"
    else:
        parser = _build_parser()
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    if args.mode == "labels":
        labels, heightmap = load_label_heightmap_pair(
            args.labels,
            args.heightmap,
            background=args.background,
            preprocess_labels=args.preprocess_labels,
        )
        result = from_labels(
            labels,
            heightmap,
            pixel_size=args.pixel_size,
            voxel_depth=args.voxel_depth,
            units=args.units,
            invert_z=args.invert_z,
            background=args.background,
            drop_border_cells=args.drop_border_cells,
            junction_background=args.junction_background,
            min_junction_labels=args.min_junction_labels,
            junction_merge_epsilon=args.junction_merge_epsilon,
        )
        _finish(args, labels, heightmap, result, labels=labels)
    else:
        mask, heightmap = load_tiff_pair(args.mask, args.heightmap)
        result = from_heightmap(
            mask,
            heightmap,
            pixel_size=args.pixel_size,
            voxel_depth=args.voxel_depth,
            units=args.units,
            invert_z=args.invert_z,
        )
        _finish(args, mask, heightmap, result)


if __name__ == "__main__":
    main()
