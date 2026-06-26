# deprojpy

`deprojpy` is a behavioral Python port of the MATLAB DeProj
`deproj.from_heightmap` workflow. It parses a segmented epithelial-cell mask,
maps cell contours onto a height-map surface, computes corrected morphology,
and exports the measurements.

This package is validated with synthetic invariants and the original DeProj
sample images. It is not yet certified against MATLAB golden outputs.

## Installation

From a local checkout:

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m compileall deprojpy
```

The test suite uses the local sample TIFF files under `samples/`.

## Python usage

```python
import deprojpy as dp

mask, heightmap = dp.load_tiff_pair(
    "Segmentation-2.tif",
    "HeightMap-2.tif",
)
result = dp.from_heightmap(
    mask,
    heightmap,
    pixel_size=0.183,
    voxel_depth=1.0,
    units="µm",
    invert_z=True,
    inpaint_zeros=True,
    prune_zeros=True,
)

df = result.to_dataframe()
result.to_csv("measurements.csv")
```

Input arrays use image indexing `(row, column)`. Returned boundaries, centers,
and junction centroids use geometric `(x, y, z)` order in the selected physical
units.

The segmentation must be a binary-like image with black/zero connected ridges
and one nonzero value for cell interiors. The height map must be a 2-D image of
the same shape whose values encode the tissue's Z position.

## Labeled-image input

For label images where each pixel stores a cell ID, install the sibling
`labelimage-tools` package locally and use `from_labels`:

```bash
python -m pip install -e /home/mw/Documents/Pasteur/Code/tissue_processing/labelimage-tools
```

```python
import deprojpy as dp

labels, heightmap = dp.load_label_heightmap_pair(
    "samples/Labels-2.tif",
    "samples/HeightMap-2.tif",
)
result = dp.from_labels(
    labels,
    heightmap,
    pixel_size=0.183,
    voxel_depth=1.0,
    units="µm",
    invert_z=True,
)
df = result.to_dataframe()
```

The labeled path starts from detailed label contours, not simplified
vertex-model polygons. Cell boundaries are then simplified in pixel coordinates
before height-map sampling; the internal default tolerance is 0.5 px, so this
step is independent of physical `pixel_size`. Junctions are detected from 3×3
label neighborhoods; subpixel junction centroids become graph nodes; cells are
associated to junctions by label membership and ordered along their contour.
Original segmentation IDs are preserved as `source_label` in the result
dataframe.

## Surface distances

DeProjPy also includes reusable calculators for distances constrained to the
height-map surface. A straight-line surface distance samples the height map along
the straight segment in `xy` and measures the lifted 3-D polyline. This is fast
and useful for all-pairs cell-center distances, but it is not a true geodesic.
For approximate geodesics, build a sparse `SurfaceGraph`; `connectivity="8"` is
a good default, while `"16"` reduces grid-direction bias at higher cost.

```python
from deprojpy.surface_distance import (
    SurfaceDistanceCalculator,
    SurfaceGraph,
    cell_centers_xy_pixels,
)

calc = SurfaceDistanceCalculator.from_result(
    result,
    heightmap,
    prepared=False,
    invert_z=True,
)
boundary_calc = SurfaceDistanceCalculator.from_cell_boundaries(
    result,
    method="linear",
    extrapolation="linear",
)
centers = cell_centers_xy_pixels(result)

d_heightmap = calc.straight_distance(centers[10], centers[200])
d_boundary = boundary_calc.straight_distance(centers[10], centers[200])
graph = SurfaceGraph.from_calculator(boundary_calc, step="auto", connectivity="16")
d_graph, path_xy = graph.distance(centers[10], centers[200], return_path=True)
D = calc.straight_pairwise_distances(centers[:100])
```

Use `SurfaceDistanceCalculator.from_result(...)` when you have a DeProj result,
so pixel size, units, and height-map preparation stay consistent while the
surface is still built from a height map. Use
`SurfaceDistanceCalculator.from_cell_boundaries(result)` when path
visualization or distance calculations should follow a raster surface
interpolated from the deprojected cell-boundary `(x, y, z)` points instead of
the original height map; `method` controls the scattered interpolation and
`extrapolation` controls how pixels outside the boundary-point convex hull are
filled. If you only have exported cell centers or external point arrays, use
`SurfaceDistanceCalculator.from_heightmap(...)` and pass `pixel_size` and
`voxel_depth` manually. All-pairs graph geodesics are intentionally not computed
by default; use `SurfaceGraph.distances_from_source(...)` when comparing one
source to many targets.

Consult the cookbook in [`docs/cookbook.md`](docs/cookbook.md) for
copy-pastable Python snippets: exporting measurements, saving plots,
customizing feature maps, composing plots on matplotlib axes, and checking
whether a run looks sane.

## Optional command line helper

The command-line entry point is useful for quick smoke checks, but it is not the
main API surface.

```bash
deprojpy-smoke \
  samples/Segmentation-2.tif \
  samples/HeightMap-2.tif \
  --pixel-size 0.183 \
  --voxel-depth 1.0 \
  --units µm \
  --invert-z \
  --csv measurements.csv
```

Add `--plots plots/` to save mask, height-map, feature-map,
histogram, and 3-D-boundary PNGs.

For labeled images:

```bash
deprojpy-smoke labels samples/Labels-2.tif samples/HeightMap-2.tif --csv labels.csv
```

## Plots and plotting

```python
import deprojpy as dp

mask, heightmap = dp.load_tiff_pair("samples/Segmentation-2.tif", "samples/HeightMap-2.tif")
result = dp.from_heightmap(mask, heightmap, pixel_size=0.183, units="µm")
paths = dp.save_plots(
    "plots",
    mask,
    heightmap,
    result,
    features=("area", "eccentricity"),
)
```

Plotting helpers in `deprojpy.plotting` accept optional matplotlib axes, so
they can be used directly in scripts and notebooks without calling
`plt.show()`.

## Validation status

With the original sample files, the expected result is:

- image shape `(282, 508)`;
- 426 retained cells;
- 920 actual junction graph nodes;
- finite positive cell areas and perimeters.

The original MATLAB README prints 1,840 junctions because MATLAB `numel`
counts both columns of its 920-by-2 node table. Numerical parity with MATLAB
ellipse fits and other measurements remains to be checked against future
golden outputs.
