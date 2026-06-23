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

The test suite finds optional MATLAB samples through
`DEPROJ_MATLAB_SAMPLES`, common sibling checkout paths, or `./samples`.
Sample-dependent tests skip clearly when those files are unavailable.

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

## Command line

```bash
deprojpy-smoke \
  ../DeProj-matlab/samples/Segmentation-2.tif \
  ../DeProj-matlab/samples/HeightMap-2.tif \
  --pixel-size 0.183 \
  --voxel-depth 1.0 \
  --units µm \
  --invert-z \
  --csv measurements.csv
```

Add `--diagnostics diagnostics/` to save mask, height-map, feature-map,
histogram, and 3-D-boundary PNGs.

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
