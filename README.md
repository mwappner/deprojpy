# deprojpy

`deprojpy` is a small behavioral Python port of the computational
`deproj.from_heightmap` workflow from the MATLAB DeProj project. It converts a
black-ridge/white-cell mask into cell polygons and a junction graph, maps the
polygons onto a height map, and exports corrected morphology measurements.

This first milestone is validated with synthetic geometry tests and the
repository sample. It is not yet a certified numerical clone of MATLAB DeProj.

## Install and run

```bash
cd DeProj-python
python -m pip install -e '.[test]'
pytest

deprojpy-smoke \
  ../DeProj-matlab/samples/Segmentation-2.tif \
  ../DeProj-matlab/samples/HeightMap-2.tif \
  --csv measurements.csv
```

Or from Python:

```python
import deprojpy as dp

mask, heightmap = dp.load_tiff_pair("Segmentation-2.tif", "HeightMap-2.tif")
result = dp.from_heightmap(
    mask, heightmap,
    pixel_size=0.183,
    voxel_depth=1.0,
    units="µm",
    invert_z=True,
    inpaint_zeros=True,
    prune_zeros=True,
)
df = result.to_dataframe()
```

Coordinates returned by the package are explicit `(x, y, z)` values. XY uses
zero-based pixel centers before physical scaling. Zeros in the height map are
optionally filled with biharmonic interpolation; any remaining zero/NaN cells
are pruned.

The sample has 426 retained cells and 920 actual junction nodes. The original
README reports 1840 because MATLAB's `numel(junction_graph.Nodes)` counts both
columns of its 920-by-2 node table.
