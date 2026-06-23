from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import tifffile

import deprojpy as dp
from deprojpy.cli import _save_diagnostics
from deprojpy.diagnostics import (
    plot_3d_boundaries,
    plot_feature_histograms,
    plot_feature_map,
    plot_heightmap_with_centers,
    plot_mask_objects,
)
from deprojpy.models import DATAFRAME_COLUMNS, DeprojResult


def single_cell_mask(shape=(15, 15)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:11, 4:11] = 255
    return mask


def test_flat_surface_preserves_projected_metrics():
    mask = single_cell_mask()
    result = dp.from_heightmap(
        mask,
        np.full(mask.shape, 5.0),
        inpaint_zeros=False,
        prune_zeros=False,
    )
    cell = result.epicells[0]
    assert np.isclose(cell.area, cell.uncorrected_area)
    assert np.isclose(cell.perimeter, cell.uncorrected_perimeter)


def test_tilted_surface_has_finite_larger_metrics():
    mask = single_cell_mask()
    y, x = np.indices(mask.shape)
    heightmap = 5.0 + 0.2 * x + 0.1 * y
    result = dp.from_heightmap(mask, heightmap, inpaint_zeros=False, prune_zeros=False)
    cell = result.epicells[0]
    assert np.isfinite(
        [cell.area, cell.perimeter, cell.uncorrected_area, cell.uncorrected_perimeter]
    ).all()
    assert cell.area >= cell.uncorrected_area
    assert cell.perimeter >= cell.uncorrected_perimeter


@pytest.mark.parametrize("parameter", ["pixel_size", "voxel_depth"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_physical_scales_must_be_positive_finite(parameter, value):
    mask = single_cell_mask()
    arguments = {parameter: value}
    with pytest.raises(ValueError, match=parameter):
        dp.from_heightmap(mask, np.ones(mask.shape), **arguments)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        dp.from_heightmap(single_cell_mask(), np.ones((12, 12)))


def test_heightmap_requires_finite_values():
    mask = single_cell_mask()
    with pytest.raises(ValueError, match="finite"):
        dp.from_heightmap(mask, np.full(mask.shape, np.nan))


def test_all_zero_heightmap_has_helpful_pruning_error():
    mask = single_cell_mask()
    with pytest.raises(ValueError, match="only zeros"):
        dp.from_heightmap(mask, np.zeros(mask.shape))


def test_dataframe_schema_and_csv_export(tmp_path):
    mask = single_cell_mask()
    result = dp.from_heightmap(mask, np.ones(mask.shape), inpaint_zeros=False, prune_zeros=False)
    frame = result.to_dataframe()
    assert frame.columns.tolist() == DATAFRAME_COLUMNS
    output = tmp_path / "nested" / "measurements.csv"
    output.parent.mkdir()
    result.to_csv(output)
    assert output.is_file()
    assert pd.read_csv(output).columns.tolist() == DATAFRAME_COLUMNS


def test_load_tiff_pair_rejects_mismatch(tmp_path):
    mask_path = tmp_path / "mask.tif"
    heightmap_path = tmp_path / "heightmap.tif"
    tifffile.imwrite(mask_path, np.zeros((10, 10), dtype=np.uint8))
    tifffile.imwrite(heightmap_path, np.zeros((11, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        dp.load_tiff_pair(mask_path, heightmap_path)


def test_diagnostics_handle_empty_results():
    result = DeprojResult([], nx.Graph(), source_shape=(10, 10))
    frame = result.to_dataframe()
    plotters = [
        lambda: plot_mask_objects(np.zeros((10, 10)), result),
        lambda: plot_heightmap_with_centers(np.full((10, 10), np.nan), result),
        lambda: plot_feature_map(result),
        lambda: plot_3d_boundaries(result),
        lambda: plot_feature_histograms(frame),
    ]
    for plot in plotters:
        figure, _ = plot()
        assert isinstance(figure, plt.Figure)
        plt.close(figure)


def test_diagnostics_can_be_saved(tmp_path):
    mask = single_cell_mask()
    result = dp.from_heightmap(mask, np.ones(mask.shape), inpaint_zeros=False, prune_zeros=False)
    paths = _save_diagnostics(
        tmp_path / "diagnostics",
        mask,
        np.ones(mask.shape),
        result,
        result.to_dataframe(),
    )
    assert {path.name for path in paths} == {
        "mask_objects.png",
        "heightmap_centers.png",
        "area_map.png",
        "feature_histograms.png",
        "boundaries_3d.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_feature_plots_handle_nan_values():
    mask = single_cell_mask()
    result = dp.from_heightmap(
        mask, np.ones(mask.shape), inpaint_zeros=False, prune_zeros=False
    )
    result.epicells[0].area = np.nan
    for plot in (plot_feature_map, plot_3d_boundaries):
        figure, _ = plot(result, "area")
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
