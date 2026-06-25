from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import tifffile

import deprojpy as dp
import deprojpy.core as core
from deprojpy.models import DATAFRAME_COLUMNS, DeprojResult
from deprojpy.objects import MaskObject
from deprojpy.plotting import (
    plot_3d_boundaries,
    plot_feature_histograms,
    plot_feature_map,
    plot_heightmap_with_centers,
    plot_mask_objects,
    plot_relative_error_map,
    save_plots,
)


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


def test_boundary_simplification_happens_inside_shared_pipeline(monkeypatch):
    top = np.column_stack([np.arange(2, 13), np.full(11, 2)])
    right = np.column_stack([np.full(8, 12), np.arange(3, 11)])
    bottom = np.column_stack([np.arange(11, 1, -1), np.full(10, 10)])
    left = np.column_stack([np.full(7, 2), np.arange(9, 2, -1)])
    boundary = np.vstack([top, right, bottom, left]).astype(float)
    obj = MaskObject(
        boundary=boundary,
        center=boundary.mean(axis=0),
        junction_ids=np.array([], int),
    )
    heightmap = np.ones((15, 15), dtype=float)

    monkeypatch.setattr(core, "BOUNDARY_SIMPLIFICATION_TOLERANCE_PX", 0.0)
    detailed = core._from_objects_and_graph(
        [obj],
        nx.Graph(),
        heightmap,
        source_shape=heightmap.shape,
        inpaint_zeros=False,
        prune_zeros=False,
    )

    monkeypatch.setattr(core, "BOUNDARY_SIMPLIFICATION_TOLERANCE_PX", 2.0)
    simplified = core._from_objects_and_graph(
        [obj],
        nx.Graph(),
        heightmap,
        source_shape=heightmap.shape,
        inpaint_zeros=False,
        prune_zeros=False,
    )

    assert len(simplified.epicells[0].boundary) < len(detailed.epicells[0].boundary)


def test_boundary_simplification_point_count_is_independent_of_pixel_size():
    mask = single_cell_mask()
    heightmap = np.ones(mask.shape)

    result_px1 = dp.from_heightmap(
        mask,
        heightmap,
        pixel_size=1.0,
        inpaint_zeros=False,
        prune_zeros=False,
    )
    result_px02 = dp.from_heightmap(
        mask,
        heightmap,
        pixel_size=0.2,
        inpaint_zeros=False,
        prune_zeros=False,
    )

    cell1 = result_px1.epicells[0]
    cell02 = result_px02.epicells[0]
    assert len(cell1.boundary) == len(cell02.boundary)
    assert np.isclose(cell02.uncorrected_area, cell1.uncorrected_area * 0.2**2)
    assert np.isclose(cell02.uncorrected_perimeter, cell1.uncorrected_perimeter * 0.2)


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


def test_plots_handle_empty_results():
    result = DeprojResult([], nx.Graph(), source_shape=(10, 10))
    frame = result.to_dataframe()
    plotters = [
        lambda: plot_mask_objects(np.zeros((10, 10)), result),
        lambda: plot_heightmap_with_centers(np.full((10, 10), np.nan), result),
        lambda: plot_feature_map(result),
        lambda: plot_relative_error_map(result),
        lambda: plot_3d_boundaries(result),
        lambda: plot_feature_histograms(frame),
    ]
    for plot in plotters:
        figure, _ = plot()
        assert isinstance(figure, plt.Figure)
        plt.close(figure)


def test_plotters_return_figures_and_axes():
    mask = single_cell_mask()
    heightmap = np.ones(mask.shape)
    result = dp.from_heightmap(mask, heightmap, inpaint_zeros=False, prune_zeros=False)
    frame = result.to_dataframe()
    plotters = [
        lambda: plot_mask_objects(mask, result),
        lambda: plot_heightmap_with_centers(heightmap, result),
        lambda: plot_feature_map(result),
        lambda: plot_relative_error_map(result),
        lambda: plot_3d_boundaries(result),
        lambda: plot_feature_histograms(frame),
    ]
    for plot in plotters:
        figure, axes = plot()
        assert isinstance(figure, plt.Figure)
        assert axes is not None
        plt.close(figure)


def test_plotters_accept_supplied_axes():
    mask = single_cell_mask()
    heightmap = np.ones(mask.shape)
    result = dp.from_heightmap(mask, heightmap, inpaint_zeros=False, prune_zeros=False)

    figure, ax = plt.subplots()
    returned_figure, returned_ax = plot_feature_map(result, "area", ax=ax, colorbar=False)
    assert returned_figure is figure
    assert returned_ax is ax
    plt.close(figure)

    figure, ax = plt.subplots()
    returned_figure, returned_ax = plot_relative_error_map(
        result, "area", ax=ax, colorbar=False
    )
    assert returned_figure is figure
    assert returned_ax is ax
    plt.close(figure)

    figure, ax = plt.subplots()
    returned_figure, returned_ax = plot_heightmap_with_centers(
        heightmap, result, ax=ax, colorbar=False
    )
    assert returned_figure is figure
    assert returned_ax is ax
    plt.close(figure)

    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    returned_figure, returned_ax = plot_3d_boundaries(result, "area", ax=ax, colorbar=False)
    assert returned_figure is figure
    assert returned_ax is ax
    plt.close(figure)


def test_feature_plotters_reject_unknown_features():
    mask = single_cell_mask()
    result = dp.from_heightmap(mask, np.ones(mask.shape), inpaint_zeros=False, prune_zeros=False)
    with pytest.raises(ValueError, match="unknown Epicell feature"):
        plot_feature_map(result, "not_a_feature")
    with pytest.raises(ValueError, match="unknown Epicell feature"):
        plot_3d_boundaries(result, "not_a_feature")
    with pytest.raises(ValueError, match="unknown relative-error metric"):
        plot_relative_error_map(result, "not_a_metric")


def test_histograms_mark_missing_columns():
    mask = single_cell_mask()
    result = dp.from_heightmap(mask, np.ones(mask.shape), inpaint_zeros=False, prune_zeros=False)
    figure, axes = plot_feature_histograms(
        result.to_dataframe(), columns=["area", "missing_column"]
    )
    assert any("Missing column" in text.get_text() for text in axes.flat[1].texts)
    plt.close(figure)


def test_plots_can_be_saved(tmp_path):
    mask = single_cell_mask()
    result = dp.from_heightmap(mask, np.ones(mask.shape), inpaint_zeros=False, prune_zeros=False)
    paths = save_plots(
        tmp_path / "plots",
        mask,
        np.ones(mask.shape),
        result,
        result.to_dataframe(),
        features=("area", "eccentricity"),
    )
    assert all(type(path) is type(tmp_path) for path in paths)
    assert {path.name for path in paths} == {
        "mask_objects.png",
        "heightmap_centers.png",
        "area_map.png",
        "eccentricity_map.png",
        "feature_histograms.png",
        "boundaries_3d.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert not plt.get_fignums()


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
