import matplotlib.pyplot as plt
import numpy as np

import deprojpy as dp
from deprojpy.diagnostics import (
    plot_3d_boundaries,
    plot_feature_histograms,
    plot_feature_map,
    plot_heightmap_with_centers,
    plot_mask_objects,
)


def test_sample_end_to_end(matlab_samples):
    mask, heightmap = dp.load_tiff_pair(
        matlab_samples / "Segmentation-2.tif",
        matlab_samples / "HeightMap-2.tif",
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
    frame = result.to_dataframe()
    assert mask.shape == (282, 508)
    assert len(result.epicells) == 426
    assert len(frame) == 426
    assert {
        "id",
        "area",
        "perimeter",
        "curv_mean",
        "ellipse_a",
        "n_neighbors",
    } <= set(frame)
    assert np.isfinite(frame["area"]).all() and (frame["area"] > 0).all()
    assert np.isfinite(frame["perimeter"]).all() and (frame["perimeter"] > 0).all()
    plotters = [
        lambda: plot_mask_objects(mask, result),
        lambda: plot_heightmap_with_centers(heightmap, result),
        lambda: plot_feature_map(result),
        lambda: plot_3d_boundaries(result),
        lambda: plot_feature_histograms(frame),
    ]
    for plot in plotters:
        figure, _ = plot()
        plt.close(figure)
