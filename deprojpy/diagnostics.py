"""Compatibility re-exports for the renamed :mod:`deprojpy.plotting` module.

New code should import plotting helpers from :mod:`deprojpy.plotting`.
"""

from .plotting import (
    plot_3d_boundaries,
    plot_feature_histograms,
    plot_feature_map,
    plot_heightmap_with_centers,
    plot_mask_objects,
    plot_relative_error_map,
    save_plots,
)

__all__ = [
    "plot_3d_boundaries",
    "plot_feature_histograms",
    "plot_feature_map",
    "plot_heightmap_with_centers",
    "plot_mask_objects",
    "plot_relative_error_map",
    "save_plots",
]
