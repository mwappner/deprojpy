from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .models import DeprojResult

CMAP = plt.colormaps.get_cmap("viridis")


def _normalizer(values: np.ndarray) -> Normalize:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return Normalize(0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        high = low + 1.0
    return Normalize(low, high)


def _feature_values(result: DeprojResult, feature: str) -> np.ndarray:
    if result.epicells and not hasattr(result.epicells[0], feature):
        raise ValueError(f"unknown Epicell feature: {feature!r}")
    return np.asarray([float(getattr(cell, feature)) for cell in result.epicells], dtype=float)


def plot_mask_objects(mask: np.ndarray, result: DeprojResult):
    """Plot retained boundaries over the input mask."""
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap="gray")
    for cell in result.epicells:
        p = cell.boundary / result.pixel_size
        ax.plot(p[:, 0], p[:, 1], linewidth=0.5)
    ax.set_title(f"Parsed mask: {len(result.epicells)} retained cells")
    ax.set_xlabel("column / x (pixels)")
    ax.set_ylabel("row / y (pixels)")
    ax.set_aspect("equal")
    return fig, ax


def plot_heightmap_with_centers(heightmap: np.ndarray, result: DeprojResult):
    """Plot the input height map and retained cell centers."""
    fig, ax = plt.subplots()
    image = ax.imshow(heightmap, cmap="viridis")
    centers = np.asarray(
        [c.center[:2] / result.pixel_size for c in result.epicells], dtype=float
    ).reshape(-1, 2)
    if len(centers):
        ax.scatter(centers[:, 0], centers[:, 1], s=3, c="red")
    fig.colorbar(image, ax=ax, label="height-map value")
    ax.set_title("Height map and retained cell centers")
    ax.set_xlabel("column / x (pixels)")
    ax.set_ylabel("row / y (pixels)")
    ax.set_aspect("equal")
    return fig, ax


def plot_feature_map(result: DeprojResult, feature: str = "area"):
    """Color cell polygons by a scalar :class:`Epicell` feature."""
    fig, ax = plt.subplots()
    values = _feature_values(result, feature)
    normalizer = _normalizer(values)
    for cell in result.epicells:
        p = cell.boundary
        value = float(getattr(cell, feature))
        color = CMAP(normalizer(value)) if np.isfinite(value) else "0.7"
        ax.fill(
            p[:, 0],
            p[:, 1],
            color=color,
            alpha=0.5,
        )
    scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=CMAP)
    fig.colorbar(scalar_map, ax=ax, label=feature)
    ax.set_title(f"Cell feature: {feature}")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_aspect("equal")
    return fig, ax


def plot_3d_boundaries(result: DeprojResult, feature: str = "area"):
    """Plot deprojected cell boundaries in three dimensions."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    values = _feature_values(result, feature)
    normalizer = _normalizer(values)
    for cell, value in zip(result.epicells, values, strict=True):
        p = np.vstack([cell.boundary, cell.boundary[0]])
        color = CMAP(normalizer(value)) if np.isfinite(value) else "0.7"
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=0.6)
    ax.set_title(f"Deprojected boundaries colored by {feature}")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_zlabel(f"z ({result.units})")
    if result.epicells:
        ax.set_box_aspect((1, 1, 0.5))
    return fig, ax


def plot_feature_histograms(dataframe):
    """Plot compact histograms of core morphology measurements."""
    columns = ["area", "perimeter", "eccentricity", "n_neighbors"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, column in zip(axes.flat, columns, strict=True):
        values = (
            dataframe[column].to_numpy(dtype=float)
            if column in dataframe
            else np.array([], dtype=float)
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            ax.hist(finite)
        else:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        ax.set_title(column.replace("_", " "))
        ax.set_ylabel("cell count")
    fig.suptitle("DeProj morphology distributions")
    fig.tight_layout()
    return fig, axes
