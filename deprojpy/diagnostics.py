from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .models import DeprojResult

cmap = plt.colormaps.get_cmap("viridis")

def plot_mask_objects(mask: np.ndarray, result: DeprojResult):
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap="gray")
    for cell in result.epicells:
        p = cell.boundary / result.pixel_size
        ax.plot(p[:, 0], p[:, 1], linewidth=0.5)
    ax.set_title(f"{len(result.epicells)} cells")
    return fig, ax


def plot_heightmap_with_centers(heightmap: np.ndarray, result: DeprojResult):
    fig, ax = plt.subplots()
    image = ax.imshow(heightmap, cmap="viridis")
    centers = np.array([c.center[:2] / result.pixel_size for c in result.epicells])
    if len(centers):
        ax.scatter(centers[:, 0], centers[:, 1], s=3, c="red")
    fig.colorbar(image, ax=ax)
    return fig, ax


def plot_feature_map(result: DeprojResult, feature: str = "area"):
    fig, ax = plt.subplots()
    values = np.array([float(getattr(c, feature)) for c in result.epicells])
    normalizer = (
        Normalize(np.nanmin(values), np.nanmax(values))
        if len(values)
        else Normalize()
    )
    for cell in result.epicells:
        p = cell.boundary
        ax.fill(
            p[:, 0],
            p[:, 1],
            color=cmap(normalizer(float(getattr(cell, feature)))),
            alpha=0.5,
        )
    ax.set_aspect("equal")
    return fig, ax


def plot_3d_boundaries(result: DeprojResult, feature: str = "area"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    values = np.array([float(getattr(c, feature)) for c in result.epicells])
    normalizer = Normalize(np.nanmin(values), np.nanmax(values)) if len(values) else Normalize()
    for cell, value in zip(result.epicells, values):
        p = np.vstack([cell.boundary, cell.boundary[0]])
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=cmap(normalizer(value)), linewidth=0.6)
    return fig, ax


def plot_feature_histograms(dataframe):
    columns = ["area", "perimeter", "eccentricity", "n_neighbors"]
    axes = dataframe[columns].hist(figsize=(8, 6))
    return axes.ravel()[0].figure, axes
