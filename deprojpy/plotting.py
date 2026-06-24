from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .models import DeprojResult, Epicell

CMAP = plt.colormaps.get_cmap("viridis")
EPICELL_FEATURES = {field.name for field in fields(Epicell)}
ERROR_METRICS = {
    "area": ("area_error", "uncorrected_area"),
    "perimeter": ("perimeter_error", "uncorrected_perimeter"),
}


def _normalizer(
    values: np.ndarray, vmin: float | None = None, vmax: float | None = None
) -> Normalize:
    finite = values[np.isfinite(values)]
    low = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    high = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 1.0)
    if low == high:
        high = low + 1.0
    return Normalize(low, high)


def _feature_values(result: DeprojResult, feature: str) -> np.ndarray:
    if feature not in EPICELL_FEATURES:
        known = ", ".join(sorted(EPICELL_FEATURES))
        raise ValueError(f"unknown Epicell feature {feature!r}; choose one of: {known}")
    return np.asarray([float(getattr(cell, feature)) for cell in result.epicells], dtype=float)


def _relative_error_values(result: DeprojResult, metric: str, as_percent: bool) -> np.ndarray:
    if metric not in ERROR_METRICS:
        known = ", ".join(sorted(ERROR_METRICS))
        raise ValueError(f"unknown relative-error metric {metric!r}; choose one of: {known}")
    numerator, denominator = ERROR_METRICS[metric]
    absolute = _feature_values(result, numerator)
    baseline = _feature_values(result, denominator)
    values = np.divide(
        absolute,
        baseline,
        out=np.full_like(absolute, np.nan, dtype=float),
        where=baseline != 0,
    )
    if as_percent:
        values = values * 100.0
    return values


def _parent_figure(ax):
    return ax.figure


def plot_mask_objects(
    mask: np.ndarray,
    result: DeprojResult,
    *,
    ax=None,
    title: str | None = None,
    linewidth: float = 0.5,
    boundary_color=None,
):
    """Plot retained boundaries over the input mask."""
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    ax.imshow(mask, cmap="gray")
    for cell in result.epicells:
        p = cell.boundary / result.pixel_size
        ax.plot(p[:, 0], p[:, 1], linewidth=linewidth, color=boundary_color)
    ax.set_title(title or f"Parsed mask: {len(result.epicells)} retained cells")
    ax.set_xlabel("column / x (pixels)")
    ax.set_ylabel("row / y (pixels)")
    ax.set_aspect("equal")
    return fig, ax


def plot_label_objects(
    labels: np.ndarray,
    result: DeprojResult | None = None,
    *,
    ax=None,
    background: int = 0,
    use_graph_coloring: bool = True,
    K: int = 8,
    seed: int | None = None,
    title: str | None = None,
    show_junctions: bool = True,
    show_cell_boundaries: bool = True,
):
    """Plot a labeled image with optional DeProjPy boundaries and junctions."""
    try:
        import labelimage_tools as lit
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_label_objects requires labelimage-tools. Install it with: "
            "python -m pip install -e "
            "/home/mw/Documents/Pasteur/Code/tissue_processing/labelimage-tools"
        ) from exc
    fig, ax = lit.plot_label_image(
        labels,
        ax=ax,
        background=background,
        use_graph_coloring=use_graph_coloring,
        K=K,
        seed=seed,
        title=title or "Labeled cell image",
    )
    if show_cell_boundaries:
        lit.plot_contours(labels, ax=ax, background=background, color="black", linewidth=0.5)
    if result is not None:
        for cell in result.epicells:
            p = cell.boundary[:, :2] / result.pixel_size
            ax.plot(p[:, 0], p[:, 1], color="white", linewidth=0.6, alpha=0.8)
        if show_junctions:
            centroids = []
            for _, data in result.junction_graph.nodes(data=True):
                if "centroid" in data:
                    centroids.append(np.asarray(data["centroid"][:2]) / result.pixel_size)
            if centroids:
                xy = np.asarray(centroids)
                ax.scatter(xy[:, 0], xy[:, 1], s=12, c="red", edgecolors="white", linewidths=0.4)
    return fig, ax


def plot_heightmap_with_centers(
    heightmap: np.ndarray,
    result: DeprojResult,
    *,
    ax=None,
    cmap="viridis",
    title: str | None = None,
    center_size: float = 3,
    center_color="red",
    colorbar: bool = True,
    colorbar_label: str = "height-map value",
):
    """Plot the input height map and retained cell centers."""
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    image = ax.imshow(heightmap, cmap=cmap)
    centers = np.asarray(
        [c.center[:2] / result.pixel_size for c in result.epicells], dtype=float
    ).reshape(-1, 2)
    if len(centers):
        ax.scatter(centers[:, 0], centers[:, 1], s=center_size, c=center_color)
    if colorbar:
        fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title or "Height map and retained cell centers")
    ax.set_xlabel("column / x (pixels)")
    ax.set_ylabel("row / y (pixels)")
    ax.set_aspect("equal")
    return fig, ax


def plot_feature_map(
    result: DeprojResult,
    feature: str = "area",
    *,
    ax=None,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    alpha: float = 1,
    missing_color="0.7",
):
    """Color cell polygons by a scalar :class:`Epicell` feature."""
    values = _feature_values(result, feature)
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    normalizer = _normalizer(values, vmin, vmax)
    color_map = plt.colormaps.get_cmap(cmap)
    for cell in result.epicells:
        p = cell.boundary
        value = float(getattr(cell, feature))
        color = color_map(normalizer(value)) if np.isfinite(value) else missing_color
        ax.fill(
            p[:, 0],
            p[:, 1],
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_title(title or f"Cell feature: {feature}")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_aspect("equal")

    if colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        fig.colorbar(scalar_map, ax=ax, label=colorbar_label or feature)
    return fig, ax


def _plot_values_as_feature_map(
    result: DeprojResult,
    values: np.ndarray,
    *,
    ax=None,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    alpha: float = 1,
    missing_color="0.7",
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    normalizer = _normalizer(values, vmin, vmax)
    color_map = plt.colormaps.get_cmap(cmap)
    for cell, value in zip(result.epicells, values, strict=True):
        p = cell.boundary
        color = color_map(normalizer(value)) if np.isfinite(value) else missing_color
        ax.fill(
            p[:, 0],
            p[:, 1],
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_title(title or "Cell feature map")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_aspect("equal")

    if colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        fig.colorbar(scalar_map, ax=ax, label=colorbar_label)
    return fig, ax


def plot_relative_error_map(
    result: DeprojResult,
    metric: str = "area",
    *,
    ax=None,
    cmap="coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    alpha: float = 1,
    missing_color="0.7",
    as_percent: bool = True,
):
    """Color cell polygons by relative area or perimeter deprojection error.

    ``metric`` must be ``"area"`` or ``"perimeter"``. Values are computed as
    ``(corrected - projected) / projected`` from the corresponding absolute
    error and uncorrected measurement fields. By default values are shown as
    percentages.
    """
    values = _relative_error_values(result, metric, as_percent=as_percent)
    unit = "%" if as_percent else "fraction"
    return _plot_values_as_feature_map(
        result,
        values,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title or f"Relative {metric} error",
        colorbar=colorbar,
        colorbar_label=colorbar_label or f"relative {metric} error ({unit})",
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        missing_color=missing_color,
    )


def plot_3d_boundaries(
    result: DeprojResult,
    feature: str = "area",
    *,
    ax=None,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    linewidth: float = 0.6,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    missing_color="0.7",
):
    """Plot deprojected cell boundaries in three dimensions."""
    values = _feature_values(result, feature)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = _parent_figure(ax)
        if not hasattr(ax, "get_zlim"):
            raise ValueError("plot_3d_boundaries requires a 3D matplotlib axis")
    normalizer = _normalizer(values, vmin, vmax)
    color_map = plt.colormaps.get_cmap(cmap)
    for cell, value in zip(result.epicells, values, strict=True):
        p = np.vstack([cell.boundary, cell.boundary[0]])
        color = color_map(normalizer(value)) if np.isfinite(value) else missing_color
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=linewidth)
    if colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        fig.colorbar(scalar_map, ax=ax, label=colorbar_label or feature, shrink=0.7)
    ax.set_title(title or f"Deprojected boundaries colored by {feature}")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_zlabel(f"z ({result.units})")  # type: ignore[assignment]
    if result.epicells:
        ax.set_box_aspect((1, 1, 0.5))  # type: ignore[assignment]
    return fig, ax


def _integer_bins(values: np.ndarray) -> np.ndarray | str:
    finite = values[np.isfinite(values)]
    if not finite.size or not np.allclose(finite, np.round(finite)):
        return "auto"
    low = int(np.floor(finite.min()))
    high = int(np.ceil(finite.max()))
    return np.arange(low - 0.5, high + 1.5, 1.0)


def plot_feature_histograms(
    dataframe,
    *,
    columns=("area", "perimeter", "eccentricity", "n_neighbors"),
    bins="auto",
    infer_integer_columns: bool = True,
    title: str | None = "DeProj morphology distributions",
    figsize=None,
):
    """Plot compact histograms of core morphology measurements."""
    columns = tuple(columns)
    count = max(len(columns), 1)
    ncols = min(3, count)
    nrows = int(np.ceil(count / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, column in zip(axes.flat, columns, strict=False):
        ax.set_title(column.replace("_", " "))
        ax.set_ylabel("cell count")
        if column not in dataframe:
            ax.text(0.5, 0.5, "Missing column", ha="center", va="center")
            continue
        values = dataframe[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
            continue
        column_bins = (
            _integer_bins(finite)
            if infer_integer_columns and bins == "auto"
            else bins
        )
        ax.hist(finite, bins=column_bins)
    for ax in axes.flat[len(columns) :]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def save_plots(
    directory,
    mask: np.ndarray | None = None,
    heightmap: np.ndarray | None = None,
    result: DeprojResult | None = None,
    dataframe=None,
    *,
    labels: np.ndarray | None = None,
    features=("area",),
    dpi: int = 150,
    close: bool = True,
) -> list[Path]:
    """Save the standard plot bundle and return written paths."""
    if heightmap is None or result is None:
        raise ValueError("save_plots requires heightmap=... and result=...")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if dataframe is None:
        dataframe = result.to_dataframe()

    plot_specs = [
        ("heightmap_centers.png", lambda: plot_heightmap_with_centers(heightmap, result)),
        ("feature_histograms.png", lambda: plot_feature_histograms(dataframe)),
        ("boundaries_3d.png", lambda: plot_3d_boundaries(result, "area")),
    ]
    if labels is not None:
        plot_specs.insert(0, ("label_objects.png", lambda: plot_label_objects(labels, result)))
    elif mask is not None:
        plot_specs.insert(0, ("mask_objects.png", lambda: plot_mask_objects(mask, result)))
    else:
        raise ValueError("save_plots requires either mask=... or labels=...")
    plot_specs.extend(
        (f"{feature}_map.png", lambda feature=feature: plot_feature_map(result, feature))
        for feature in features
    )

    written = []
    for filename, make_plot in plot_specs:
        figure, _ = make_plot()
        path = directory / filename
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(figure)
        written.append(path)
    return written
