from __future__ import annotations

import warnings

import numpy as np
from skimage.measure import approximate_polygon


def reduce_boundary(boundary: np.ndarray, tolerance: float = 0.005) -> np.ndarray:
    if len(boundary) < 4:
        return boundary.copy()
    closed = np.vstack([boundary, boundary[0]])
    reduced = approximate_polygon(closed[:, :2], tolerance=tolerance)
    if len(reduced) > 1 and np.allclose(reduced[0], reduced[-1]):
        reduced = reduced[:-1]
    indices = [int(np.argmin(np.sum((boundary[:, :2] - point) ** 2, axis=1))) for point in reduced]
    return boundary[np.asarray(indices)]


def polygon_metrics(boundary: np.ndarray) -> tuple[float, float, float, float]:
    centered = boundary - boundary.mean(axis=0)
    nxt = np.roll(centered, -1, axis=0)
    area3d = 0.5 * np.linalg.norm(np.cross(centered, nxt), axis=1).sum()
    x, y = centered[:, 0], centered[:, 1]
    area2d = 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    perimeter3d = np.linalg.norm(np.roll(centered, -1, axis=0) - centered, axis=1).sum()
    perimeter2d = np.linalg.norm(
        np.roll(centered[:, :2], -1, axis=0) - centered[:, :2], axis=1
    ).sum()
    return float(area3d), float(perimeter3d), float(area2d), float(perimeter2d)


def fit_plane(boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = boundary - boundary.mean(axis=0)
    _, _, rotation = np.linalg.svd(centered, full_matrices=False)
    rotation = rotation.T
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return rotation, rot_to_euler_zxz(rotation)


def rot_to_euler_zxz(r: np.ndarray) -> np.ndarray:
    beta = np.arccos(np.clip(r[2, 2], -1.0, 1.0))
    if abs(np.sin(beta)) > 1e-12:
        alpha = np.arctan2(r[0, 2], -r[1, 2])
        gamma = np.arctan2(r[2, 0], r[2, 1])
    elif r[2, 2] > 0:
        alpha, gamma = np.arctan2(-r[0, 1], r[0, 0]), 0.0
    else:
        alpha, gamma = -np.arctan2(-r[0, 1], r[0, 0]), 0.0
    return np.array([alpha, beta, gamma], dtype=float)


def fit_ellipse_3d(boundary: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit a moment-equivalent ellipse in the local best-fit plane."""
    center = boundary.mean(axis=0)
    local = (boundary - center) @ rotation
    xy = local[:, :2]
    try:
        covariance = np.cov(xy, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        # For points sampled around an ellipse boundary, variance = axis^2 / 2.
        axes = np.sqrt(np.maximum(2.0 * values, 0.0))
        a, b = float(axes[0]), float(axes[1])
        theta = float(np.arctan2(vectors[1, 0], vectors[0, 0]))
        major_local = np.array([np.cos(theta), np.sin(theta), 0.0])
        major_global = major_local @ rotation.T
        direction = float(np.arctan2(major_global[1], major_global[0]))
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2))) if a > 0 else np.nan
        return np.array([*center, a, b, theta]), eccentricity, direction
    except np.linalg.LinAlgError:
        warnings.warn("ellipse fit failed", RuntimeWarning, stacklevel=2)
        return np.full(6, np.nan), np.nan, np.nan
