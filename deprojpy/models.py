from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

DATAFRAME_COLUMNS = [
    "id",
    "source_label",
    "xc",
    "yc",
    "zc",
    "area",
    "perimeter",
    "euler_alpha",
    "euler_beta",
    "euler_gamma",
    "curv_mean",
    "curv_gauss",
    "curv_k1",
    "curv_k2",
    "ellipse_xc",
    "ellipse_yc",
    "ellipse_zc",
    "ellipse_a",
    "ellipse_b",
    "ellipse_theta",
    "eccentricity",
    "proj_direction",
    "uncorrected_area",
    "uncorrected_perimeter",
    "n_neighbors",
    "area_error",
    "perimeter_error",
]


@dataclass
class Epicell:
    """Measurements and geometry for one deprojected epithelial cell.

    ``boundary`` has shape ``(n, 3)`` and ``center`` has shape ``(3,)`` in
    geometric ``(x, y, z)`` order. Coordinates, lengths, and areas use the
    parent result's physical units, except angles (radians), eccentricity, and
    neighbor counts.
    """

    id: int
    source_label: int | None
    boundary: np.ndarray
    center: np.ndarray
    junction_ids: np.ndarray
    n_neighbors: int
    area: float
    perimeter: float
    euler_angles: np.ndarray
    curvatures: np.ndarray
    ellipse_fit: np.ndarray
    eccentricity: float
    proj_direction: float
    uncorrected_area: float
    uncorrected_perimeter: float
    area_error: float
    perimeter_error: float


@dataclass
class DeprojResult:
    """Collection returned by :func:`deprojpy.from_heightmap`.

    The source image shape is stored in array ``(row, column)`` order, while
    all cell boundaries, centers, and junction centroids are geometric
    ``(x, y, z)`` coordinates. ``pixel_size`` and ``voxel_depth`` record the
    conversions used to express those coordinates in ``units``.
    """

    epicells: list[Epicell]
    junction_graph: nx.Graph
    units: str = "pixels"
    pixel_size: float = 1.0
    voxel_depth: float = 1.0
    source_shape: tuple[int, int] = (0, 0)
    prepared_heightmap: np.ndarray | None = field(default=None, repr=False)

    @property
    def cells(self) -> list[Epicell]:
        return self.epicells

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per cell using the stable morphology column schema.

        Position, length, and area columns use this result's physical
        ``units``; Euler/orientation columns are radians and curvature columns
        use inverse physical units. An empty result still returns all columns.
        """
        rows = []
        for cell in self.epicells:
            rows.append(
                {
                    "id": cell.id,
                    "source_label": cell.source_label,
                    "xc": cell.center[0],
                    "yc": cell.center[1],
                    "zc": cell.center[2],
                    "area": cell.area,
                    "perimeter": cell.perimeter,
                    "euler_alpha": cell.euler_angles[0],
                    "euler_beta": cell.euler_angles[1],
                    "euler_gamma": cell.euler_angles[2],
                    "curv_mean": cell.curvatures[0],
                    "curv_gauss": cell.curvatures[1],
                    "curv_k1": cell.curvatures[2],
                    "curv_k2": cell.curvatures[3],
                    "ellipse_xc": cell.ellipse_fit[0],
                    "ellipse_yc": cell.ellipse_fit[1],
                    "ellipse_zc": cell.ellipse_fit[2],
                    "ellipse_a": cell.ellipse_fit[3],
                    "ellipse_b": cell.ellipse_fit[4],
                    "ellipse_theta": cell.ellipse_fit[5],
                    "eccentricity": cell.eccentricity,
                    "proj_direction": cell.proj_direction,
                    "uncorrected_area": cell.uncorrected_area,
                    "uncorrected_perimeter": cell.uncorrected_perimeter,
                    "n_neighbors": cell.n_neighbors,
                    "area_error": cell.area_error,
                    "perimeter_error": cell.perimeter_error,
                }
            )
        return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)

    def to_csv(self, path: str | Path, **kwargs: object) -> None:
        """Write :meth:`to_dataframe` to CSV.

        ``path`` may be a string or :class:`pathlib.Path`. Additional keyword
        arguments are forwarded to :meth:`pandas.DataFrame.to_csv`; ``index``
        defaults to ``False``.
        """
        kwargs.setdefault("index", False)
        self.to_dataframe().to_csv(path, **kwargs) # type: ignore[no-untyped-call]
