from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class Epicell:
    id: int
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
        rows = []
        for cell in self.epicells:
            rows.append(
                {
                    "id": cell.id,
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
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path, **kwargs: object) -> None:
        kwargs.setdefault("index", False)
        self.to_dataframe().to_csv(path, **kwargs)

