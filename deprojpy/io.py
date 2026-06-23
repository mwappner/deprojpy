from pathlib import Path

import numpy as np
import tifffile


def load_tiff_pair(
    mask_path: str | Path, heightmap_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load a DeProj-compatible segmentation mask and height map.

    Parameters
    ----------
    mask_path, heightmap_path:
        Paths to two-dimensional TIFF images. Array axes follow image
        convention ``(row, column)``.

    Returns
    -------
    mask, heightmap:
        NumPy arrays with identical shapes. Geometry produced later by
        :func:`deprojpy.from_heightmap` uses explicit ``(x, y, z)`` order.
    """
    mask = np.asarray(tifffile.imread(mask_path))
    heightmap = np.asarray(tifffile.imread(heightmap_path))
    if mask.ndim != 2 or heightmap.ndim != 2:
        raise ValueError("mask and heightmap must both be 2-D images")
    if mask.shape != heightmap.shape:
        raise ValueError(f"shape mismatch: mask {mask.shape}, heightmap {heightmap.shape}")
    return mask, heightmap
