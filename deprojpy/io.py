from pathlib import Path

import numpy as np
import tifffile


def load_tiff_pair(mask_path: str | Path, heightmap_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a segmentation mask and height map, checking their XY shapes."""
    mask = np.asarray(tifffile.imread(mask_path))
    heightmap = np.asarray(tifffile.imread(heightmap_path))
    if mask.ndim != 2 or heightmap.ndim != 2:
        raise ValueError("mask and heightmap must both be 2-D images")
    if mask.shape != heightmap.shape:
        raise ValueError(f"shape mismatch: mask {mask.shape}, heightmap {heightmap.shape}")
    return mask, heightmap

