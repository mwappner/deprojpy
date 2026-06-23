from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


def find_deproj_samples() -> Path | None:
    """Find the local sample TIFF files shipped with this repository."""
    samples = Path(__file__).resolve().parents[1] / "samples"
    if (samples / "Segmentation-2.tif").is_file() and (samples / "HeightMap-2.tif").is_file():
        return samples
    return None


@pytest.fixture(scope="session")
def deproj_samples() -> Path:
    samples = find_deproj_samples()
    if samples is None:
        pytest.skip("local DeProj sample TIFF files not found under ./samples")
    return samples
