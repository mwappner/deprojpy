from __future__ import annotations

import os
from pathlib import Path

import pytest


def find_matlab_samples() -> Path | None:
    """Find optional upstream DeProj samples without assuming one checkout layout."""
    candidates: list[Path] = []
    if configured := os.environ.get("DEPROJ_MATLAB_SAMPLES"):
        candidates.append(Path(configured).expanduser())
    repository = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repository.parent / "DeProj-matlab" / "samples",
            repository.parent / "DeProj" / "samples",
            repository / "samples",
        ]
    )
    for candidate in candidates:
        if (candidate / "Segmentation-2.tif").is_file() and (
            candidate / "HeightMap-2.tif"
        ).is_file():
            return candidate
    return None


@pytest.fixture(scope="session")
def matlab_samples() -> Path:
    samples = find_matlab_samples()
    if samples is None:
        pytest.skip(
            "DeProj MATLAB samples not found; set DEPROJ_MATLAB_SAMPLES or "
            "place them in a common sibling samples directory"
        )
    return samples
