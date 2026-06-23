from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_examples_can_be_called_from_python(tmp_path, deproj_samples):
    repository = Path(__file__).resolve().parents[1]
    run_sample = _load_example(repository / "examples" / "01_run_sample.py")
    gallery = _load_example(repository / "examples" / "02_plot_gallery.py")

    _, frame, csv_path = run_sample.run_sample(
        deproj_samples / "Segmentation-2.tif",
        deproj_samples / "HeightMap-2.tif",
        tmp_path / "run_sample",
    )
    assert csv_path.is_file()
    assert len(frame) == 426

    paths = gallery.make_gallery(
        deproj_samples / "Segmentation-2.tif",
        deproj_samples / "HeightMap-2.tif",
        tmp_path / "gallery",
        dpi=80,
    )
    assert {path.name for path in paths} >= {
        "mask_objects.png",
        "area_map.png",
        "boundaries_3d.png",
        "neighbor_map.png",
        "error_maps_2x2.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
