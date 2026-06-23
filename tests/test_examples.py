from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_example_scripts_show_help():
    repository = Path(__file__).resolve().parents[1]
    for script in [
        repository / "examples" / "01_run_sample.py",
        repository / "examples" / "02_plot_gallery.py",
    ]:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout
