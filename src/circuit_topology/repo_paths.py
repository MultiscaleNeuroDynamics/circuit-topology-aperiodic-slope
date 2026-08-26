"""Repository and generated-workspace path helpers."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("MICRONS_DATA_ROOT", REPOSITORY_ROOT / "data" / "microns")
).resolve()
WORKSPACE_ROOT = Path(
    os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")
).resolve()
ALLEN_ROOT = WORKSPACE_ROOT / "ALLEN"
FILE_ROOT = ALLEN_ROOT / "file"
SAVE_ROOT = ALLEN_ROOT / "save"
RESULTS_ROOT = ALLEN_ROOT / "results"
MECHANISM_ROOT = REPOSITORY_ROOT / "mechanisms"


def ensure_workspace() -> None:
    """Create generated-data directories without overwriting existing results."""
    for path in (FILE_ROOT, SAVE_ROOT, RESULTS_ROOT):
        path.mkdir(parents=True, exist_ok=True)

