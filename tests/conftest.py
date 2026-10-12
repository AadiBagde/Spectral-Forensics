"""Pytest fixtures and import path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectral_forensics.paths import ProjectPaths, get_project_paths


@pytest.fixture(scope="session")
def project_paths() -> ProjectPaths:
    return get_project_paths(ROOT)


@pytest.fixture
def tmp_gray_image(tmp_path: Path) -> Path:
    """Write a small synthetic grayscale PNG."""
    array = np.zeros((64, 64), dtype=np.uint8)
    array[16:48, 16:48] = 200
    path = tmp_path / "synthetic.png"
    Image.fromarray(array, mode="L").save(path)
    return path


@pytest.fixture
def tmp_rgb_image(tmp_path: Path) -> Path:
    array = np.zeros((80, 120, 3), dtype=np.uint8)
    array[:, :, 0] = 40
    array[:, :, 1] = 120
    array[:, :, 2] = 200
    path = tmp_path / "rgb.png"
    Image.fromarray(array, mode="RGB").save(path)
    return path
