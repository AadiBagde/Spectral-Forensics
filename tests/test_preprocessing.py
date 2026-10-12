"""Preprocessing and image-loading tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spectral_forensics.config import PreprocessConfig
from spectral_forensics.preprocessing.loader import ImageLoadError, load_image, probe_image
from spectral_forensics.preprocessing.transform import preprocess_image


def test_probe_and_load_roundtrip(tmp_gray_image: Path):
    meta = probe_image(tmp_gray_image)
    assert meta.width == 64
    assert meta.height == 64
    assert meta.sha256
    image = load_image(tmp_gray_image)
    assert image.size == (64, 64)


def test_preprocess_shape_and_range(tmp_rgb_image: Path):
    image = load_image(tmp_rgb_image)
    cfg = PreprocessConfig(image_size=256)
    array = preprocess_image(image, cfg)
    assert array.shape == (256, 256)
    assert array.dtype == np.float32
    assert float(array.min()) >= 0.0
    assert float(array.max()) <= 1.0


def test_missing_image_raises(tmp_path: Path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "does_not_exist.png")


def test_corrupted_image_raises(tmp_path: Path):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"this is not a png file")
    with pytest.raises(ImageLoadError):
        probe_image(bad)


def test_unsupported_extension_raises(tmp_path: Path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageLoadError):
        load_image(text_file)


def test_preprocess_deterministic(tmp_gray_image: Path):
    image = load_image(tmp_gray_image)
    a = preprocess_image(image)
    b = preprocess_image(image)
    np.testing.assert_array_equal(a, b)
