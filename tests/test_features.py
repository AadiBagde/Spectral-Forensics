"""Feature extraction consistency tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectral_forensics.config import DEFAULT_FEATURES, ENGINEERED_FEATURE_NAMES
from spectral_forensics.features.extract import analyze_image, extract_feature_vector
from spectral_forensics.features.schema import load_feature_schema, save_feature_schema
from spectral_forensics.preprocessing.loader import ImageLoadError

# Total expected features = radial bins + engineered scalars
EXPECTED_N_FEATURES = DEFAULT_FEATURES.n_radial_bins + len(ENGINEERED_FEATURE_NAMES)


def test_feature_vector_length(tmp_gray_image: Path):
    vector = extract_feature_vector(tmp_gray_image)
    assert vector.shape == (EXPECTED_N_FEATURES,), (
        f"Expected {EXPECTED_N_FEATURES} features "
        f"({DEFAULT_FEATURES.n_radial_bins} radial + {len(ENGINEERED_FEATURE_NAMES)} engineered), "
        f"got {vector.shape[0]}"
    )
    assert np.isfinite(vector).all()


def test_analyze_image_bundle(tmp_rgb_image: Path):
    result = analyze_image(tmp_rgb_image)
    assert result.preprocessed.shape == (256, 256)
    assert result.spectrum.log_power.shape == (256, 256)
    assert result.n_features == EXPECTED_N_FEATURES
    assert len(result.feature_names) == EXPECTED_N_FEATURES
    assert result.feature_names[0] == "freq_0"
    assert result.feature_names[DEFAULT_FEATURES.n_radial_bins - 1] == "freq_127"
    # Check engineered feature names are present at the end
    for eng_name in ENGINEERED_FEATURE_NAMES:
        assert eng_name in result.feature_names, f"Missing engineered feature: {eng_name}"


def test_feature_extraction_repeatable(tmp_gray_image: Path):
    a = extract_feature_vector(tmp_gray_image)
    b = extract_feature_vector(tmp_gray_image)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_corrupted_image_feature_error(tmp_path: Path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not-an-image")
    with pytest.raises(ImageLoadError):
        extract_feature_vector(bad)


def test_feature_schema_roundtrip(tmp_path: Path):
    path = tmp_path / "feature_schema.json"
    save_feature_schema(path)
    data = load_feature_schema(path)
    assert data["feature_config"]["schema_version"] == DEFAULT_FEATURES.schema_version
    assert len(data["feature_names"]) == EXPECTED_N_FEATURES
    # Confirm both radial and engineered names are present
    assert data["feature_names"][0] == "freq_0"
    assert data["feature_names"][-1] == ENGINEERED_FEATURE_NAMES[-1]
