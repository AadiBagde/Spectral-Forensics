"""End-to-end spectral feature extraction for a single image.

This is the Phase 1 public engine API used by scripts, tests, and later by
Phase 2 training and Phase 3 inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spectral_forensics.config import DEFAULT_FEATURES, ENGINEERED_FEATURE_NAMES, FeatureConfig
from spectral_forensics.frequency.fft import SpectrumBundle, compute_fft_spectrum
from spectral_forensics.frequency.radial import radial_mean_profile
from spectral_forensics.preprocessing.loader import ImageMetadata, load_image, probe_image
from spectral_forensics.preprocessing.transform import preprocess_image


@dataclass
class AnalysisResult:
    """Complete forensic-engine output for one image (no classifier)."""

    metadata: ImageMetadata
    preprocessed: np.ndarray
    spectrum: SpectrumBundle
    radial_profile: np.ndarray
    feature_vector: np.ndarray
    feature_names: list[str]
    config: FeatureConfig
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.feature_vector.shape[0])


def compute_engineered_features(
    radial_profile: np.ndarray,
    spectrum_2d: np.ndarray,
) -> np.ndarray:
    """Compute the 7 engineered spectral scalar features (PROJECT_OVERVIEW §6.4).

    Parameters
    ----------
    radial_profile:
        1D radial mean log-power profile (128 bins for 256x256 images).
    spectrum_2d:
        2D centered log-power spectrum array (HxW).

    Returns
    -------
    np.ndarray of shape (7,) in the order defined by ENGINEERED_FEATURE_NAMES:
        hf_energy_ratio, spectral_entropy, spectral_centroid, spectral_variance,
        peak_magnitude, peak_bin, hv_symmetry.
    """
    eps = 1e-8
    n = len(radial_profile)
    radii = np.arange(n, dtype=np.float64)

    total_power = radial_profile.sum() + eps

    # 1. High-frequency energy ratio: fraction of power in upper half of bins
    hf_energy_ratio = float(radial_profile[n // 2 :].sum() / total_power)

    # 2. Spectral entropy: Shannon entropy of the normalized radial profile
    p = radial_profile / total_power
    p = np.clip(p, eps, None)
    spectral_entropy = float(-np.sum(p * np.log(p)))

    # 3. Spectral centroid: power-weighted mean frequency bin
    spectral_centroid = float(np.sum(radii * radial_profile) / total_power)

    # 4. Spectral variance: power-weighted variance around the centroid
    spectral_variance = float(
        np.sum(((radii - spectral_centroid) ** 2) * radial_profile) / total_power
    )

    # 5 & 6. Peak magnitude and bin (DC component at bin 0 excluded)
    profile_no_dc = radial_profile[1:] if n > 1 else radial_profile
    peak_magnitude = float(profile_no_dc.max())
    peak_bin = int(np.argmax(profile_no_dc)) + 1  # +1 to account for DC skip

    # 7. H/V symmetry: Pearson correlation between horizontal and vertical
    #    spectral slices through the spectrum centre.
    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2
    h_slice = spectrum_2d[cy, :].astype(np.float64)
    v_slice = spectrum_2d[:, cx].astype(np.float64)
    min_len = min(len(h_slice), len(v_slice))
    h_slice = h_slice[:min_len]
    v_slice = v_slice[:min_len]
    h_std = h_slice.std()
    v_std = v_slice.std()
    if h_std < eps or v_std < eps:
        hv_symmetry = 0.0
    else:
        hv_symmetry = float(np.corrcoef(h_slice, v_slice)[0, 1])
    if not np.isfinite(hv_symmetry):
        hv_symmetry = 0.0

    return np.array(
        [
            hf_energy_ratio,
            spectral_entropy,
            spectral_centroid,
            spectral_variance,
            peak_magnitude,
            float(peak_bin),
            hv_symmetry,
        ],
        dtype=np.float64,
    )


def extract_feature_vector(
    image_path: Path | str,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Return the fixed-length spectral feature vector for *image_path*."""
    return analyze_image(image_path, config=config).feature_vector


def analyze_image(
    image_path: Path | str,
    config: FeatureConfig | None = None,
) -> AnalysisResult:
    """Run preprocessing, FFT, radial analysis, and feature extraction."""
    cfg = config or DEFAULT_FEATURES
    path = Path(image_path)

    metadata = probe_image(path)
    image = load_image(path)
    preprocessed = preprocess_image(image, cfg.preprocess)
    spectrum = compute_fft_spectrum(preprocessed)

    if cfg.spectrum_representation == "log1p_power":
        source_spectrum = spectrum.log_power
    elif cfg.spectrum_representation == "log1p_magnitude":
        source_spectrum = spectrum.log_magnitude
    elif cfg.spectrum_representation == "power":
        source_spectrum = spectrum.power
    elif cfg.spectrum_representation == "magnitude":
        source_spectrum = spectrum.magnitude
    else:
        raise ValueError(
            f"Unsupported spectrum_representation: {cfg.spectrum_representation}"
        )

    if cfg.radial_reduction != "mean":
        raise ValueError(f"Unsupported radial_reduction: {cfg.radial_reduction}")

    radial = radial_mean_profile(source_spectrum)
    expected = cfg.preprocess.image_size // 2
    if radial.shape[0] != expected:
        raise RuntimeError(
            f"Radial profile length {radial.shape[0]} != expected {expected}"
        )

    radial_names = [f"{cfg.feature_prefix}{i}" for i in range(radial.shape[0])]

    if cfg.engineered_features:
        eng_vec = compute_engineered_features(radial, source_spectrum)
        feature_vector = np.concatenate([radial, eng_vec]).astype(np.float64)
        feature_names = radial_names + list(ENGINEERED_FEATURE_NAMES)
    else:
        feature_vector = radial.astype(np.float64, copy=True)
        feature_names = radial_names

    return AnalysisResult(
        metadata=metadata,
        preprocessed=preprocessed,
        spectrum=spectrum,
        radial_profile=radial,
        feature_vector=feature_vector,
        feature_names=feature_names,
        config=cfg,
    )


def features_to_record(
    result: AnalysisResult,
    *,
    label: int | None = None,
    class_name: str | None = None,
    relative_path: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Convert an AnalysisResult into a flat CSV/dict record."""
    record: dict[str, Any] = {
        "image": result.metadata.filename,
        "sha256": result.metadata.sha256,
        "width_orig": result.metadata.width,
        "height_orig": result.metadata.height,
        "mode_orig": result.metadata.mode,
    }
    if relative_path is not None:
        record["relative_path"] = relative_path
    if split is not None:
        record["split"] = split
    if label is not None:
        record["label"] = label
    if class_name is not None:
        record["class"] = class_name

    for name, value in zip(result.feature_names, result.feature_vector):
        record[name] = float(value)
    return record
