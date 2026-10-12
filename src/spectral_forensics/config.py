"""Central configuration for preprocessing and feature extraction.

These settings must stay identical for feature generation, training (Phase 2),
and inference (Phase 3). Changing them requires a new feature schema version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FEATURE_SCHEMA_VERSION = "1.1.0"

# Ordered list of the 7 engineered scalar features appended after the radial bins.
# Changing this list or its order requires a new schema version.
ENGINEERED_FEATURE_NAMES = [
    "hf_energy_ratio",       # fraction of power in the top half of radial bins
    "spectral_entropy",      # Shannon entropy of the normalized radial profile
    "spectral_centroid",     # power-weighted mean frequency bin index
    "spectral_variance",     # power-weighted variance around the centroid
    "peak_magnitude",        # max value in radial profile (DC bin excluded)
    "peak_bin",              # bin index of that peak
    "hv_symmetry",           # Pearson r between horizontal and vertical spectral slices
]


@dataclass(frozen=True)
class PreprocessConfig:
    """Image preprocessing settings used before FFT."""

    image_size: int = 256
    color_mode: str = "L"  # grayscale baseline
    resample: str = "lanczos"
    normalize: bool = True
    pixel_scale: float = 255.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureConfig:
    """Spectral feature extraction settings and schema metadata."""

    schema_version: str = FEATURE_SCHEMA_VERSION
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    spectrum_representation: str = "log1p_power"
    radial_reduction: str = "mean"
    n_radial_bins: int = 128  # image_size // 2 for 256x256
    feature_prefix: str = "freq_"
    engineered_features: bool = True  # append the 7 engineered scalars
    label_map: dict[str, int] = field(
        default_factory=lambda: {"REAL": 0, "FAKE": 1}
    )

    def feature_names(self) -> list[str]:
        radial = [f"{self.feature_prefix}{i}" for i in range(self.n_radial_bins)]
        if self.engineered_features:
            return radial + ENGINEERED_FEATURE_NAMES
        return radial

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


DEFAULT_PREPROCESS = PreprocessConfig()
DEFAULT_FEATURES = FeatureConfig()

