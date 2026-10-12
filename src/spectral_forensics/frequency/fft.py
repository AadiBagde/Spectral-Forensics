"""2D FFT spectrum computations.

Ported from the verified baseline in scripts/std1.py and expanded to expose
intermediate representations for visualization and Phase 2/3 reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumBundle:
    """Centered frequency-domain representations for one image."""

    fft_shifted: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray
    log_magnitude: np.ndarray
    log_power: np.ndarray


def compute_fft_spectrum(image_array: np.ndarray) -> SpectrumBundle:
    """Compute centered magnitude/power spectra from a 2D image array.

    Parameters
    ----------
    image_array:
        2D float array (typically preprocessed grayscale image).
    """
    if image_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D image array, got shape {image_array.shape}"
        )
    if not np.isfinite(image_array).all():
        raise ValueError("Image array contains non-finite values")

    fft = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    power = magnitude ** 2
    return SpectrumBundle(
        fft_shifted=fft_shifted,
        magnitude=magnitude,
        power=power,
        log_magnitude=np.log1p(magnitude),
        log_power=np.log1p(power),
    )


def log_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """Return log1p magnitude spectrum."""
    return np.log1p(magnitude)


def log_power(power: np.ndarray) -> np.ndarray:
    """Return log1p power spectrum."""
    return np.log1p(power)
