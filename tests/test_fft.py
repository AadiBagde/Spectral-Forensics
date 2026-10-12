"""FFT engine correctness tests."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_forensics.frequency.fft import compute_fft_spectrum
from spectral_forensics.frequency.radial import radial_mean_profile


def test_constant_image_energy_at_center():
    image = np.ones((64, 64), dtype=np.float32)
    spectrum = compute_fft_spectrum(image)
    cy, cx = 32, 32
    # DC component dominates after fftshift.
    assert spectrum.magnitude[cy, cx] == pytest.approx(spectrum.magnitude.max())
    assert np.isfinite(spectrum.log_power).all()


def test_spectrum_shapes_match_input():
    image = np.random.default_rng(0).random((128, 128), dtype=np.float32)
    spectrum = compute_fft_spectrum(image)
    assert spectrum.magnitude.shape == (128, 128)
    assert spectrum.power.shape == (128, 128)
    assert spectrum.log_magnitude.shape == (128, 128)
    assert spectrum.log_power.shape == (128, 128)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        compute_fft_spectrum(np.ones((8, 8, 3)))


def test_rejects_non_finite():
    image = np.ones((16, 16), dtype=np.float32)
    image[0, 0] = np.nan
    with pytest.raises(ValueError):
        compute_fft_spectrum(image)


def test_horizontal_sinusoid_has_vertical_frequency_peaks():
    """A horizontal cosine should concentrate energy along the ky axis centre column neighbourhood."""
    n = 64
    ys = np.arange(n)
    # Horizontal stripes vary along y => frequency along vertical axis.
    image = np.cos(2 * np.pi * 4 * ys / n).astype(np.float32)
    image = np.tile(image[:, None], (1, n))
    spectrum = compute_fft_spectrum(image)
    mag = spectrum.magnitude
    # Peak off DC should exist.
    mag_no_dc = mag.copy()
    mag_no_dc[n // 2, n // 2] = 0
    peak_y, peak_x = np.unravel_index(np.argmax(mag_no_dc), mag.shape)
    assert abs(peak_x - n // 2) <= 1
    assert abs(peak_y - (n // 2 + 4)) <= 1 or abs(peak_y - (n // 2 - 4)) <= 1


def test_radial_profile_length():
    spectrum = np.ones((256, 256), dtype=np.float64)
    profile = radial_mean_profile(spectrum)
    assert profile.shape == (128,)
    assert np.isfinite(profile).all()
