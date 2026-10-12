"""Consistent image preprocessing for FFT / feature extraction.

Logic mirrors the working baseline in scripts/std1.py so training and
inference stay aligned.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from spectral_forensics.config import DEFAULT_PREPROCESS, PreprocessConfig


_RESAMPLE_MAP = {
    "lanczos": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
}


def preprocess_image(
    image: Image.Image,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Convert a PIL image to a normalized float32 array for FFT.

    Pipeline:
        colour conversion → resize → float32 → optional [0, 1] normalization
    """
    cfg = config or DEFAULT_PREPROCESS
    resample = _RESAMPLE_MAP.get(cfg.resample.lower())
    if resample is None:
        raise ValueError(
            f"Unsupported resample method '{cfg.resample}'. "
            f"Choose from: {sorted(_RESAMPLE_MAP)}"
        )

    processed = image.convert(cfg.color_mode)
    processed = processed.resize((cfg.image_size, cfg.image_size), resample)
    array = np.asarray(processed, dtype=np.float32)

    if cfg.normalize:
        array = array / float(cfg.pixel_scale)

    return array
