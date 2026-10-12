"""Feature schema persistence for reproducible Phase 2 / Phase 3 inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spectral_forensics.config import DEFAULT_FEATURES, FeatureConfig
from spectral_forensics.utils.validation import ensure_directory


def save_feature_schema(
    path: Path,
    config: FeatureConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write feature configuration JSON used to reproduce extraction."""
    cfg = config or DEFAULT_FEATURES
    payload: dict[str, Any] = {
        "package": "spectral_forensics",
        "feature_config": cfg.to_dict(),
        "feature_names": cfg.feature_names(),
        "description": (
            "Radial mean of log1p(power) spectrum after grayscale resize "
            f"{cfg.preprocess.image_size}x{cfg.preprocess.image_size}."
        ),
    }
    if extra:
        payload["extra"] = extra

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_feature_schema(path: Path) -> dict[str, Any]:
    """Load a previously saved feature schema JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
