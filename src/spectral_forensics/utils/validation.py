"""Small path and file validation helpers."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    """Return True if *path* has a supported image extension."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def ensure_directory(path: Path) -> Path:
    """Create *path* if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
