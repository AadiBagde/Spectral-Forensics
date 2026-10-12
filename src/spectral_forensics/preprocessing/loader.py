"""Safe image loading and metadata collection."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from spectral_forensics.utils.validation import is_image_file


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or validated."""


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata recorded for a successfully probed image."""

    path: str
    filename: str
    format: str | None
    mode: str
    width: int
    height: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_image(path: Path | str) -> ImageMetadata:
    """Validate readability and return metadata without returning pixel data."""
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Image not found: {path}")
    if not is_image_file(path):
        raise ImageLoadError(f"Unsupported or non-image file: {path}")

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            metadata = ImageMetadata(
                path=str(path.resolve()),
                filename=path.name,
                format=image.format,
                mode=image.mode,
                width=image.width,
                height=image.height,
                sha256=_sha256_file(path),
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Unreadable image '{path}': {exc}") from exc

    return metadata


def load_image(path: Path | str) -> Image.Image:
    """Load an image, fully decoding pixels.

    Returns a copy so the file handle can be closed immediately.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Image not found: {path}")
    if not is_image_file(path):
        raise ImageLoadError(f"Unsupported or non-image file: {path}")

    try:
        with Image.open(path) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Unreadable image '{path}': {exc}") from exc
