"""Centralized project paths independent of the current working directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate Spectral_Forensics root by walking upward from *start*.

    Recognition markers: PROJECT_OVERVIEW.md and datasets/ side by side.
    """
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        overview = candidate / "PROJECT_OVERVIEW.md"
        datasets = candidate / "datasets"
        if overview.is_file() and datasets.is_dir():
            return candidate

    # Fallback: package lives at <root>/src/spectral_forensics/
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved absolute paths for the Spectral Forensics project."""

    root: Path

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def raw(self) -> Path:
        return self.datasets / "raw"

    @property
    def parquet_images(self) -> Path:
        return self.datasets / "parquet_images"

    @property
    def processed(self) -> Path:
        return self.datasets / "processed"

    @property
    def manifests(self) -> Path:
        return self.datasets / "manifests"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def spectral_visualizations(self) -> Path:
        return self.reports / "spectral_visualizations"

    def split_dir(self, split: str) -> Path:
        return self.parquet_images / split

    def class_dir(self, split: str, class_name: str) -> Path:
        return self.parquet_images / split / class_name


def get_project_paths(start: Path | None = None) -> ProjectPaths:
    """Return project paths resolved from package location or *start*."""
    return ProjectPaths(root=find_project_root(start))
