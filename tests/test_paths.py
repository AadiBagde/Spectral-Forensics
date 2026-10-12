"""Path resolution tests."""

from __future__ import annotations

from pathlib import Path

from spectral_forensics.paths import find_project_root, get_project_paths


def test_find_project_root_from_package():
    root = find_project_root()
    assert (root / "PROJECT_OVERVIEW.md").is_file()
    assert (root / "datasets").is_dir()


def test_get_project_paths_absolute(project_paths):
    assert project_paths.root.is_absolute()
    assert project_paths.parquet_images == project_paths.root / "datasets" / "parquet_images"
    assert project_paths.processed == project_paths.root / "datasets" / "processed"


def test_find_project_root_from_nested(tmp_path: Path, project_paths):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    # Walking from an unrelated temp dir should still fall back via package file.
    root = find_project_root(nested)
    assert root == project_paths.root or (root / "PROJECT_OVERVIEW.md").exists()
