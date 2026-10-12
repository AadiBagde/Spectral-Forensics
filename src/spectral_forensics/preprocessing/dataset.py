"""Dataset folder validation for parquet_images/.

Reports discrepancies (counts, readability, filename/folder mismatches).
Does not move, rename, delete, or silently correct dataset files.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectral_forensics.config import DEFAULT_FEATURES
from spectral_forensics.paths import ProjectPaths, get_project_paths
from spectral_forensics.preprocessing.loader import ImageLoadError, probe_image
from spectral_forensics.utils.logging import get_logger
from spectral_forensics.utils.validation import ensure_directory, is_image_file

logger = get_logger(__name__)

EXPECTED_SPLITS = ("train", "test")
EXPECTED_CLASSES = ("REAL", "FAKE")
REPORTED_COUNTS = {
    ("train", "REAL"): 489,
    ("train", "FAKE"): 457,
    ("test", "REAL"): 95,
    ("test", "FAKE"): 81,
}


@dataclass
class ClassCount:
    split: str
    class_name: str
    actual: int
    reported: int | None
    matches_reported: bool


@dataclass
class FilenamePrefixIssue:
    """Image whose filename prefix disagrees with its parent class folder."""

    relative_path: str
    folder_class: str
    filename_prefix: str


@dataclass
class DatasetValidationReport:
    """Structured validation outcome for parquet_images/."""

    dataset_root: str
    generated_at: str
    splits_present: list[str]
    class_counts: list[ClassCount] = field(default_factory=list)
    total_images: int = 0
    readable_images: int = 0
    unreadable: list[dict[str, str]] = field(default_factory=list)
    missing_directories: list[str] = field(default_factory=list)
    unexpected_files: list[str] = field(default_factory=list)
    filename_prefix_mismatches: list[FilenamePrefixIssue] = field(
        default_factory=list
    )
    extra_unprefixed_in_real: list[str] = field(default_factory=list)
    duplicate_basenames_across_splits: list[str] = field(default_factory=list)
    modes: dict[str, int] = field(default_factory=dict)
    formats: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _filename_prefix(name: str) -> str | None:
    lower = name.lower()
    if lower.startswith("real_"):
        return "real"
    if lower.startswith("fake_"):
        return "fake"
    return None


def validate_parquet_images(
    paths: ProjectPaths | None = None,
    *,
    write_manifest: bool = True,
) -> DatasetValidationReport:
    """Validate train/test REAL/FAKE layout, readability, and naming consistency."""
    paths = paths or get_project_paths()
    root = paths.parquet_images
    generated_at = datetime.now(timezone.utc).isoformat()

    report = DatasetValidationReport(
        dataset_root=str(root.resolve()),
        generated_at=generated_at,
        splits_present=[],
    )

    if not root.is_dir():
        report.ok = False
        report.notes.append(f"Dataset root does not exist: {root}")
        return report

    manifest_rows: list[dict[str, Any]] = []
    basenames_by_split: dict[str, set[str]] = {s: set() for s in EXPECTED_SPLITS}
    mode_counter: Counter[str] = Counter()
    format_counter: Counter[str] = Counter()

    for split in EXPECTED_SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            report.missing_directories.append(str(split_dir))
            report.ok = False
            continue
        report.splits_present.append(split)

        for class_name in EXPECTED_CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                report.missing_directories.append(str(class_dir))
                report.ok = False
                continue

            files = sorted(p for p in class_dir.iterdir() if p.is_file())
            image_files = [p for p in files if is_image_file(p)]
            non_images = [p for p in files if not is_image_file(p)]
            for path in non_images:
                report.unexpected_files.append(
                    str(path.relative_to(paths.root))
                )

            reported = REPORTED_COUNTS.get((split, class_name))
            actual = len(image_files)
            report.class_counts.append(
                ClassCount(
                    split=split,
                    class_name=class_name,
                    actual=actual,
                    reported=reported,
                    matches_reported=reported is None or actual == reported,
                )
            )
            if reported is not None and actual != reported:
                report.ok = False
                report.notes.append(
                    f"Count mismatch {split}/{class_name}: "
                    f"actual={actual}, reported={reported}"
                )

            for path in image_files:
                report.total_images += 1
                basenames_by_split[split].add(path.name)
                rel = path.relative_to(paths.root).as_posix()

                prefix = _filename_prefix(path.name)
                if prefix == "real" and class_name != "REAL":
                    report.filename_prefix_mismatches.append(
                        FilenamePrefixIssue(rel, class_name, prefix)
                    )
                elif prefix == "fake" and class_name != "FAKE":
                    report.filename_prefix_mismatches.append(
                        FilenamePrefixIssue(rel, class_name, prefix)
                    )
                elif prefix is None and class_name == "REAL":
                    report.extra_unprefixed_in_real.append(rel)

                try:
                    meta = probe_image(path)
                    report.readable_images += 1
                    mode_counter[meta.mode] += 1
                    format_counter[meta.format or "UNKNOWN"] += 1
                    label_value = DEFAULT_FEATURES.label_map[class_name]
                    manifest_rows.append(
                        {
                            "relative_path": rel,
                            "filename": path.name,
                            "split": split,
                            "class": class_name,
                            "label": label_value,
                            "width": meta.width,
                            "height": meta.height,
                            "mode": meta.mode,
                            "format": meta.format,
                            "sha256": meta.sha256,
                            "filename_prefix": prefix or "",
                            "prefix_matches_folder": (
                                (prefix == "real" and class_name == "REAL")
                                or (prefix == "fake" and class_name == "FAKE")
                                or prefix is None
                            ),
                        }
                    )
                except ImageLoadError as exc:
                    report.ok = False
                    report.unreadable.append(
                        {"path": rel, "error": str(exc)}
                    )

    overlap = sorted(
        basenames_by_split["train"] & basenames_by_split["test"]
    )
    report.duplicate_basenames_across_splits = overlap
    report.modes = dict(mode_counter)
    report.formats = dict(format_counter)

    n_mismatch = len(report.filename_prefix_mismatches)
    if n_mismatch:
        report.ok = False
        report.notes.append(
            f"Found {n_mismatch} filename-prefix vs folder-class mismatches. "
            "download.py names REAL files real_*.png and FAKE files fake_*.png; "
            "folder labels used by feature extraction follow the parent folder, "
            "not the filename prefix. Investigate before Phase 2 training."
        )

    if report.extra_unprefixed_in_real:
        report.notes.append(
            f"Found {len(report.extra_unprefixed_in_real)} REAL images without "
            "real_/fake_ prefixes (possible manually added photos). "
            "Provenance is not recorded in parquet extract naming."
        )

    if overlap:
        report.notes.append(
            f"{len(overlap)} basenames appear in both train and test. "
            "This may be sequential naming reuse, not content duplication; "
            "content-level dedup is recommended before training."
        )

    raw_images = paths.raw / "Images"
    if raw_images.is_dir():
        report.notes.append(
            f"Duplicate image tree detected at {raw_images}. "
            "Keep datasets/raw/ source files separate from parquet_images/; "
            "avoid editing the wrong copy."
        )

    if write_manifest:
        ensure_directory(paths.manifests)
        manifest_csv = paths.manifests / "parquet_images_manifest.csv"
        fieldnames = [
            "relative_path",
            "filename",
            "split",
            "class",
            "label",
            "width",
            "height",
            "mode",
            "format",
            "sha256",
            "filename_prefix",
            "prefix_matches_folder",
        ]
        with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        summary_json = paths.manifests / "dataset_validation_report.json"
        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)

        logger.info("Wrote manifest: %s", manifest_csv)
        logger.info("Wrote validation report: %s", summary_json)

    return report
