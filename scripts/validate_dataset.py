"""CLI: validate parquet_images/ structure, readability, and naming consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from spectral_forensics.paths import get_project_paths
from spectral_forensics.preprocessing.dataset import validate_parquet_images
from spectral_forensics.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Spectral Forensics parquet_images dataset."
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write manifest/report files under datasets/manifests/.",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print the full validation report JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    paths = get_project_paths()

    logger.info("Project root: %s", paths.root)
    logger.info("Validating: %s", paths.parquet_images)

    report = validate_parquet_images(
        paths,
        write_manifest=not args.no_manifest,
    )

    print()
    print("=" * 70)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Root: {report.dataset_root}")
    print(f"Total images: {report.total_images}")
    print(f"Readable: {report.readable_images}")
    print(f"Unreadable: {len(report.unreadable)}")
    print()
    print("Class counts:")
    for row in report.class_counts:
        status = "OK" if row.matches_reported else "MISMATCH"
        print(
            f"  {row.split}/{row.class_name}: actual={row.actual} "
            f"reported={row.reported} [{status}]"
        )
    print()
    print(
        f"Filename-prefix vs folder mismatches: "
        f"{len(report.filename_prefix_mismatches)}"
    )
    if report.filename_prefix_mismatches:
        for issue in report.filename_prefix_mismatches[:8]:
            print(
                f"  - {issue.relative_path} "
                f"(folder={issue.folder_class}, prefix={issue.filename_prefix})"
            )
        remaining = len(report.filename_prefix_mismatches) - 8
        if remaining > 0:
            print(f"  ... and {remaining} more")
    print()
    print(f"Extra unprefixed REAL files: {len(report.extra_unprefixed_in_real)}")
    print(
        f"Basenames shared by train and test: "
        f"{len(report.duplicate_basenames_across_splits)}"
    )
    print()
    if report.notes:
        print("Notes:")
        for note in report.notes:
            print(f"  * {note}")
    print()
    print(f"Overall OK flag: {report.ok}")
    print("=" * 70)

    if args.json_stdout:
        print(json.dumps(report.to_dict(), indent=2))

    # Non-zero exit when structural/readability problems exist.
    # Prefix mismatches are reported as failures so Phase 2 does not proceed blindly.
    return 0 if report.ok and not report.unreadable else 1


if __name__ == "__main__":
    sys.exit(main())
