"""CLI: extract spectral features for train/test splits into CSV."""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from spectral_forensics.config import DEFAULT_FEATURES
from spectral_forensics.features.batch import extract_split_features
from spectral_forensics.features.schema import save_feature_schema
from spectral_forensics.paths import get_project_paths
from spectral_forensics.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract radial spectral features from parquet_images/."
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Which split(s) to process (default: both).",
    )
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=None,
        help="Optional limit used only for smoke tests (not written by default).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    paths = get_project_paths()
    cfg = DEFAULT_FEATURES

    logger.info("Project root: %s", paths.root)
    logger.info(
        "Feature schema %s | image_size=%d | representation=%s",
        cfg.schema_version,
        cfg.preprocess.image_size,
        cfg.spectrum_representation,
    )

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        if args.dry_run_limit is not None:
            logger.warning(
                "--dry-run-limit is reserved for ad-hoc experiments; "
                "full extraction ignores it in this script version."
            )
        df = extract_split_features(split, paths=paths, config=cfg, write_csv=True)
        logger.info("%s rows: %d", split, len(df))

    save_feature_schema(paths.processed / "feature_schema.json", cfg)
    print()
    print("Feature extraction complete.")
    print(f"  {paths.processed / 'train_features.csv'}")
    print(f"  {paths.processed / 'test_features.csv'}")
    print(f"  {paths.processed / 'feature_schema.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
