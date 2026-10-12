"""Batch feature extraction over parquet_images splits."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from spectral_forensics.config import DEFAULT_FEATURES, FeatureConfig
from spectral_forensics.features.extract import analyze_image, features_to_record
from spectral_forensics.features.schema import save_feature_schema
from spectral_forensics.paths import ProjectPaths, get_project_paths
from spectral_forensics.preprocessing.loader import ImageLoadError
from spectral_forensics.utils.logging import get_logger
from spectral_forensics.utils.validation import ensure_directory, is_image_file

logger = get_logger(__name__)


def iter_split_images(
    paths: ProjectPaths,
    split: str,
    class_names: Iterable[str] = ("REAL", "FAKE"),
) -> list[tuple[Path, str, int]]:
    """Return (path, class_name, label) for images in a split."""
    cfg = DEFAULT_FEATURES
    items: list[tuple[Path, str, int]] = []
    for class_name in class_names:
        class_dir = paths.class_dir(split, class_name)
        if not class_dir.is_dir():
            logger.warning("Missing class directory: %s", class_dir)
            continue
        label = cfg.label_map[class_name]
        for path in sorted(class_dir.iterdir()):
            if is_image_file(path):
                items.append((path, class_name, label))
    return items


def extract_split_features(
    split: str,
    paths: ProjectPaths | None = None,
    config: FeatureConfig | None = None,
    *,
    write_csv: bool = True,
) -> pd.DataFrame:
    """Extract features for one dataset split and optionally write CSV."""
    paths = paths or get_project_paths()
    cfg = config or DEFAULT_FEATURES
    records = []
    errors = []

    items = iter_split_images(paths, split)
    logger.info("Extracting features for %s (%d images)", split, len(items))

    for index, (image_path, class_name, label) in enumerate(items, start=1):
        try:
            result = analyze_image(image_path, config=cfg)
            rel = image_path.relative_to(paths.root).as_posix()
            records.append(
                features_to_record(
                    result,
                    label=label,
                    class_name=class_name,
                    relative_path=rel,
                    split=split,
                )
            )
        except (ImageLoadError, ValueError, RuntimeError, OSError) as exc:
            errors.append({"path": str(image_path), "error": str(exc)})
            logger.error("Failed %s: %s", image_path, exc)

        if index % 50 == 0 or index == len(items):
            logger.info("  progress %d/%d", index, len(items))

    dataframe = pd.DataFrame.from_records(records)

    if write_csv:
        ensure_directory(paths.processed)
        output = paths.processed / f"{split}_features.csv"
        dataframe.to_csv(output, index=False)
        schema_path = paths.processed / "feature_schema.json"
        save_feature_schema(
            schema_path,
            cfg,
            extra={
                "source_split": split,
                "n_rows_written": int(len(dataframe)),
                "n_errors": len(errors),
                "errors_sample": errors[:20],
            },
        )
        logger.info("Saved %s (%d rows)", output, len(dataframe))
        logger.info("Saved feature schema %s", schema_path)

    if errors:
        logger.warning("%d images failed during %s extraction", len(errors), split)

    return dataframe
