"""Phase 2 Stage 1: Extract features from verified dataset."""

import argparse
import sys
import json
from datetime import datetime
import pandas as pd

import _bootstrap  # noqa: F401
from spectral_forensics.config import DEFAULT_FEATURES
from spectral_forensics.features.batch import extract_split_features
from spectral_forensics.paths import get_project_paths, ProjectPaths
from spectral_forensics.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

class VerifiedProjectPaths(ProjectPaths):
    @property
    def parquet_images(self):
        return self.datasets / "verified_parquet_images"

def main():
    setup_logging()
    base_paths = get_project_paths()
    paths = VerifiedProjectPaths(root=base_paths.root)
    cfg = DEFAULT_FEATURES

    logger.info("Extracting verified features from: %s", paths.parquet_images)

    # Validate counts before extraction
    expected_counts = {
        "train": {"REAL": 457, "FAKE": 369},
        "test": {"REAL": 81, "FAKE": 65}
    }

    for split, classes in expected_counts.items():
        for cls, expected in classes.items():
            class_dir = paths.class_dir(split, cls)
            if not class_dir.exists():
                logger.error("Missing expected class directory: %s", class_dir)
                return 1
            actual = len(list(class_dir.glob("*.*")))
            if actual != expected:
                logger.error("Expected %d images in %s but found %d", expected, class_dir, actual)
                return 1

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_directory": str(paths.parquet_images),
        "schema_version": cfg.schema_version,
        "results": {}
    }

    total_failures = 0

    for split in ["train", "test"]:
        # Extract features (write_csv=False to avoid overwriting old CSVs and allow custom naming)
        df = extract_split_features(split, paths=paths, config=cfg, write_csv=False)
        
        expected_total = sum(expected_counts[split].values())
        if len(df) != expected_total:
            logger.error("Split %s yielded %d rows, expected %d", split, len(df), expected_total)
            total_failures += (expected_total - len(df))
        
        # Verify feature columns exist and are numeric and finite
        from spectral_forensics.config import ENGINEERED_FEATURE_NAMES
        feature_cols = [f"{cfg.feature_prefix}{i}" for i in range(cfg.n_radial_bins)]
        eng_cols = [c for c in ENGINEERED_FEATURE_NAMES if c in df.columns]
        all_feature_cols = feature_cols + eng_cols
        logger.info("Split %s: %d radial + %d engineered = %d total feature cols",
                    split, len(feature_cols), len(eng_cols), len(all_feature_cols))
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            logger.error("Missing feature columns: %s", missing_cols)
            return 1
            
        nulls = df[all_feature_cols].isnull().sum().sum()
        if nulls > 0:
            logger.error("Found %d null/NaN feature values in %s", nulls, split)
            return 1
            
        inf_mask = df[all_feature_cols].isin([float('inf'), float('-inf')])
        infs = inf_mask.sum().sum()
        if infs > 0:
            logger.error("Found %d infinite feature values in %s", infs, split)
            return 1

        output_path = paths.processed / f"verified_{split}_features.csv"
        df.to_csv(output_path, index=False)
        logger.info("Saved %s with %d rows", output_path, len(df))
        
        summary["results"][split] = {
            "output_path": str(output_path),
            "rows": len(df),
            "classes": df["class"].value_counts().to_dict()
        }

    summary["total_failures"] = total_failures
    
    summary_path = paths.manifests / "verified_features_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Saved extraction summary to %s", summary_path)
    
    if total_failures > 0:
        logger.error("Extraction completed with %d failures", total_failures)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
