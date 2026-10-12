"""Extract features from the expanded 6,000 GenImage dataset and retrain models."""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

import _bootstrap  # noqa: F401
from spectral_forensics.paths import get_project_paths
from spectral_forensics.config import DEFAULT_FEATURES, ENGINEERED_FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from spectral_forensics.features.extract import analyze_image, features_to_record
from spectral_forensics.classification.training import get_all_models, validate_features
from spectral_forensics.evaluation.interface import evaluate_predictions
from spectral_forensics.utils.logging import get_logger, setup_logging
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = get_logger(__name__)

def extract_genimage_split(split_name: str, split_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Extract 135 features from a GenImage split folder (real/ and ai/)."""
    if out_csv.exists() and out_csv.stat().st_size > 1000:
        logger.info("Found existing features at %s, loading...", out_csv)
        return pd.read_csv(out_csv)

    class_mapping = {
        "real": ("REAL", 0),
        "ai": ("FAKE", 1)
    }

    records = []
    cfg = DEFAULT_FEATURES
    paths = get_project_paths()

    images_to_process = []
    for folder_name, (class_name, label) in class_mapping.items():
        folder_path = split_dir / folder_name
        if not folder_path.is_dir():
            continue
        for img_path in sorted(folder_path.glob("*.png")):
            images_to_process.append((img_path, class_name, label))

    logger.info("Extracting %d images for split '%s'...", len(images_to_process), split_name)
    t0 = time.time()
    
    for idx, (img_path, class_name, label) in enumerate(images_to_process, 1):
        try:
            result = analyze_image(img_path, config=cfg)
            rel = img_path.relative_to(paths.root).as_posix()
            records.append(
                features_to_record(
                    result,
                    label=label,
                    class_name=class_name,
                    relative_path=rel,
                    split=split_name,
                )
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path.name, exc)

        if idx % 200 == 0 or idx == len(images_to_process):
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info("  Processed %d/%d (%.1f img/s)", idx, len(images_to_process), rate)

    df = pd.DataFrame.from_records(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Saved %s with %d rows and %d columns", out_csv, len(df), len(df.columns))
    return df

def main():
    setup_logging()
    paths = get_project_paths()
    genimage_dir = paths.datasets / "raw" / "genimage"

    train_dir = genimage_dir / "train"
    test_dir = genimage_dir / "test"

    train_csv = paths.processed / "genimage_train_features.csv"
    test_csv = paths.processed / "genimage_test_features.csv"

    # 1. Feature Extraction
    logger.info("=== STEP 1: Feature Extraction on 6,000 Images ===")
    df_train = extract_genimage_split("train", train_dir, train_csv)
    df_test = extract_genimage_split("test", test_dir, test_csv)

    # 2. Select Features (128 radial + 7 engineered = 135)
    feature_cols = [c for c in df_train.columns if c.startswith('freq_')]
    eng_cols = [c for c in ENGINEERED_FEATURE_NAMES if c in df_train.columns]
    feature_cols = feature_cols + eng_cols

    logger.info("Using %d total features (%d radial + %d engineered)",
                len(feature_cols), len(feature_cols) - len(eng_cols), len(eng_cols))

    validate_features(df_train, feature_cols)
    validate_features(df_test, feature_cols)

    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values

    # 3. Model Training & Cross-Validation
    logger.info("=== STEP 2: Model Training & 5-Fold Cross Validation ===")
    models = get_all_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    best_name = None
    best_f1 = -1

    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        mean_acc = np.mean(scores["test_accuracy"])
        mean_f1 = np.mean(scores["test_f1"])
        mean_auc = np.mean(scores["test_roc_auc"])
        logger.info("%-18s | Acc: %.4f | F1: %.4f | ROC-AUC: %.4f", name, mean_acc, mean_f1, mean_auc)

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_name = name

    logger.info("Best Performer: %s with CV F1=%.4f", best_name, best_f1)

    # 4. Fit Best Model on Full 4,000 Training Samples
    best_pipeline = models[best_name]
    logger.info("Fitting %s on full 4,000 sample training set...", best_name)
    best_pipeline.fit(X_train, y_train)

    # 5. Evaluate on 2,000 Unseen Test Set
    logger.info("=== STEP 3: Evaluation on 2,000 Unseen Test Images ===")
    y_pred = best_pipeline.predict(X_test)
    if hasattr(best_pipeline, "decision_function"):
        y_score = best_pipeline.decision_function(X_test)
    else:
        y_score = best_pipeline.predict_proba(X_test)[:, 1]

    eval_result = evaluate_predictions(y_test, y_pred, y_score)
    metrics = eval_result.metrics
    logger.info("TEST ACCURACY:     %.4f", metrics["accuracy"])
    logger.info("TEST F1 SCORE:     %.4f", metrics["f1_score"])
    logger.info("TEST ROC-AUC:      %.4f", metrics.get("roc_auc", float('nan')))
    logger.info("CONFUSION MATRIX:  \n%s", eval_result.confusion_matrix)

    # 6. Save Model Artifact to svm_v1
    model_dir = paths.models / "svm_v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_dir / "model.joblib")

    import json
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({
            "model_id": "svm_v1",
            "algorithm": best_name,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "training_dataset": "GenImage (ADM/ImageNet) 6,000 subset",
            "training_date": datetime.now(timezone.utc).isoformat(),
            "test_accuracy": metrics["accuracy"],
            "test_f1": metrics["f1_score"],
            "test_roc_auc": metrics.get("roc_auc", None),
            "n_train_samples": len(y_train),
            "n_test_samples": len(y_test),
        }, f, indent=2)

    logger.info("SUCCESS: Updated %s with retrained 6,000-image model!", model_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
