"""Phase 2: ML Classifier Training, Validation, and Evaluation."""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

import _bootstrap  # noqa: F401
from spectral_forensics.paths import get_project_paths
from spectral_forensics.utils.logging import get_logger, setup_logging
from spectral_forensics.classification.training import get_all_models, validate_features
from spectral_forensics.evaluation.interface import evaluate_predictions

logger = get_logger(__name__)

def plot_cv_comparison(cv_results: dict, output_path: Path):
    """Plot cross-validation F1 scores for each model."""
    models = list(cv_results.keys())
    f1_scores = [np.mean(cv_results[m]["f1"]) for m in models]
    f1_std = [np.std(cv_results[m]["f1"]) for m in models]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, f1_scores, yerr=f1_std, capsize=5, color=['#4c72b0', '#55a868', '#c44e52'])
    plt.title('Cross-Validation F1 Score Comparison (FAKE Class)')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(cm, display_labels, output_path: Path):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=display_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_score, output_path: Path):
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title("ROC Curve (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    setup_logging()
    paths = get_project_paths()
    
    # 1. Load Data
    train_path = paths.processed / "verified_train_features.csv"
    test_path = paths.processed / "verified_test_features.csv"
    
    logger.info("Loading train data from %s", train_path)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Radial bins (freq_0..freq_127) + engineered features (hf_energy_ratio, etc.)
    feature_cols = [c for c in df_train.columns if c.startswith('freq_')]
    from spectral_forensics.config import ENGINEERED_FEATURE_NAMES
    eng_cols = [c for c in ENGINEERED_FEATURE_NAMES if c in df_train.columns]
    feature_cols = feature_cols + eng_cols
    if len(feature_cols) < 128:
        logger.error("Expected at least 128 feature columns, found %d", len(feature_cols))
        return 1
    logger.info("Using %d feature columns (%d radial + %d engineered)",
                len(feature_cols), len(feature_cols) - len(eng_cols), len(eng_cols))
        
    validate_features(df_train, feature_cols)
    validate_features(df_test, feature_cols)
    
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values
    
    # 2. Cross-Validation
    models = get_all_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = {}
    best_model_name = None
    best_f1 = -1
    
    logger.info("Starting cross-validation...")
    for name, pipeline in models.items():
        logger.info("Evaluating %s...", name)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
        cv_results[name] = {metric: scores[f"test_{metric}"].tolist() for metric in scoring}
        
        mean_f1 = np.mean(scores["test_f1"])
        logger.info("%s Mean F1: %.4f (std: %.4f)", name, mean_f1, np.std(scores["test_f1"]))
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = name
            
    logger.info("Selected Best Model: %s with F1=%.4f", best_model_name, best_f1)
    
    # Setup Phase 2 experiment directory
    exp_dir = paths.experiments / "phase2"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot CV comparison
    plot_cv_comparison(cv_results, exp_dir / "cv_model_comparison.png")
    
    # 3. Final Training & Evaluation
    best_pipeline = models[best_model_name]
    logger.info("Retraining %s on full train set...", best_model_name)
    best_pipeline.fit(X_train, y_train)
    
    logger.info("Evaluating on held-out test set...")
    t0 = time.time()
    y_pred = best_pipeline.predict(X_test)
    if hasattr(best_pipeline, "predict_proba"):
        y_score = best_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = best_pipeline.decision_function(X_test)
    inference_time = time.time() - t0
    
    eval_result = evaluate_predictions(y_test, y_pred, y_score)
    
    logger.info("Test Results:")
    logger.info("Accuracy: %.4f", eval_result.metrics["accuracy"])
    logger.info("F1 Score: %.4f", eval_result.metrics["f1_score"])
    logger.info("Inference Time for %d samples: %.4f seconds", len(y_test), inference_time)
    
    # Plot CM & ROC
    plot_confusion_matrix(eval_result.confusion_matrix, ["REAL", "FAKE"], exp_dir / "test_confusion_matrix.png")
    plot_roc_curve(y_test, y_score, exp_dir / "test_roc_curve.png")
    
    # 4. Save Final Results
    final_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "selected_model": best_model_name,
        "cv_results": cv_results,
        "test_metrics": eval_result.metrics,
        "test_classification_report": eval_result.extras["classification_report"],
        "inference_time_seconds": inference_time,
        "n_test_samples": len(y_test)
    }
    
    report_path = exp_dir / "final_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Saved final report to %s", report_path)
    
    # 5. Model Persistence
    model_id = f"{best_model_name.lower()}_v1"
    model_dir = paths.models / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_pipeline, model_dir / "model.joblib")

    from spectral_forensics.config import FEATURE_SCHEMA_VERSION
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({
            "model_id": model_id,
            "algorithm": best_model_name,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "training_date": datetime.utcnow().isoformat(),
            "test_accuracy": eval_result.metrics["accuracy"],
            "test_f1": eval_result.metrics["f1_score"],
            "test_roc_auc": eval_result.metrics.get("roc_auc", None)
        }, f, indent=2)
        
    logger.info("Saved model artifact to %s", model_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
