"""Phase 2: Robustness testing script."""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import _bootstrap  # noqa: F401
from spectral_forensics.paths import get_project_paths, ProjectPaths
from spectral_forensics.utils.logging import get_logger, setup_logging
from spectral_forensics.features.batch import iter_split_images
from spectral_forensics.features.extract import analyze_image
from spectral_forensics.classification.interface import load_model, predict_features
from spectral_forensics.preprocessing.perturbations import apply_jpeg_compression, apply_resize, apply_blur, apply_noise
from spectral_forensics.evaluation.interface import evaluate_predictions

logger = get_logger(__name__)

class VerifiedProjectPaths(ProjectPaths):
    @property
    def parquet_images(self):
        return self.datasets / "verified_parquet_images"

def get_perturbation_config():
    return {
        "jpeg": {"func": apply_jpeg_compression, "levels": [90, 75, 50]},
        "blur": {"func": apply_blur, "levels": [1.0, 2.0, 3.0]},
        "resize": {"func": apply_resize, "levels": [0.75, 0.5, 0.25]},
        "noise": {"func": apply_noise, "levels": [10.0, 25.0, 50.0]}
    }

def main():
    setup_logging()
    base_paths = get_project_paths()
    paths = VerifiedProjectPaths(root=base_paths.root)
    
    # Locate model
    model_dir = paths.models / "svm_v1"
    if not model_dir.exists():
        logger.error("Model not found at %s. Please run ML classifier training first.", model_dir)
        return 1
        
    logger.info("Loading model from %s", model_dir)
    model = load_model(model_dir)
    
    test_images = iter_split_images(paths, "test")
    if not test_images:
        logger.error("No test images found.")
        return 1
        
    logger.info("Loaded %d test images for robustness testing.", len(test_images))
    
    perturbations = get_perturbation_config()
    
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Iterate over perturbations
        for p_name, p_config in perturbations.items():
            func = p_config["func"]
            levels = p_config["levels"]
            results[p_name] = {}
            
            for level in levels:
                logger.info("Running perturbation: %s at level %s", p_name, level)
                y_true = []
                y_pred = []
                y_score = []
                
                for img_path, class_name, label in test_images:
                    # 1. Load image
                    try:
                        img = Image.open(img_path)
                        img.load()
                    except Exception as e:
                        logger.error("Failed to load %s: %s", img_path, e)
                        continue
                        
                    # 2. Perturb
                    perturbed_img = func(img, level)
                    
                    # 3. Save temporarily to run standard extract API
                    tmp_file = temp_path / f"temp_{img_path.name}"
                    perturbed_img.save(tmp_file, format="PNG")
                    
                    # 4. Extract features
                    try:
                        analysis = analyze_image(tmp_file)
                        features = analysis.feature_vector
                    except Exception as e:
                        logger.error("Extraction failed for perturbed %s: %s", img_path, e)
                        continue
                        
                    # 5. Predict
                    pred = predict_features(model, features)
                    
                    y_true.append(label)
                    y_pred.append(pred.label)
                    y_score.append(pred.score)
                    
                # Evaluate this level
                eval_res = evaluate_predictions(np.array(y_true), np.array(y_pred), np.array(y_score))
                results[p_name][str(level)] = eval_res.metrics
                logger.info("%s (%s) -> F1: %.4f", p_name, level, eval_res.metrics["f1_score"])
                
    # Save & Plot Results
    out_dir = paths.experiments / "phase2" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = out_dir / "robustness_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model.model_id,
            "results": results
        }, f, indent=2)
        
    logger.info("Saved robustness report to %s", report_path)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (p_name, p_res) in enumerate(results.items()):
        ax = axes[idx]
        levels = list(p_res.keys())
        f1_scores = [p_res[l]["f1_score"] for l in levels]
        acc_scores = [p_res[l]["accuracy"] for l in levels]
        
        x = np.arange(len(levels))
        width = 0.35
        
        ax.bar(x - width/2, f1_scores, width, label='F1 Score')
        ax.bar(x + width/2, acc_scores, width, label='Accuracy')
        
        ax.set_title(f"Robustness: {p_name.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylim(0, 1.0)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_plots.png")
    plt.close()
    
    logger.info("Robustness testing complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
