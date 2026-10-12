import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score

from spectral_forensics.config import FeatureConfig, PreprocessConfig
from spectral_forensics.features.extract import analyze_image

# --- Custom Preprocessing Functions ---

def custom_preprocess_padded(image: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """Aspect-ratio-preserving resize, pad to square."""
    processed = image.convert(cfg.color_mode)
    processed = ImageOps.pad(processed, (cfg.image_size, cfg.image_size), method=Image.Resampling.LANCZOS, color=0)
    array = np.asarray(processed, dtype=np.float32)
    if cfg.normalize:
        array = array / float(cfg.pixel_scale)
    return array

def custom_preprocess_crop(image: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """Center crop to square, then resize."""
    processed = image.convert(cfg.color_mode)
    processed = ImageOps.fit(processed, (cfg.image_size, cfg.image_size), method=Image.Resampling.LANCZOS)
    array = np.asarray(processed, dtype=np.float32)
    if cfg.normalize:
        array = array / float(cfg.pixel_scale)
    return array

# We need to monkey-patch `preprocess_image` in `spectral_forensics.features.extract` 
# dynamically because `analyze_image` calls `preprocess_image` directly.
import spectral_forensics.preprocessing.transform as transform_module
original_preprocess = transform_module.preprocess_image

def extract_all_features(image_paths, labels, config_name):
    print(f"\nExtracting features for config: {config_name}")
    X = []
    y = []
    
    for path, label in zip(image_paths, labels):
        try:
            res = analyze_image(path)
            X.append(res.feature_vector)
            y.append(label)
        except Exception as e:
            print(f"Error extracting {path}: {e}")
            
    return np.array(X), np.array(y)

def run_experiment():
    train_dir = Path("datasets/raw/Images/train")
    if not train_dir.exists():
        print(f"Train dir {train_dir} not found.")
        return
        
    image_paths = []
    labels = []
    
    for cls_name, lbl in [("REAL", 0), ("FAKE", 1)]:
        for p in (train_dir / cls_name).glob("*.*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image_paths.append(p)
                labels.append(lbl)
                
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(image_paths, labels))
    
    configs = {
        "Base (Stretch)": original_preprocess,
        "Padded (Aspect Ratio Maintained)": custom_preprocess_padded,
        "Center Crop": custom_preprocess_crop
    }
    
    results = []
    feature_distributions = {}
    
    for cfg_name, preprocess_fn in configs.items():
        # Monkey patch correctly in the extract module where it is used
        import spectral_forensics.features.extract as extract_module
        extract_module.preprocess_image = preprocess_fn
        
        X, y = extract_all_features(image_paths, labels, cfg_name)
        feature_distributions[cfg_name] = X
        
        cfg_metrics = {
            "Accuracy": [], "Macro F1": [], "Precision": [], "Recall": [], 
            "Balanced Acc": [], "ROC-AUC": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Phase 2 baseline pipeline
            clf = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
            clf.fit(X_train, y_train)
            
            preds = clf.predict(X_val)
            scores = clf.decision_function(X_val)
            
            cfg_metrics["Accuracy"].append(accuracy_score(y_val, preds))
            cfg_metrics["Macro F1"].append(f1_score(y_val, preds, average="macro"))
            cfg_metrics["Precision"].append(precision_score(y_val, preds))
            cfg_metrics["Recall"].append(recall_score(y_val, preds))
            cfg_metrics["Balanced Acc"].append(balanced_accuracy_score(y_val, preds))
            cfg_metrics["ROC-AUC"].append(roc_auc_score(y_val, scores))
            
        avg_metrics = {k: np.mean(v) for k, v in cfg_metrics.items()}
        print(f"\n--- {cfg_name} Results ---")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")
            
        avg_metrics["Config"] = cfg_name
        results.append(avg_metrics)
        
    # Plot feature distributions
    plt.figure(figsize=(10, 6))
    for cfg_name, X in feature_distributions.items():
        mean_profile = np.mean(X, axis=0)
        plt.plot(mean_profile, label=cfg_name)
    plt.title("Mean Radial Profile per Preprocessing Method")
    plt.xlabel("Frequency Bin (Radius)")
    plt.ylabel("Log1p Power")
    plt.legend()
    plt.grid(True)
    
    out_dir = Path("experiments/model_improvement/resizing_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "feature_distributions.png")
    
    df_results = pd.DataFrame(results).set_index("Config")
    df_results.to_csv(out_dir / "resizing_metrics.csv")
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    run_experiment()
