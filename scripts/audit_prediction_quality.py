import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score
from spectral_forensics.classification.interface import load_model, predict_features
from spectral_forensics.features.extract import extract_feature_vector

def evaluate_dataset(model, dataset_path: Path, name: str):
    print(f"\nEvaluating dataset: {name} at {dataset_path}")
    
    if not dataset_path.exists():
        print("Dataset path does not exist.")
        return
        
    y_true = []
    y_pred = []
    y_scores = []
    files = []
    
    for class_name, label in [("REAL", 0), ("FAKE", 1)]:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"Directory missing: {class_dir}")
            continue
            
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
                
            try:
                features = extract_feature_vector(img_path)
                pred = predict_features(model, features)
                
                y_true.append(label)
                y_pred.append(pred.label)
                y_scores.append(pred.score)
                files.append((img_path, label, pred.label, pred.score))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    if not y_true:
        print("No valid images found.")
        return
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = float('nan')
        
    print(f"Total Samples: {len(y_true)}")
    print(f"Confusion Matrix (REAL=0, FAKE=1):\n{cm}")
    print(f"FAKE Precision: {precision:.4f}")
    print(f"FAKE Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Analyze errors
    false_positives = [f for f in files if f[1] == 0 and f[2] == 1]
    false_negatives = [f for f in files if f[1] == 1 and f[2] == 0]
    
    print(f"\nFalse Positives (Predicted FAKE, actually REAL): {len(false_positives)}")
    print(f"False Negatives (Predicted REAL, actually FAKE): {len(false_negatives)}")
    
    # Print a few examples
    for err_type, err_list in [("FP", false_positives), ("FN", false_negatives)]:
        if err_list:
            print(f"\nSample {err_type}s:")
            for e in err_list[:5]:
                print(f"  {e[0].name} (Score: {e[3]:.4f})")

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from spectral_forensics.paths import get_project_paths

    _paths = get_project_paths()
    model = load_model(_paths.models / "svm_v1")
    print(f"Loaded model: {model.model_id}")
    
    test_set_path = _paths.datasets / "verified_parquet_images" / "test"
    evaluate_dataset(model, test_set_path, "Existing Test Set (Verified Parquet)")

