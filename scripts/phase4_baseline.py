"""Stage 1: Establish a Reproducible Baseline."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score

def main():
    root = Path("d:/Projects/MultiMediaForensics/Spectral_Forensics")
    out_dir = root / "experiments" / "phase4" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Phase 2 test features
    test_df = pd.read_csv(root / "datasets" / "processed" / "verified_test_features.csv")
    train_df = pd.read_csv(root / "datasets" / "processed" / "verified_train_features.csv")
    
    # Load model and predict (via inference API)
    import sys
    sys.path.insert(0, str(root / "src"))
    from spectral_forensics.classification.interface import load_model, predict_features
    
    model = load_model(root / "models" / "svm_v1")
    
    # Features are 'freq_0' to 'freq_127'
    feature_cols = [f"freq_{i}" for i in range(128)]
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values # FAKE is 1, REAL is 0
    
    # We can get predictions directly
    y_pred = []
    y_scores = []
    for fv in X_test:
        pred = predict_features(model, fv)
        y_pred.append(1 if pred.class_name == "FAKE" else 0)
        y_scores.append(pred.score)
        
    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Binary FAKE F1
    prec_fake, rec_fake, f1_fake, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
    
    # Macro and Weighted
    macro_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro')[2]
    weighted_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')[2]
    
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except ValueError:
        roc_auc = None

    record = {
        "model_artifact": "models/svm_v1",
        "feature_schema": "128 ordered features (freq_0 to freq_127)",
        "datasets": {
            "train_records": len(train_df),
            "test_records": len(test_df)
        },
        "metrics": {
            "accuracy": accuracy,
            "real_class": {
                "precision": prec[0], "recall": rec[0], "f1": f1[0]
            },
            "fake_class": {
                "precision": prec[1], "recall": rec[1], "f1": f1[1]
            },
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }
    }
    
    with open(out_dir / "baseline_record.json", "w") as f:
        json.dump(record, f, indent=4)
        
    md_content = f"""# Phase 4 Baseline Record

**Model:** {record["model_artifact"]}
**Feature Schema:** {record["feature_schema"]}
**Training Records:** {record['datasets']['train_records']}
**Test Records:** {record['datasets']['test_records']}

## Evaluation Metrics (Held-out Test Set)
- **Accuracy:** {accuracy:.4f}
- **FAKE-Class F1:** {f1_fake:.4f} (Precision: {prec[1]:.4f}, Recall: {rec[1]:.4f})
- **REAL-Class F1:** {f1[0]:.4f} (Precision: {prec[0]:.4f}, Recall: {rec[0]:.4f})
- **Macro F1:** {macro_f1:.4f}
- **Weighted F1:** {weighted_f1:.4f}
- **ROC-AUC:** {roc_auc if roc_auc else 'N/A'}

### Confusion Matrix
| | Predicted REAL (0) | Predicted FAKE (1) |
|---|---|---|
| **Actual REAL (0)** | {cm[0][0]} | {cm[0][1]} |
| **Actual FAKE (1)** | {cm[1][0]} | {cm[1][1]} |

*This baseline is preserved as the standard for Phase 4 model improvements.*
"""
    with open(out_dir / "baseline_record.md", "w") as f:
        f.write(md_content)
        
    print("Baseline established successfully.")

if __name__ == "__main__":
    main()
