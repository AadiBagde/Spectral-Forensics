"""Stage 5: Final Comparison."""

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score
import joblib

def main():
    root = Path("d:/Projects/MultiMediaForensics/Spectral_Forensics")
    out_dir = root / "experiments" / "phase4" / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_df = pd.read_csv(root / "datasets" / "processed" / "verified_test_features.csv")
    
    feature_cols = [f"freq_{i}" for i in range(128)]
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Evaluate Baseline
    import sys
    sys.path.insert(0, str(root / "src"))
    from spectral_forensics.classification.interface import load_model, predict_features
    baseline_model = load_model(root / "models" / "svm_v1")
    
    y_pred_base = []
    for fv in X_test:
        pred = predict_features(baseline_model, fv)
        y_pred_base.append(1 if pred.class_name == "FAKE" else 0)
        
    base_acc = accuracy_score(y_test, y_pred_base)
    base_prec, base_rec, base_f1, _ = precision_recall_fscore_support(y_test, y_pred_base, average='binary', pos_label=1)
    
    # Evaluate Candidate
    candidate_pipe = joblib.load(root / "models" / "phase4_candidate" / "model.joblib")
    y_pred_cand = candidate_pipe.predict(X_test)
    
    cand_acc = accuracy_score(y_test, y_pred_cand)
    cand_prec, cand_rec, cand_f1, _ = precision_recall_fscore_support(y_test, y_pred_cand, average='binary', pos_label=1)
    
    diff_acc = cand_acc - base_acc
    diff_f1 = cand_f1 - base_f1
    
    report = {
        "baseline": {
            "accuracy": base_acc,
            "fake_f1": base_f1
        },
        "candidate": {
            "accuracy": cand_acc,
            "fake_f1": cand_f1
        },
        "difference": {
            "accuracy_pp": diff_acc,
            "fake_f1_abs": diff_f1
        },
        "conclusion": "Experimental model selected" if cand_f1 > base_f1 else "Retained baseline"
    }
    
    with open(out_dir / "final_comparison.json", "w") as f:
        json.dump(report, f, indent=4)
        
    md = f"""# Final Model Comparison

| Metric | Baseline (SVM v1) | Candidate | Difference |
|---|---|---|---|
| **Accuracy** | {base_acc:.4f} | {cand_acc:.4f} | {diff_acc:+.4f} |
| **FAKE F1** | {base_f1:.4f} | {cand_f1:.4f} | {diff_f1:+.4f} |

**Conclusion:** {report['conclusion']}
"""
    with open(out_dir / "final_comparison.md", "w") as f:
        f.write(md)

    print("Final comparison complete.")

if __name__ == "__main__":
    main()
