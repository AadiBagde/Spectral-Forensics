"""Stage 4: Controlled Model Improvement."""

import json
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    root = Path("d:/Projects/MultiMediaForensics/Spectral_Forensics")
    out_dir = root / "experiments" / "phase4" / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_csv(root / "datasets" / "processed" / "verified_train_features.csv")
    
    feature_cols = [f"freq_{i}" for i in range(128)]
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values # 1=FAKE, 0=REAL
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We evaluate SVC, RF, LR
    # We want to optimize for F1 macro or FAKE F1
    scoring = 'f1' # Binary F1 for pos_label=1 (FAKE)
    
    print("Running GridSearchCV for SVM...")
    svm_param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto', 0.1, 1],
        'classifier__kernel': ['rbf', 'poly']
    }
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42, class_weight='balanced'))
    ])
    svm_grid = GridSearchCV(svm_pipe, svm_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    svm_grid.fit(X_train, y_train)
    
    print("Running GridSearchCV for Random Forest...")
    rf_param_grid = {
        'classifier__n_estimators': [100, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_leaf': [1, 5]
    }
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    
    print("Running GridSearchCV for Logistic Regression...")
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2']
    }
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    
    best_models = {
        "SVM": (svm_grid.best_score_, svm_grid.best_params_, svm_grid.best_estimator_),
        "RandomForest": (rf_grid.best_score_, rf_grid.best_params_, rf_grid.best_estimator_),
        "LogisticRegression": (lr_grid.best_score_, lr_grid.best_params_, lr_grid.best_estimator_),
    }
    
    best_name = max(best_models, key=lambda k: best_models[k][0])
    best_score, best_params, best_estimator = best_models[best_name]
    
    print(f"Best Model: {best_name} with CV FAKE-F1 = {best_score:.4f}")
    print(f"Params: {best_params}")
    
    # Save selection details
    report = {
        "scoring_metric": "FAKE-class F1 (binary)",
        "cv_folds": 5,
        "candidates": {
            "SVM": {"best_score": svm_grid.best_score_, "best_params": svm_grid.best_params_},
            "RandomForest": {"best_score": rf_grid.best_score_, "best_params": rf_grid.best_params_},
            "LogisticRegression": {"best_score": lr_grid.best_score_, "best_params": lr_grid.best_params_}
        },
        "selected_model": best_name
    }
    
    with open(out_dir / "model_selection_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Save the experimental candidate model
    candidate_dir = root / "models" / "phase4_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, candidate_dir / "model.joblib")
    with open(candidate_dir / "metadata.json", "w") as f:
        json.dump({
            "model_type": best_name,
            "params": best_params,
            "cv_f1": best_score,
            "feature_schema": "128 ordered features (freq_0 to freq_127)"
        }, f, indent=4)
        
    print("Experimental model saved to models/phase4_candidate/")

if __name__ == "__main__":
    main()
