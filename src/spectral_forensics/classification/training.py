"""Model training definitions for Phase 2."""

from __future__ import annotations

from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

def get_logistic_regression_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'))
    ])

def get_random_forest_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ('classifier', RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced'))
    ])

def get_svm_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=random_state, probability=True, class_weight='balanced'))
    ])

def get_all_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        'LogisticRegression': get_logistic_regression_pipeline(random_state),
        'RandomForest': get_random_forest_pipeline(random_state),
        'SVM': get_svm_pipeline(random_state)
    }

def validate_features(df: pd.DataFrame, feature_cols: list[str]):
    """Validate that features are present, numeric, and finite."""
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    
    if df[feature_cols].isnull().any().any():
        raise ValueError("Missing values found in feature columns.")
        
    if df[feature_cols].isin([np.inf, -np.inf]).any().any():
        raise ValueError("Infinite values found in feature columns.")
