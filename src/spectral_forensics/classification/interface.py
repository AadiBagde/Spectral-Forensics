"""Classification API for Phase 2 integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import warnings
import joblib

import numpy as np

from spectral_forensics.config import FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class PredictionResult:
    """Classifier output contract."""

    label: int
    class_name: str
    score: float
    model_id: str
    extras: dict[str, Any]


@dataclass
class ModelHandle:
    """Model container filled in Phase 2."""

    model_id: str
    feature_schema_version: str
    artifact_path: Path
    estimator: Any = None
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Warn immediately if the loaded model was trained on a different schema
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            warnings.warn(
                f"Model '{self.model_id}' was trained with feature schema "
                f"{self.feature_schema_version!r} but the current engine "
                f"produces schema {FEATURE_SCHEMA_VERSION!r}. "
                "Predictions may be unreliable — re-extract features and retrain.",
                UserWarning,
                stacklevel=2,
            )


def load_model(path: Path | str) -> ModelHandle:
    """Load a trained model artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    
    # We expect a directory containing model.joblib and metadata.json
    if path.is_dir():
        model_path = path / "model.joblib"
        meta_path = path / "metadata.json"
    else:
        # If it's a file, assume it's the joblib, and meta is alongside
        model_path = path
        meta_path = path.with_name(path.stem + "_metadata.json")

    estimator = joblib.load(model_path)
    
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return ModelHandle(
        model_id=metadata.get("model_id", "unknown"),
        feature_schema_version=metadata.get("feature_schema_version", "unknown"),
        artifact_path=path,
        estimator=estimator,
        feature_names=metadata.get("feature_names", []),
    )


def predict_features(
    model: ModelHandle,
    feature_vector: np.ndarray,
) -> PredictionResult:
    """Predict REAL/FAKE from a feature vector."""
    if model.estimator is None:
        raise ValueError("Model estimator is not loaded.")
        
    # Ensure shape is 2D: (1, n_features)
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
        
    pred = model.estimator.predict(feature_vector)[0]
    
    # Get decision function score if available
    score = 0.0
    if hasattr(model.estimator, "decision_function"):
        score = float(model.estimator.decision_function(feature_vector)[0])
    elif hasattr(model.estimator, "predict_proba"):
        probs = model.estimator.predict_proba(feature_vector)
        # Assuming classes_ are [0, 1] for [REAL, FAKE]
        classes = list(model.estimator.classes_)
        if 1 in classes:
            idx = classes.index(1)
            score = probs[0][idx]
        else:
            score = float(pred) # fallback
    else:
        score = float(pred)
        
    class_name = "FAKE" if pred == 1 else "REAL"
    
    return PredictionResult(
        label=int(pred),
        class_name=class_name,
        score=float(score),
        model_id=model.model_id,
        extras={}
    )
