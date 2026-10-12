"""Classification interfaces (Phase 2).

Phase 1 only defines the contract so the dashboard and training code can
share one loading/prediction API later.
"""

from spectral_forensics.classification.interface import (
    ModelHandle,
    PredictionResult,
    load_model,
    predict_features,
)

__all__ = [
    "ModelHandle",
    "PredictionResult",
    "load_model",
    "predict_features",
]
