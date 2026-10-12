"""Evaluation API for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

@dataclass
class EvaluationResult:
    """Container for classification metrics."""

    metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> EvaluationResult:
    """Compute classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass # e.g. only one class in y_true
            
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return EvaluationResult(
        metrics=metrics,
        confusion_matrix=cm,
        extras={"classification_report": report}
    )
