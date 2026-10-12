"""Reusable UI components for Streamlit dashboard."""

import streamlit as st
from pathlib import Path
from spectral_forensics.classification.interface import load_model, ModelHandle
from spectral_forensics.paths import get_project_paths
import joblib

@st.cache_resource
def get_classification_model() -> ModelHandle | None:
    """Load and cache the SVM model."""
    paths = get_project_paths()
    model_dir = paths.models / "svm_v1"
    try:
        model = load_model(model_dir)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def display_header(title: str):
    """Display standard header and explanations."""
    st.markdown(f"## {title}")
    st.caption("Detection of AI-Generated Images via 2D Fast Fourier Transform and Spectral Artifact Analysis")

def display_limitations():
    """Display model interpretations and forensic limitations responsibly."""
    st.markdown("---")
    st.caption("**Forensic Limitations & Interpretation**")
    st.caption("""
    - **Model Prediction:** This result is a model prediction based on learned spectral patterns from a specific training set. It is **not** definitive proof of an image's origin.
    - **Score Interpretation:** The 'score' is a decision function estimate representing separation from the learned SVM boundary, **not** a calibrated probability.
    - **Generalization:** This model has not been proven to generalize to unseen image generators. Standard transformations (e.g., resizing, compression) alter spectral signatures and affect predictions.
    """)
