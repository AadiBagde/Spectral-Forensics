"""Streamlit application entry point."""

import json
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from spectral_forensics.ui.components import display_header, display_limitations, get_classification_model
from spectral_forensics.ui.single_image import render_single_image_page
from spectral_forensics.ui.batch import render_batch_analysis_page
from spectral_forensics.ui.theme import inject_custom_css
from spectral_forensics.paths import get_project_paths

# Set page config
st.set_page_config(
    page_title="Spectral Forensics",
    page_icon="🔬",
    layout="wide",
)


def render_robustness_sandbox_page(model):
    """Tab 3 — Interactive Robustness Sandbox."""
    from PIL import Image
    import tempfile
    from pathlib import Path
    from spectral_forensics.preprocessing.perturbations import (
        apply_jpeg_compression, apply_blur, apply_resize, apply_noise
    )
    from spectral_forensics.features.extract import analyze_image
    from spectral_forensics.classification.interface import predict_features
    from spectral_forensics.visualization.spectrum import plot_analysis_figure
    import matplotlib.pyplot as plt

    st.markdown(
        "Upload an image, apply a perturbation, and see how the model prediction "
        "changes. This is a live demonstration — not a formal dataset-level robustness test."
    )

    uploaded_file = st.file_uploader(
        "Image for robustness test", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    if uploaded_file is None:
        return

    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.load()
    except Exception:
        st.error("Invalid image file.")
        return

    st.divider()
    col_ctrl, col_preview = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.subheader("Transformation Controls")
        perturb_type = st.selectbox(
            "Perturbation Type",
            ["None", "JPEG Compression", "Gaussian Blur", "Resize (downscale+upscale)", "Gaussian Noise"],
            key="sb_type",
        )
        perturb_level = None
        perturbed = img.copy()

        if perturb_type == "JPEG Compression":
            perturb_level = st.slider("JPEG Quality", 10, 95, 75, key="sb_jpeg")
            perturbed = apply_jpeg_compression(img, float(perturb_level))
        elif perturb_type == "Gaussian Blur":
            perturb_level = st.slider("Blur Radius", 0.5, 5.0, 1.5, step=0.5, key="sb_blur")
            perturbed = apply_blur(img, float(perturb_level))
        elif perturb_type == "Resize (downscale+upscale)":
            perturb_level = st.slider("Scale Factor", 0.1, 0.9, 0.5, step=0.1, key="sb_resize")
            perturbed = apply_resize(img, float(perturb_level))
        elif perturb_type == "Gaussian Noise":
            perturb_level = st.slider("Noise Std Dev", 5.0, 80.0, 25.0, step=5.0, key="sb_noise")
            perturbed = apply_noise(img, float(perturb_level))

    with col_preview:
        st.subheader("Before vs After")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.image(img, caption="Original", use_container_width=True)
        with pc2:
            st.image(perturbed, caption=f"After: {perturb_type}", use_container_width=True)

    if model is None:
        st.warning("Model offline. Cannot run inference.")
        return

    with st.spinner("Running spectral analysis on both images..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            orig_path = tmp / "original.png"
            img.save(orig_path)
            orig_res = analyze_image(orig_path)
            orig_pred = predict_features(model, orig_res.feature_vector)

            perturbed_path = tmp / "perturbed.png"
            perturbed.save(perturbed_path)
            pert_res = analyze_image(perturbed_path)
            pert_pred = predict_features(model, pert_res.feature_vector)

    st.divider()
    st.subheader("Prediction Comparison")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("Original — Classification", orig_pred.class_name)
        st.metric("Original — Decision Score", f"{orig_pred.score:.4f}")
    with mc2:
        changed = orig_pred.class_name != pert_pred.class_name
        delta_score = pert_pred.score - orig_pred.score
        st.metric(
            "Perturbed — Classification",
            pert_pred.class_name,
            delta="⚡ Label flipped!" if changed else None,
        )
        st.metric("Perturbed — Decision Score", f"{pert_pred.score:.4f}", delta=f"{delta_score:+.4f}")

    st.caption(
        "**Note:** This is a live single-image demonstration. "
        "Formal robustness results across all 146 test images are in `experiments/phase2/robustness/`."
    )


def render_evaluation_page():
    """Tab 4 — Saved Evaluation Metrics from Phase 2."""
    paths = get_project_paths()
    report_path = paths.experiments / "phase2" / "final_evaluation_report.json"
    cm_path = paths.experiments / "phase2" / "test_confusion_matrix.png"
    roc_path = paths.experiments / "phase2" / "test_roc_curve.png"
    cv_path = paths.experiments / "phase2" / "cv_model_comparison.png"

    st.header("Model Evaluation Results")
    st.caption("Metrics saved from the Phase 2 training run. These are real measured results on the held-out test set.")

    if not report_path.exists():
        st.error(f"Evaluation report not found at `{report_path}`. Run `scripts/student2_ml_classifier.py` first.")
        return

    with open(report_path) as f:
        report = json.load(f)

    m = report.get("test_metrics", {})
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
    mc2.metric("F1 Score (FAKE)", f"{m.get('f1_score', 0):.4f}")
    mc3.metric("Precision", f"{m.get('precision', 0):.4f}")
    mc4.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}" if m.get('roc_auc') else "N/A")

    st.divider()
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")
    with col_roc:
        st.subheader("ROC Curve")
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("ROC curve image not found.")

    if cv_path.exists():
        st.subheader("Cross-Validation Model Comparison")
        st.image(str(cv_path), use_container_width=True)

    st.divider()
    with st.expander("Full CV Results (JSON)"):
        st.json(report.get("cv_results", {}))

    st.caption(
        f"**Model:** `{report.get('selected_model', 'N/A')}` | "
        f"**Test Samples:** `{report.get('n_test_samples', 'N/A')}` | "
        f"**Trained:** `{report.get('timestamp', 'N/A')}`"
    )


def render_about_page():
    st.header("Methodology")
    st.write("""
    **Spectral Forensics** provides a scientifically verifiable pipeline to analyze the frequency spectrum of digital images, identifying structural artifacts often introduced by generative AI upsampling algorithms.
    
    ### Pipeline Overview
    1. **Image Ingestion and Preprocessing:** Conversion to grayscale and uniform spatial dimensions (256×256).
    2. **2D Fast Fourier Transform (FFT):** Transformation from spatial to frequency domain.
    3. **Log Power Spectrum Visualization:** Visualizes the magnitude of frequency components.
    4. **Radial Frequency Feature Extraction:** Azimuthal integration collapses the 2D matrix into a 128-bin 1D radial profile.
    5. **Engineered Spectral Features (v1.1.0):** Seven additional scalar features — high-frequency energy ratio, spectral entropy, centroid, variance, peak magnitude/bin, and H/V symmetry — are appended for a 135-element feature vector.
    6. **SVM Classification:** A Support Vector Machine evaluates the feature vector against learned boundaries.
    7. **Report Generation:** Export predictions and parameters for external review.
    
    ### Evaluation Metrics
    Results are displayed in the **Evaluation** tab with the latest trained model metrics.
    
    *Note: These metrics are historical records for this specific training dataset. They do not guarantee performance on individual uploaded images or unseen generator architectures.*
    """)
    display_limitations()


def main():
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("### SPECTRAL FORENSICS")
        st.caption("Detection of AI-Generated Images via 2D Fast Fourier Transform")
        st.divider()

        # Load model
        model = get_classification_model()
        if model is None:
            st.error("Status: ERROR (Model Missing)", icon="❌")
        else:
            st.success(f"Status: ONLINE ({model.model_id})", icon="✅")
            st.caption(f"Schema: `v{model.feature_schema_version}`")

        st.divider()
        page = st.radio(
            "Workstation Modules",
            ["Single Image Analysis", "Batch Analysis", "Robustness Sandbox", "Evaluation", "Methodology"],
        )

    # Main Content Area
    if page == "Single Image Analysis":
        display_header("Single Image Analysis")
        render_single_image_page(model)
        st.divider()
        display_limitations()
    elif page == "Batch Analysis":
        display_header("Batch Analysis")
        render_batch_analysis_page(model)
        st.divider()
        display_limitations()
    elif page == "Robustness Sandbox":
        display_header("Robustness Sandbox")
        render_robustness_sandbox_page(model)
    elif page == "Evaluation":
        render_evaluation_page()
    elif page == "Methodology":
        render_about_page()


if __name__ == "__main__":
    main()
