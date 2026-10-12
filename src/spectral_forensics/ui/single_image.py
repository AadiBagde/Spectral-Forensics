"""Single-image analysis dashboard page."""

import tempfile
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from PIL import Image
from spectral_forensics.features.extract import analyze_image
from spectral_forensics.classification.interface import predict_features, ModelHandle
from spectral_forensics.visualization.spectrum import plot_analysis_figure
from spectral_forensics.reporting.pdf_generator import generate_single_image_pdf


def _render_verdict(pred_class: str, pred_score: float, engineered_feats: dict[str, float] | None = None):
    """Render an intuitive, user-friendly forensic verdict card with plain-language interpretation."""
    is_fake = pred_class == "FAKE"
    color = "#ef4444" if is_fake else "#22c55e"
    icon = "🤖" if is_fake else "📷"
    bg = "rgba(239,68,68,0.08)" if is_fake else "rgba(34,197,94,0.08)"
    border = "#ef4444" if is_fake else "#22c55e"
    label = "LIKELY AI-GENERATED (SYNTHETIC)" if is_fake else "LIKELY AUTHENTIC (CAMERA CAPTURE)"

    # Compute a human-friendly confidence estimate from margin score (sigmoid mapping)
    import math
    prob_fake = 1.0 / (1.0 + math.exp(-pred_score * 2.5))
    confidence_pct = prob_fake * 100 if is_fake else (1.0 - prob_fake) * 100
    confidence_pct = max(min(confidence_pct, 99.4), 50.1)

    st.markdown(
        f"""
        <div style="
            background:{bg};
            border:2px solid {border};
            border-radius:12px;
            padding:20px 24px;
            margin-bottom:16px;
        ">
            <div style="font-size:1.6rem;font-weight:800;color:{color};display:flex;align-items:center;gap:10px;">
                <span>{icon}</span> <span>{label}</span>
            </div>
            <div style="margin-top:8px;font-size:1.05rem;color:#e5e7eb;">
                Forensic Confidence: <b>{confidence_pct:.1f}%</b>
            </div>
            <div style="color:#9ca3af;font-size:0.85rem;margin-top:6px;">
                Raw SVM Margin Score: <code>{pred_score:+.4f}</code> (Positive = synthetic indicators; Negative = organic camera noise)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar indicator
    normalized = min(max((pred_score + 2.5) / 5.0, 0.0), 1.0)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.caption("👈 Camera / Authentic Evidence")
    with col_b:
        st.markdown("<p style='text-align:right;font-size:0.8rem;color:#9ca3af;'>AI / Synthetic Evidence 👉</p>", unsafe_allow_html=True)
    st.progress(normalized)


def _render_forensic_explanation(pred_class: str, engineered_feats: dict[str, float]):
    """Explain why the image received this verdict in simple, non-expert terms."""
    is_fake = pred_class == "FAKE"
    hf_energy = engineered_feats.get("hf_energy_ratio", 0.0)
    symmetry = engineered_feats.get("hv_symmetry", 0.0)
    entropy = engineered_feats.get("spectral_entropy", 0.0)

    st.markdown("### 🔍 Forensic Findings Explained")
    
    if is_fake:
        st.markdown(
            """
            > **Why was this flagged as AI-Generated?**
            > 
            > Digital generative models (Diffusion, Midjourney, GANs) construct images using deconvolution or latent diffusion grids. 
            > This process leaves microscopic mathematical artifacts that optical camera sensors never produce:
            """
        )
    else:
        st.markdown(
            """
            > **Why was this judged as Authentic?**
            > 
            > Real photographs captured through an optical glass lens follow natural physical laws of light diffraction. 
            > High-frequency energy decays smoothly from the center outwards without unnatural repeating lattice patterns:
            """
        )

    # Indicator Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        hf_state = "Elevated" if hf_energy > 0.15 else "Normal"
        hf_desc = "Synthetic frequency spikes detected" if hf_energy > 0.15 else "Smooth natural lens roll-off"
        st.metric(
            label="High-Frequency Energy",
            value=f"{hf_energy * 100:.1f}%",
            delta=hf_state,
            delta_color="inverse" if is_fake else "normal",
            help="AI generators often exhibit abnormal energy spikes in high frequencies due to pixel upsampling lattices."
        )
        st.caption(hf_desc)
    with c2:
        sym_desc = "Directional grid alignment" if symmetry > 0.4 else "Natural isotropic dispersion"
        st.metric(
            label="Horizontal/Vertical Symmetry",
            value=f"{symmetry:.2f}",
            delta="Aligned" if symmetry > 0.4 else "Dispersed",
            delta_color="off",
            help="High directional symmetry along Cartesian axes is a common telltale sign of neural network convolutions."
        )
        st.caption(sym_desc)
    with c3:
        st.metric(
            label="Spectral Entropy",
            value=f"{entropy:.2f}",
            help="Measures the disorder/distribution of frequency bands. Synthetic imagery frequently shows clustered peaks."
        )
        st.caption("Distribution complexity")


def render_single_image_page(model: ModelHandle | None):
    st.markdown(
        "Upload any photograph or image to inspect for synthetic AI generation artifacts "
        "using 2D Fast Fourier Transform (FFT) and high-frequency spectral forensic metrics."
    )

    uploaded_file = st.file_uploader(
        "Candidate Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    if uploaded_file is None:
        return

    file_size = uploaded_file.size
    if file_size > 20 * 1024 * 1024:
        st.error("File exceeds 20MB limit.")
        return

    # --- Validate & load the image first, before building any UI columns ---
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.load()
    except Exception:
        st.error("Error decoding image. Please ensure it is a valid image file.")
        return

    with st.spinner("Executing spectral extraction and inference..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / uploaded_file.name
            img.save(tmp_path, format="PNG")

            try:
                analysis_result = analyze_image(tmp_path)
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
                return

            pred_class = "Unknown"
            pred_score = 0.0

            if model is not None:
                try:
                    pred = predict_features(model, analysis_result.feature_vector)
                    pred_class = pred.class_name
                    pred_score = pred.score
                except Exception as e:
                    st.error(f"Inference failed: {e}")

    # Extract named engineered features for simple user explanation
    from spectral_forensics.config import ENGINEERED_FEATURE_NAMES
    eng_dict = {}
    if len(analysis_result.feature_vector) >= 135:
        # Last 7 elements are engineered features
        eng_vals = analysis_result.feature_vector[-7:]
        eng_dict = {name: float(val) for name, val in zip(ENGINEERED_FEATURE_NAMES, eng_vals)}

    # --- Render results ---
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Source Image")
        st.image(
            img,
            caption=f"{uploaded_file.name} ({img.width}×{img.height} {img.mode})",
            use_container_width=True,
        )

    with col2:
        st.subheader("Analysis Results")
        if model is None:
            st.warning("Model offline. Inference unavailable.")
        elif pred_class == "Unknown":
            st.error("Inference failed — check logs above.")
        else:
            _render_verdict(pred_class, pred_score, eng_dict)

        with st.expander("Technical Metadata"):
            st.markdown(f"- **SHA-256:** `{analysis_result.metadata.sha256}`")
            st.markdown(
                f"- **Dimensions:** `{analysis_result.metadata.width}×{analysis_result.metadata.height}`"
            )
            st.markdown(f"- **Color Mode:** `{analysis_result.metadata.mode}`")
            st.markdown(f"- **Feature Vector Length:** `{analysis_result.n_features}`")

    st.divider()
    
    # Plain-language forensic breakdown
    if pred_class in ("REAL", "FAKE") and eng_dict:
        _render_forensic_explanation(pred_class, eng_dict)
        st.divider()

    st.divider()
    st.subheader("1. Primary Spectral Frequency Plots")
    st.markdown(
        "Standard forensic views displaying the **Original Preprocessed Frame**, "
        "**2D Log Power Spectrum**, and **1D Radial Frequency Profile**."
    )

    from spectral_forensics.visualization.spectrum import plot_analysis_figure, plot_detailed_forensic_figure

    with tempfile.TemporaryDirectory() as tmpdir2:
        # Standard Technical Plot (Original, 2D FFT, Radial Profile)
        fig1 = plot_analysis_figure(analysis_result, original=img)
        st.pyplot(fig1)

        fig_path1 = Path(tmpdir2) / "figure_primary.png"
        fig1.savefig(fig_path1, dpi=120, bbox_inches="tight")
        plt.close(fig1)

        # Detailed Forensic Diagnostic Plot (HF Lattice Probe, Axial Slices, Energy Bands)
        st.subheader("2. Detailed Forensic Diagnostic Panels")
        st.markdown(
            "Advanced diagnostic views isolating **High-Frequency Residual Lattices**, "
            "**Horizontal vs. Vertical Convolution Symmetry**, and **Spectral Energy Band Distribution**."
        )
        
        # Calculate confidence for chart styling
        import math
        prob_fake = 1.0 / (1.0 + math.exp(-pred_score * 2.5))
        confidence_pct = prob_fake * 100 if pred_class == "FAKE" else (1.0 - prob_fake) * 100
        confidence_pct = max(min(confidence_pct, 99.4), 50.1)

        fig2 = plot_detailed_forensic_figure(analysis_result, pred_class=pred_class, confidence_pct=confidence_pct)
        st.pyplot(fig2)

        fig_path2 = Path(tmpdir2) / "figure_diagnostic.png"
        fig2.savefig(fig_path2, dpi=120, bbox_inches="tight")
        plt.close(fig2)

        metadata_dict = {
            "sha256": analysis_result.metadata.sha256,
            "width": analysis_result.metadata.width,
            "height": analysis_result.metadata.height,
            "mode": analysis_result.metadata.mode,
        }

        pdf_bytes = generate_single_image_pdf(
            filename=uploaded_file.name,
            metadata=metadata_dict,
            pred_class=pred_class,
            pred_score=pred_score,
            figure_path=str(fig_path1),
        )

    st.download_button(
        label="📄 Download Complete Forensic Dossier (PDF)",
        data=pdf_bytes,
        file_name=f"{uploaded_file.name}_forensic_report.pdf",
        mime="application/pdf",
    )
