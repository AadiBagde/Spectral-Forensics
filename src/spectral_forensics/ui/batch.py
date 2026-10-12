"""Batch image analysis dashboard page."""

import tempfile
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from spectral_forensics.features.extract import analyze_image
from spectral_forensics.classification.interface import predict_features, ModelHandle
from spectral_forensics.reporting.pdf_generator import generate_batch_pdf

def render_batch_analysis_page(model: ModelHandle | None):
    st.markdown("Upload multiple images to process them sequentially. A results table and exportable forensic reports will be generated.")
    
    if model is None:
        st.error("Model offline. Batch prediction is unavailable.", icon="❌")
        return
        
    uploaded_files = st.file_uploader("Candidate Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files queued.**")
        
        if len(uploaded_files) > 1000:
            st.warning(f"We recommend processing batches of 1000 or fewer for optimal performance.", icon="⚠️")
            
        if st.button("Initialize Batch Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                
                for idx, uf in enumerate(uploaded_files):
                    status_text.caption(f"Processing {uf.name} ({idx+1}/{len(uploaded_files)})...")
                    
                    row = {
                        "filename": uf.name,
                        "status": "Success",
                        "predicted_class": None,
                        "model_score": None
                    }
                    
                    if uf.size > 20 * 1024 * 1024:
                        row["status"] = "Error: File exceeds 20MB limit"
                    else:
                        try:
                            img = Image.open(uf)
                            img.verify()
                            uf.seek(0)
                            img = Image.open(uf)
                            img.load()
                            
                            temp_file = tmp_path / uf.name
                            img.save(temp_file, format="PNG")
                            
                            analysis_result = analyze_image(temp_file)
                            pred = predict_features(model, analysis_result.feature_vector)
                            
                            row["predicted_class"] = pred.class_name
                            row["model_score"] = round(pred.score, 4)
                            
                        except Exception as e:
                            row["status"] = f"Error: {str(e)}"
                            
                    results.append(row)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
            status_text.caption("Batch processing completed.")
            
            df = pd.DataFrame(results)
            
            st.subheader("Batch Results")
            st.dataframe(df, use_container_width=True)
            
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary Statistics")
                success_count = df[df["status"] == "Success"].shape[0]
                error_count = len(df) - success_count
                st.markdown(f"- **Total Processed:** `{len(df)}`")
                st.markdown(f"- **Successful:** `{success_count}`")
                st.markdown(f"- **Errors:** `{error_count}`")

                if success_count > 0:
                    counts = df[df["status"] == "Success"]["predicted_class"].value_counts()
                    fake_c = int(counts.get("FAKE", 0))
                    real_c = int(counts.get("REAL", 0))

                    labels = ["REAL", "FAKE"]
                    if error_count > 0:
                        labels.append("Error")
                    values = [real_c, fake_c]
                    if error_count > 0:
                        values.append(error_count)
                    colors = ["#22c55e", "#ef4444", "#f59e0b"]

                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        marker=dict(colors=colors[:len(labels)]),
                        textfont=dict(size=14),
                    )])
                    fig.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#d1d5db"),
                        showlegend=True,
                        legend=dict(font=dict(color="#d1d5db")),
                        height=260,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                        
            with col2:
                st.subheader("Export Reports")
                st.markdown("Download the results table or generate a detailed PDF report for this batch.")
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name='spectral_forensics_batch.csv',
                    mime='text/csv',
                )
                
                pdf_bytes = generate_batch_pdf(results)
                st.download_button(
                    label="Download Batch Report (PDF)",
                    data=pdf_bytes,
                    file_name='spectral_forensics_batch_report.pdf',
                    mime='application/pdf',
                )
