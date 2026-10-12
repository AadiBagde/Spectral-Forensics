# Spectral Forensics Demonstration Guide

## 1. Environment Setup

The project supports clean installation via Python `venv`.

```bash
# Activate existing virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install requirements (streamlit and reportlab are included)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
*(Validation Note: Clean installation on Python 3.12 was successfully verified during Phase 5 auditing.)*

## 2. Launching the Dashboard

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```
This will start the local server, typically accessible at `http://localhost:8501`.

## 3. Recommended Single-Image Demonstration
1. Navigate to the **"Single Image"** tab on the sidebar.
2. Click **"Browse files"** and select a verified AI-generated test image (e.g., from `datasets/verified_parquet_images/test/FAKE/`). For a real image, use `datasets/verified_parquet_images/test/REAL/`.
3. Observe the system process the 2D FFT, followed by the generation of the power spectrum and radial profile visualizations.
4. Point out the **Model Decision Score**. Explain that it reflects spectral density against the learned SVM hyperplane, rather than a perfect probability of truth.
5. Click **"Download PDF Forensic Report"** and open the resulting PDF. Showcase the embedded metadata, metrics, and explicitly stated methodological limitations.

## 4. Recommended Batch Demonstration
1. Navigate to the **"Batch Analysis"** tab on the sidebar.
2. Select multiple images (mix of REAL and FAKE) and upload them.
3. Observe the live iteration.
4. **Validation Check:** The batch analyzer gracefully catches errors. If you manually insert an unsupported or corrupted file, the status will show "Error", but the batch loop will continue.
5. Highlight the **"Download Results as CSV"** functionality for data exports.
6. Generate and open the **Batch PDF Report**. Highlight the summarized table of predictions without overwhelming visualization bloat.

## 5. Expected Output Locations
- Downloaded CSVs and PDFs will save to the user's default browser `Downloads` directory.
- Cached models are stored safely in memory by Streamlit during runtime.
- No files are permanently altered or overwritten in the raw dataset directories.

## 6. Known Limitations
- The model's baseline FAKE F1 score is 0.5942; false positives and false negatives will occur during the demo. 
- Generalization to completely novel AI generators has not been verified due to a lack of provenance metadata in the dataset.
- Aggressively resizing or blurring images before upload will heavily alter their high-frequency signature and likely change the prediction.
