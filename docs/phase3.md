# Phase 3: Dashboard, Batch Analysis, and PDF Reports

## Overview
Phase 3 extends the Spectral Forensics project with a professional Streamlit web dashboard. It allows users to visually interact with the forensic engine developed in Phase 1 and the SVM classifier trained in Phase 2. The interface supports single-image spectral analysis, bulk processing, and downloadable PDF forensic reports.

## Prerequisites
Ensure the environment is set up and the model is trained as per Phase 1 and 2.
Additional dependencies added in Phase 3:
- `streamlit`: Web application framework.
- `reportlab`: PDF generation library.

```bash
pip install -r requirements.txt
# or explicitly:
pip install streamlit reportlab
```

## Running the Application
Launch the Streamlit dashboard from the root of the project:
```bash
streamlit run app.py
```
This will open the application in your default web browser (typically at `http://localhost:8501`).

## Application Features

### 1. Single-Image Analysis
- **Functionality**: Upload a single image (PNG/JPG/JPEG, max 20MB) to analyze its frequency spectrum.
- **Visualizations**: Displays the preprocessed grayscale image, 2D log-power spectrum, and the 1D radial frequency profile.
- **Prediction**: Predicts `REAL` or `FAKE` using the Phase 2 SVM model and displays the raw decision score.
- **Reporting**: Click "Download PDF Forensic Report" to generate a detailed PDF containing the metadata, predictions, visualizations, and methodological limitations.

### 2. Batch Analysis
- **Functionality**: Upload multiple images simultaneously.
- **Processing**: Iterates through each image independently, extracting spectral features and executing inference.
- **Output**: Displays a results table showing processing status, predicted class, and model score for each image.
- **Export**: Results can be downloaded as a CSV or aggregated into a comprehensive Batch PDF Forensic Report.

## Architecture and Inference Flow
The application acts as a presentation layer around the existing Phase 1 and Phase 2 engine. It explicitly **does not** retrain the model or alter the core spectral mathematics.
1. The model is eagerly loaded and cached at startup (`get_classification_model()`).
2. An uploaded image is decoded, validated, and temporarily written to disk.
3. `analyze_image()` (from Phase 1) processes the temporary file, generating the `AnalysisResult` (including the 128 feature vector).
4. `predict_features()` (from Phase 2) processes the vector against the `svm_v1` pipeline.
5. `reportlab` generates static PDF buffers from the assembled strings, dictionaries, and matplotlib figures in memory.

## Limitations & Cautions
Responsible forensic reporting requires transparent limitations:
- **Baseline Performance**: The current model achieved a modest held-out test accuracy of 61.64% (FAKE-class F1: 0.5942).
- **Generalization**: The model has not been proven to generalize reliably to unseen generative AI models. The limited size of the test set implies variance in real-world performance.
- **Perturbations**: Image edits, severe JPEG compression, scaling, and Gaussian blur mathematically alter the high-frequency spectrum. While Phase 2 robustness testing showed resilience under mild perturbations, aggressive alterations *will* compromise predictions.
- **Interpretation**: A model prediction or decision score is **not** definitive proof of origin. It indicates whether the image's spectrum more closely resembles the FAKE or REAL examples in our specific training distribution.

## Tests
Phase 3 introduced UI integration tests in `tests/test_ui.py`.
Execution summary:
- Passed: **21** (including 1 UI test)
- Failed: **0**
- Skipped: **0**
