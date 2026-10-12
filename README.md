# Spectral Forensics

Detection of AI-Generated Images via 2D Fast Fourier Transform and Spectral Artifact Analysis.

See `PROJECT_OVERVIEW.md` for the full specification.

## Phase 1 status

The core forensic engine is available as an importable package under `src/spectral_forensics/`.

### Setup

Use the existing virtual environment:

```bash
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If editable install is skipped, scripts add `src/` to `sys.path` automatically.

### Run tests

```bash
.\.venv\Scripts\python.exe -m pytest
```

### Validate dataset

```bash
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

### Analyze one image (no dashboard)

```bash
.\.venv\Scripts\python.exe scripts\run_fft_analysis.py
```

### Extract features for train/test

```bash
.\.venv\Scripts\python.exe scripts\extract_features.py
```

## Phase 2 status

Phase 2 completed the ML evaluation. A trained SVM model is saved in `models/svm_v1`.
See `docs/phase2_final_audit.md` for detailed evaluation results (61.64% test accuracy, 0.5942 FAKE F1) and robustness experiments.

## Phase 3 status

Phase 3 is complete. The project now includes a Streamlit web dashboard for single-image and batch forensic analysis, with downloadable PDF reports.

To run the dashboard:
```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Final Project Status

The Spectral Forensics project is complete. It achieved the following:
- **Engineered** a reproducible 2D FFT extraction pipeline yielding 128 radial frequency features.
- **Trained and evaluated** multiple classifiers, establishing a baseline SVM model (61.64% accuracy, 0.59 F1) on a held-out test set.
- **Audited** data integrity, robustness against image perturbations, and explainability limitations.
- **Deployed** an interactive Streamlit dashboard allowing single-image visualization and batch processing, complete with CSV and PDF forensic reporting.

For full research details, methodology, and limitations, please consult the final reports and documentation:
- **Final Report:** `reports/final_project_report.pdf` (or `docs/final_project_report.md`)
- **Demonstration & Setup Guide:** `docs/final_demo_guide.md`
- **Presentation Outline:** `docs/final_presentation_outline.md`
- **Project Audit Record:** `docs/final_project_audit.md`

All historic evaluations from Phases 1-4 remain archived and documented in the `docs/` and `experiments/` directories.
