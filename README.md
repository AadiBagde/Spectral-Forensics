# 🔬 Spectral Forensics: Detection of AI-Generated Images

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Dashboard-FF4B4B.svg)](https://streamlit.io/)

> **Forensic Frequency-Domain Analysis for Detecting AI-Generated & Synthetic Imagery using 2D Fast Fourier Transform (FFT), Radial Energy Profiling, and Machine Learning.**

---

## 📌 Overview

With the rapid emergence of photorealistic generative models (Diffusion models like Stable Diffusion, Midjourney, DALL-E, and GANs like StyleGAN), spatial-domain forensic checks (pixel patterns, color histograms) often fail.

**Spectral Forensics** inspects images in the **frequency domain**:
- Neural generative models synthesize imagery using convolutional upsampling, transposed convolutions, or latent patch decoders.
- These operations leave **microscopic periodic mathematical artifacts** (checkerboard grids and high-frequency harmonic spikes) that natural camera glass lenses and physical sensors **never** produce in nature.
- By converting images using the **2D Fast Fourier Transform (FFT)**, these invisible mathematical signatures emerge as distinct harmonic energy distributions.

---

## 🚀 Key Features

- **🔬 2D Fast Fourier Transform Engine**: High-speed, reproducible 2D FFT and centered 2D log-power spectrum extraction.
- **📊 135-Dimensional Feature Representation**:
  - **128 Radial Profile Bins**: Azimuthal integration over concentric frequency radii ($r = \sqrt{u^2 + v^2}$).
  - **7 Engineered Forensic Indicators**: High-frequency energy ratio, spectral entropy, spectral centroid, spectral variance, peak harmonic magnitude, peak bin, and Cartesian axial symmetry ($H/V$ correlation).
- **🤖 Machine Learning Classification**: Support Vector Machines (RBF Kernel) & Linear Classifiers with calibrated continuous decision scores and confidence ratings.
- **🖥️ Interactive Streamlit Dashboard**:
  - **Single Image Analysis**: Upload any image for instant classification, confidence rating, and plain-language explanation of findings.
  - **Dual Diagnostic Visualization Panels**:
    - *Panel 1*: Preprocessed frame, 2D Log-Power Spectrum, and 1D Radial Profile.
    - *Panel 2*: High-Frequency Lattice Residual Heatmap, Axial Cross-Section (Grid Symmetry), and 3-Band Energy Distribution Bar Chart.
  - **Batch Analysis**: Process full directories with summary statistics and CSV exports.
  - **Robustness Sandbox**: Live interactive perturbation tests (JPEG compression, Gaussian blur, resizing, noise injection).
- **📄 Automated PDF Dossier**: Instant downloadable forensic report containing metadata, verdict, indicators, and high-resolution spectral figures.
- **🧪 100% Test Coverage**: 25 automated unit, schema, and regression test cases passing in pytest.

---

## 🛠️ Architecture & Pipeline

```
[ Input Image ] 
       │
       ▼
1. Preprocessing & Normalization
   • Grayscale conversion (Luminance channel)
   • Square center-crop / resize (256 × 256)
   • Float32 scaling ([0.0, 1.0])
       │
       ▼
2. 2D Fast Fourier Transform (FFT)
   • 2D Discrete Fourier Transform: F(u,v) = ∑∑ f(x,y) e^(-j 2π (ux/M + vy/N))
   • Center low-frequencies via fftshift
   • 2D Log-Power Spectrum: S(u,v) = log(1 + |F(u,v)|²)
       │
       ▼
3. Feature Engineering (135 Dimensions)
   • 128 Radial Mean Profile Bins
   • 7 Engineered Forensic Scalar Features:
     1. High-Frequency Energy Ratio
     2. Spectral Entropy
     3. Spectral Centroid
     4. Spectral Variance
     5. Peak Harmonic Magnitude
     6. Peak Frequency Bin
     7. Horizontal/Vertical Symmetry
       │
       ▼
4. Machine Learning Classification
   • Standardized feature scaling (StandardScaler)
   • Support Vector Machine (RBF Kernel)
   • Raw Decision Margin Score + Calibrated Forensic Confidence %
       │
       ▼
5. Interactive UI & PDF Reporting
   • Color-coded verdict (AUTHENTIC vs. AI-GENERATED)
   • Non-technical explanation of observed physics vs. synthetic lattices
   • Downloadable ReportLab PDF forensic dossier
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10 to 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AadiBagde/Spectral-Forensics.git
cd Spectral-Forensics
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```
All 25 unit and regression tests will execute.

### 3. Run Command-Line Single Image Analysis
```bash
python scripts/run_fft_analysis.py --image path/to/your/image.jpg
```

---

## 📂 Project Structure

```
Spectral-Forensics/
├── app.py                             # Main Streamlit web application
├── PRESENTATION_GUIDE.md              # Complete presentation & oral defense guide
├── PROJECT_OVERVIEW.md                # Full technical specification & project roadmap
├── README.md                          # Repository documentation & guide
├── pyproject.toml / requirements.txt  # Dependencies & project metadata
├── src/spectral_forensics/            # Core library package
│   ├── config.py                      # Feature dimensions & schema versioning
│   ├── paths.py                       # Absolute path resolution utilities
│   ├── classification/                # Model inference & prediction interfaces
│   ├── features/                      # 2D FFT & 135-feature extraction engine
│   ├── frequency/                     # FFT shift & radial integration math
│   ├── preprocessing/                 # Image loaders & spatial transformations
│   ├── reporting/                     # ReportLab PDF dossier generation
│   ├── ui/                            # Streamlit components, themes & pages
│   └── visualization/                 # Spectral matplotlib plotting engines
├── models/svm_v1/                     # Serialized model artifact & metadata JSON
├── datasets/                          # Dataset manifests & feature CSVs
│   ├── manifests/                     # Image provenance logs & hashes
│   └── processed/                     # Extracted 135-dimensional feature matrices
├── scripts/                           # Batch training, audit & download utilities
└── tests/                             # Pytest automated test suite (25 test cases)
```

---

## 🎯 Forensic Diagnostics & Physical Insights

| Forensic Indicator | Authentic Camera Photography | AI-Generated (Synthetic) Imagery |
|---|---|---|
| **High-Frequency Energy** | Decays smoothly following natural light diffraction | Shows abnormal spikes/elevations due to pixel upsampling lattices |
| **Horizontal/Vertical Symmetry** | Low/isotropic dispersion (physical sensor noise is unbiased) | Strong Cartesian axial alignment from convolutional grid kernels |
| **Radial Frequency Profile** | Monotonically decreasing curve (1/f power law) | Periodic ripples and high-frequency plateaus |
| **Decision Margin** | Negative score (lies on the camera sensor side) | Positive score (lies on the synthetic lattice side) |

> **Important Forensic Notice:**  
> Real-world scenes with tightly repeating physical patterns (e.g. brick walls, window blinds, striped fabrics) exhibit natural spatial frequencies that can mimic convolutional upsampling. Forensic scores represent statistical likelihood and should be evaluated alongside image metadata and contextual provenance.

---

## 📚 Documentation & Presentation Resources

- 📖 **Presentation & Viva Defense Guide**: [`PRESENTATION_GUIDE.md`](./PRESENTATION_GUIDE.md) — Includes slides outline, mathematical talking points, and demo walkthrough.
- 📋 **Project Technical Overview**: [`PROJECT_OVERVIEW.md`](./PROJECT_OVERVIEW.md) — Full technical architecture and phase-by-phase implementation.
- 🔍 **Prediction Quality Audit**: [`experiments/model_improvement/diagnostics/prediction_quality_audit.md`](./experiments/model_improvement/diagnostics/prediction_quality_audit.md) — Verification benchmarks and error analyses.

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
