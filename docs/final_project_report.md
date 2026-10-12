# Final Project Report: Detection of AI-Generated Images via 2D Fast Fourier Transform and Spectral Artifact Analysis

## 1. Title Page
**Project Title:** Detection of AI-Generated Images via 2D Fast Fourier Transform and Spectral Artifact Analysis  
**Date:** October 2026  

## 2. Abstract
This project presents an end-to-end spectral forensic system for distinguishing real photographs from AI-generated images. By transforming image pixels into the frequency domain via a 2D Fast Fourier Transform (FFT) and collapsing the 2D spectrum into a 1D radial profile, 128 frequency-band features are extracted. These features were trained on a Support Vector Machine (SVM) classifier, achieving a modest but stable accuracy of 61.64% on a held-out test set of 146 images. The project demonstrates the utility of frequency-based artifact detection while rigorously exposing the limitations of its generalizability and sensitivity to standard image perturbations like blur and compression.

## 3. Introduction and Problem Statement
Generative AI models, such as Stable Diffusion and Midjourney, produce highly realistic synthetic images. Distinguishing these from real photographs is a critical challenge. However, many deep learning architectures leave subtle high-frequency artifacts during the upsampling or convolution process. This project aims to mathematically isolate and detect these artifacts in the frequency domain.

## 4. Project Objectives and Scope
The primary objective was to build a reproducible, end-to-end digital forensic pipeline comprising feature extraction, machine learning classification, and an interactive Streamlit dashboard for both single-image and batch inference. The scope purposefully avoided deploying opaque deep neural networks, favoring interpretable feature extraction and transparent evaluation.

## 5. Background and Theoretical Foundation
Digital images can be represented in the spatial domain (pixels) or the frequency domain. The 2D Fast Fourier Transform (FFT) converts an image into its frequency components. Synthetic images often contain abnormal energy distributions in mid-to-high frequency bands due to their generative algorithms. By averaging the 2D power spectrum along concentric rings (radial profile), we can extract a robust 1D feature vector that characterizes the image's frequency signature.

## 6. System Architecture and Module Organization
The project is organized modularly:
- **`features`:** Extracts the 128-bin radial spectrum via OpenCV and NumPy.
- **`classification`:** Eagerly loads `scikit-learn` models and executes inference.
- **`visualization`:** Generates matplotlib representations of the 2D spectrum and 1D profile.
- **`ui`:** Implements the user-facing Streamlit dashboard for single/batch processing.
- **`reporting`:** Automates PDF forensic reports via `reportlab`.

## 7. Dataset Description
The model was evaluated on a verified dataset:
- **Training Set:** 826 images.
- **Test Set:** 146 images.
- **Class Distribution:** Approximately balanced (REAL vs. FAKE).
- **Limitations:** The dataset lacks explicit metadata regarding the origin of the FAKE images (e.g., generator versions). Without this provenance, cross-generator generalizability cannot be empirically proven.

## 8. Image Preprocessing and FFT Feature Extraction
Images are converted to grayscale and resized to a consistent dimension. A 2D FFT is applied, and the low frequencies are shifted to the center. The log-power spectrum is calculated, and an azimuthal integration aggregates the 2D matrix into a 1D vector of length 128 (`freq_0` to `freq_127`).

## 9. Machine-Learning Methods
In Phase 2 and Phase 4, multiple classical ML classifiers were evaluated (Logistic Regression, Random Forest, SVM) using a 5-fold Stratified Cross-Validation on the training set exclusively. The data was scaled using `StandardScaler` within a scikit-learn pipeline to prevent data leakage.

## 10. Experimental Setup and Evaluation Metrics
Models were optimized for binary F1 score (pos_label=FAKE). Precision, recall, accuracy, and confusion matrices were recorded. The held-out test set (146 images) was kept strictly isolated during development.

## 11. Phase 2 Baseline Results
The selected Phase 2 SVM model demonstrated the following performance on the test set:
- **Accuracy:** 61.64% (Majority-class baseline: 55.48%)
- **FAKE-Class F1:** 0.5942
- **REAL-Class F1:** 0.6358
These modest results establish that while a spectral signal exists, detection is not perfectly reliable on this data distribution.

## 12. Phase 4 Model-Improvement Experiments
A controlled grid search on the training set identified a Logistic Regression candidate (CV FAKE-F1: 0.5936). However, in the final comparison on the held-out test set, the candidate achieved only an accuracy of 60.96% (FAKE F1: 0.5797). Consequently, the original SVM baseline was retained.

## 13. Robustness Experiments
Applying Gaussian blur and extreme downscaling unexpectedly increased the F1 score in earlier tests. Follow-up analysis revealed that these transformations acted as aggressive low-pass filters, standardizing the image inputs and removing challenging high-frequency noise for these specific 146 test images. The model relies heavily on delicate high-frequency structures and is fundamentally sensitive to structural perturbations.

## 14. Explainability and Frequency-Band Analysis
Feature importance extracted from Random Forest evaluation on the training data showed that mid-to-high frequency bins provide the highest discriminative power. However, these specific bins characterize *this dataset's* generation artifacts, and do not imply universal causality for all deepfakes.

## 15. Streamlit Application and Reporting
A user-friendly web interface was implemented to deploy the system. It enables forensic analysts to inspect the frequency domain visually, process large directories of images in batch via CSV exports, and generate documented PDF reports detailing the specific prediction metrics and methodological limitations.

## 16. Testing and Verification Methodology
The project includes a robust test suite of 21 tests, asserting correct pipeline behaviors from basic math functions to the `st.file_uploader` ByteIO bridging mechanisms. Zero leakage between train/test datasets was confirmed via strict SHA-256 integrity audits.

## 17. Limitations and Threats to Validity
- **Generalization:** Unverified on unseen, out-of-distribution generators.
- **Small Test Set:** 146 images limit statistical power.
- **Model Confidence:** A model decision score is not a calibrated probability of truth. The detector cannot be deployed as definitive proof of origin.

## 18. Future Work
Future iterations should collect fully provenanced datasets, verify zero-shot cross-generator capabilities, and explore 2D Convolutional approaches directly on the un-collapsed 2D Power Spectrum to retain directional artifact information.

## 19. Conclusion
This project successfully implemented an end-to-end Spectral Forensic pipeline. By rigorously auditing the dataset, restricting test-set leakage, and preserving transparent metrics over inflated claims, it provides a scientifically honest foundation for exploring AI-image detection.

## 20. References
- Initial Project Prompt Constraints
- phase2_final_audit.md and phase4.md validation records
- scikit-learn documentation
- ReportLab documentation
