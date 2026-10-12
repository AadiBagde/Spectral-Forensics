# Resizing and Preprocessing Controlled Experiment

**Project:** Detection of AI-Generated Images via 2D Fast Fourier Transform and Spectral Artifact Analysis  
**Date:** October 10, 2026

## 1. Goal
To conduct a controlled experiment assessing whether the current image preprocessing (which stretches images to a 256x256 square, ignoring aspect ratio) degrades spectral artifacts compared to alternative aspect-ratio-preserving methods.

## 2. Current Preprocessing Implementation
- **Target Dimensions:** `256 x 256`
- **Aspect-Ratio Handling:** Ignored (images are stretched to fit the target square).
- **Interpolation:** Lanczos.
- **Color Conversion:** Grayscale (`"L"` mode).
- **FFT Normalization:** Pixel values normalized by dividing by `255.0` prior to FFT.
- **Distortion:** Yes. Because original images drastically vary in size (e.g., `400x400` to `2812x1568`), non-square images are significantly squashed or stretched, altering spatial frequencies along the compressed axis.

## 3. Experimental Setup
We established a strict, reproducible experiment using **only the 826 training images**, leaving the held-out test set completely untouched to preserve integrity.

Three configurations were tested using a 5-fold Stratified Cross-Validation loop. A new SVM model (identical hyperparameter pipeline as Phase 2) was trained from scratch for each configuration to guarantee features from different configurations never mixed.

### Configurations Evaluated:
1. **Base (Stretch):** The existing Phase 1/Phase 2 baseline.
2. **Padded (Aspect Ratio Maintained):** Images were resized via Lanczos to a maximum dimension of 256, then zero-padded (black borders) to form a `256x256` square.
3. **Center Crop:** Images were center-cropped to the largest possible square, then downsampled via Lanczos to `256x256`.

## 4. Results

The metrics below represent the cross-validation averages across all 5 folds:

| Metric | Base (Stretch) | Padded (Aspect Ratio) | Center Crop |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.6258** | 0.5909 | 0.6100 |
| **Macro F1** | **0.6245** | 0.5886 | 0.6091 |
| **Precision (FAKE)**| **0.6149** | 0.5810 | 0.5911 |
| **Recall (FAKE)** | 0.6081 | 0.5626 | **0.6172** |
| **Balanced Acc** | **0.6250** | 0.5901 | 0.6102 |
| **ROC-AUC** | 0.6620 | 0.6452 | **0.6666** |

## 5. Analysis
Contrary to the hypothesis that geometric stretching ruins spatial frequencies, the **Base (Stretch)** preprocessing yielded the highest overall performance (Accuracy, Macro F1, and Precision). 
- **Why did Padding fail?** Zero-padding introduces sharp, artificial vertical/horizontal edges at the image borders. When passed through a 2D FFT, these edges create massive cross-shaped high-frequency artifacts spanning the Fourier space. Since we take a radial average, these artificial artifacts heavily pollute the feature bins, masking the true generative model artifacts.
- **Why did Center Cropping fail?** Center cropping discards peripheral pixels. If generative artifacts (such as boundary inconsistencies, repeating textures, or upsampling noise) are distributed across the entire image, discarding up to 50% of the pixels removes critical signal.

## 6. Tests Added
We added `tests/test_resizing.py` to ensure:
- Preprocessing rigidly enforces the `256x256` dimensions.
- The preprocessing function is completely deterministic (no random cropping or jitter).
- Grayscale conversion guarantees a 2D array output.

## 7. Final Recommendations
1. **Retain Current Preprocessing:** The geometric distortion introduced by stretching is actually *less* destructive to our 1D radial spectral signature than the high-frequency pollution caused by zero-padding, and preserves more signal than center cropping. 
2. **Do Not Claim Causation:** The prior audit noted diverse image dimensions among false positives/negatives. However, this experiment proves that attempting to "fix" the aspect ratio via standard means (padding/cropping) actually *degrades* model performance. We cannot claim resizing caused the original errors.
3. **Next Steps:** Instead of altering image geometry, future improvements should investigate **Feature Engineering**—for example, employing azimuthal averaging, extracting localized block-based FFTs instead of full-image FFTs, or exploring multi-channel (RGB) spectral differences.

The baseline model `svm_v1` remains the optimal artifact for production and has not been overwritten.
