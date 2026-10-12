# Prediction Quality Validation Audit

**Project:** Detection of AI-Generated Images via 2D Fast Fourier Transform and Spectral Artifact Analysis  
**Model Version:** `svm_v1`  
**Date:** October 10, 2026

## 1. Prediction Pipeline & Class Mapping Verification
- **Inspection:** Traced `src/spectral_forensics/classification/interface.py`. 
- **Mapping:** The pipeline explicitly maps SVM output `0` to `REAL` and `1` to `FAKE`. This matches the baseline training configuration precisely.
- **Preprocessing & Features:** The pipeline utilizes the identical extraction functions (`extract_feature_vector` calling `analyze_image`), guaranteeing that the FFT transformations, feature count (128 radial bins), and spectral logic remain symmetrical to the training setup.
- **Correction Made:** A bug was identified where `predict_features` retrieved `predict_proba` (a cross-validation calibrated probability via Platt scaling) instead of `decision_function` when available, causing a mismatch where predicted classes sometimes conflicted with raw score thresholds. `interface.py` was updated to explicitly prioritize the `decision_function` raw margin for accuracy and transparency.

## 2. Evaluation on Existing Test Set
The model was tested against the 146 held-out test images (`datasets/verified_parquet_images/test`). 

**Execution Script:** `scripts/audit_prediction_quality.py`

### Classification Metrics (Historic vs. Retrained with 135 Engineered Features)

| Metric | Historic Baseline (128 Radial Bins) | Updated `svm_v1` (135 Features: 128 Radial + 7 Engineered) | Change |
|---|---|---|---|
| **True REAL (TN)** | 49 / 81 (60.5%) | **52 / 81 (64.2%)** | +3 correct (+3.7%) |
| **True FAKE (TP)** | 41 / 65 (63.1%) | **46 / 65 (70.8%)** | +5 correct (+7.7%) |
| **False Positives (FP)** | 32 | **29** | -3 errors (-9.4%) |
| **False Negatives (FN)** | 24 | **19** | -5 errors (-20.8%) |
| **FAKE Precision** | 0.5616 | **0.6133** | +5.17% |
| **FAKE Recall** | 0.6308 | **0.7077** | +7.69% |
| **Macro F1** | 0.6153 | **0.6707** | +5.54% |
| **Balanced Accuracy**| 0.6179 | **0.6748** | +5.69% |
| **ROC-AUC** | 0.6938 | **0.7064** | +1.26% |

**Finding:** The updated 135-dimensional feature representation (integrating high-frequency energy ratio, roll-off, azimuthal variance, and spectral kurtosis) significantly reduced both False Positives and False Negatives, raising FAKE recall beyond 70% and balanced accuracy to 67.5%.

## 3. Analysis of Errors (False Positives/Negatives)
A metadata probe was executed on sample failed predictions (`scripts/analyze_fp_fn.py`):
- **False Positives (Predicted FAKE, actually REAL):**
  - `real_000000.png` | Score: `0.4832` (reduced from 0.5127)
  - `real_000005.png` | Score: `0.6861` (reduced from 0.8987)
- **False Negatives (Predicted REAL, actually FAKE):**
  - `fake_000003.png` | Score: `-0.2234`
  - `fake_000013.png` | Score: `-0.0988` (closer to 0 threshold)

**Observations:**
Errors occur across drastically varied native image resolutions (ranging from 400x400 to 2812x1568). Due to preprocessing which resizes images before FFT transformation, varying aspect ratios and extensive resizing compressions interfere with the subtle high-frequency spectral bands.

## 4. External Dataset Evaluation & Need for More Data
**Finding:** Currently, there is no reliable, independent external dataset (unseen internet-scraped images or modern generative models like Midjourney v6, FLUX, SDXL) in the repository. 
**Dataset Expansion Recommendation:**
1. **Current Dataset Limitation:** The current training set consists of 826 verified images from older GAN/Parquet splits. Modern diffusion models (SDXL, DALL-E 3, Midjourney, Flux.1) do not leave the exact same grid artifact signatures as older StyleGAN or ProGAN architectures.
2. **Recommended External Sources:**
   - **GenImage Dataset:** A large-scale benchmark containing Real vs. AI-generated image pairs across multiple modern generators (Stable Diffusion, Midjourney, VQ-DM, ADM, Wukong, BigGAN).
   - **DiffusionForensics / SynthWild:** Provides images captured in-the-wild with compression (JPEG quality 75/85/95) to train resilience against social media uploads.
   - **Real Image Diversity:** Natural photos from COCO, ImageNet, or RAISE (uncompressed camera RAW/TIFF) to prevent authentic high-frequency textures (foliage, fur, architecture) from triggering False Positives.

## 5. Decision Score Representation
**Finding:** The UI correctly defines the metric as an "SVM Decision Score". Following the correction to `interface.py`, this score now accurately surfaces the SVM decision function margin rather than Platt-scaled probabilities. The UI disclaimers accurately instruct users not to interpret these values as absolute probabilities of authenticity. 

## 6. Conclusion
The model has been retrained and improved with the new feature schema. Tests pass with 100% success rate. To advance performance from the current ~67.5% balanced accuracy to >85-90%, dataset diversity (especially modern diffusion generators and uncompressed real images) is the single most impactful lever.
