# Phase 2 Final Verification Audit

## 1. Verify Evaluation Metrics
**Checks Performed:**
- Audited the `final_evaluation_report.json` and the test predictions on `verified_test_features.csv`.
- Analyzed the distribution of the 146 held-out test images.

**Findings:**
- **Metric Definitions:** The reported test F1 score of **0.5942** explicitly represents the binary F1 score for the **FAKE class (label=1)**. The FAKE label and positive-class definition remained perfectly consistent throughout training, evaluation, and reporting. 
- **Baseline Comparison:** The test set contains 81 REAL and 65 FAKE images. A naive baseline that always predicts the majority class (REAL) would achieve an accuracy of **55.48%**.
- **Model Performance:** The SVM model achieved a test accuracy of **61.64%** (90/146 correct predictions). 
- **Confusion Matrix Validation:** The accuracy and class recalls mathematically align with the confusion matrix:
  - **REAL**: 49 True Positives (REAL), 32 False Positives (predicted FAKE) — Recall: 60.5%
  - **FAKE**: 41 True Positives (FAKE), 24 False Negatives (predicted REAL) — Recall: 63.1%

*Conclusion:* The model learns a discriminative signal outperforming the baseline by ~6%, but as noted in the scientific review, its performance is modest and not yet strong enough to claim highly reliable AI-image detection.

## 2. Audit Cross-Validation and Leakage
**Checks Performed:**
- Code review of `src/spectral_forensics/classification/training.py` and `scripts/student2_ml_classifier.py`.

**Findings:**
- **No Data Leakage:** The `StandardScaler` is strictly encapsulated within the `sklearn.pipeline.Pipeline`. During the 5-fold `StratifiedKFold` cross-validation, the scaler is dynamically fitted only on the `k-1` folds, preventing train-set scaling leakage.
- **Strict Test-Set Isolation:** The 146 test records were exclusively held back until the final `best_pipeline.predict(X_test)` call. They were not used for model selection, hyperparameter tuning, thresholding, or feature selection.
- **Dataset Matching:** Training and testing inputs match exactly the outputs of `extract_verified_features.py` which are mapped deterministically from `datasets/verified_parquet_images/`.
- **Limitation Acknowledged:** While there is strict split separation based on the verified dataset structure, we cannot mathematically guarantee that cross-generator leakage or near-duplicates (from the original raw source generation) do not exist between train and test sets, as explicit cross-generator deduplication checks were not executed in Phase 2.

## 3. Audit Robustness Results
**Checks Performed:**
- Reviewed `robustness_report.json` and the execution logs of `student2_robustness.py`.
- Verified the perturbation routines in `preprocessing/perturbations.py`.

**Findings:**
- **Test Integrity:** The script correctly loaded the same 146 test images, applied perturbations in memory, and piped them through the exact same preprocessing and feature extraction logic (`analyze_image`) used during training.
- **Completion:** All 12 intended perturbation settings ran completely (146 * 12 = 1,752 individual inferences).
- **Metric Consistency:** The evaluation code used the exact same `evaluate_predictions` interface to generate the F1 score.

**Results & Analysis (Baseline FAKE F1 = 0.5942):**
- **JPEG Compression:** Level 90 (`F1: 0.5857`, `-0.0085`), Level 75 (`F1: 0.5942`, `0.0000`), Level 50 (`F1: 0.5874`, `-0.0068`).
- **Gaussian Blur:** Radius 1.0 (`F1: 0.6474`, `+0.0532`), Radius 2.0 (`F1: 0.6364`, `+0.0422`), Radius 3.0 (`F1: 0.6184`, `+0.0242`).
- **Resizing:** Scale 0.75x (`F1: 0.6164`, `+0.0222`), Scale 0.5x (`F1: 0.6410`, `+0.0468`), Scale 0.25x (`F1: 0.6383`, `+0.0441`).
- **Gaussian Noise:** Level 10 (`F1: 0.5797`, `-0.0145`), Level 25 (`F1: 0.5735`, `-0.0207`), Level 50 (`F1: 0.5986`, `+0.0044`).

*Conclusion & Limitations:* The SVM model appears relatively stable under moderate perturbations. The unusually improved scores for Blur and Resize (up to +0.05) are likely artifacts of high-frequency attenuation standardizing the signal on this specific subset of 146 images, rather than proof of generalized robustness. Robustness on *this* dataset does not imply generalization to unseen AI generators. The model was correctly *not* retrained on these perturbed images.

## 4. Verify Artifacts and Tests
**Checks Performed:**
- Execution of inference API (`scratch/audit_script.py`).
- Run of automated project tests (`pytest`).

**Findings:**
- **API Functionality:** The saved SVM pipeline successfully loads via `load_model()` in a fresh Python process. When provided a real test image via `analyze_image()`, `predict_features()` cleanly executes and outputs a valid `PredictionResult` (e.g., `Prediction on real_000000.png: label=1, score=0.563, class_name=FAKE`).
- **Test Suite Execution:** `pytest -v` was executed across `tests/`. 
  - Passed: **20**
  - Failed: **0**
  - Skipped: **0**
- **Artifact Presence:** The JSON metrics, test scripts, extraction records, and plots exist in `experiments/phase2/` and accurately mirror the data logged in this audit.

## 5. Audit Conclusion
**Status:** **PASSED**
**Blockers for Phase 3:** None.

Phase 2 is fully verified and correctly implemented. The model performs modestly but validly acts as a functional foundation. The data, API, and validation requirements are completely satisfied, qualifying the project to proceed to **Phase 3 (Streamlit Dashboard, Batch Analysis, and PDF Forensic Reporting)**.
