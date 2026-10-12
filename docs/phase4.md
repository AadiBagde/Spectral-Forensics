# Phase 4: Advanced Model Improvement & Forensic Validation

## Overview
Phase 4 focused on establishing a reproducible baseline, investigating dataset integrity and model generalizability, and experimentally seeking model improvements without over-engineering or introducing test-set leakage. The objective was to transparently assess what the spectral forensic model has learned and outline its true boundaries.

## 1. Reproducible Baseline
- **Model:** Phase 2 `models/svm_v1`
- **Evaluation Set:** 146 held-out test images
- **Baseline Accuracy:** 61.64%
- **Baseline FAKE-Class F1:** 0.5942
- *Saved under:* `experiments/phase4/baseline/`

## 2. Data Integrity Audit
- **Train/Test Leakage:** Verified via SHA-256 hashes. 0 exact duplicates span across the train/test splits.
- **Provenance limitation:** Current metadata lacks generator-specific annotations. We cannot isolate the origin of the AI-generated samples to evaluate specific cross-generator performance. 
- *Saved under:* `experiments/phase4/data_integrity/`

## 3. Generalization to Unseen Generators
Due to the dataset limitations identified above, no reliable, verifiable external test set composed of strictly disjoint/novel AI generators was available. 
**Conclusion:** We explicitly **cannot claim** that the classifier generalizes to unseen generators in the wild. Future work must establish a dedicated protocol with independently sourced, fully-provenanced images (e.g., separating StyleGAN3 training sets from Midjourney v6 test sets) to answer this question.

## 4. Controlled Model Improvement
We performed rigorous `GridSearchCV` on the 826 training samples utilizing 5-fold Stratified CV, optimizing for FAKE-class F1 score. We tested SVM, Random Forest, and Logistic Regression with varied hyperparameters.
- **Winner:** Logistic Regression (`C=0.01`, `penalty=l2`)
- **CV Score:** FAKE-F1 = 0.5936
- *Saved under:* `models/phase4_candidate/`

## 5. Final Comparison
We evaluated the new candidate on the unaltered test set and compared it to the baseline.
- **Baseline Accuracy:** 61.64% (FAKE F1: 0.5942)
- **Candidate Accuracy:** 60.96% (FAKE F1: 0.5797)
- **Difference:** Accuracy (-0.68%), F1 (-0.0145)

**Outcome:** The experimental model performed slightly worse on the held-out test set. We retained the original `svm_v1` baseline. No silent replacements were made.

## 6. Explainability & Frequency-Band Analysis
By examining the mean radial frequency profiles and querying Random Forest feature importances, we observed:
- Differences between REAL and FAKE spectra are diffuse but frequently concentrate around specific mid-to-high frequency bands, reflecting upsampling or convolution artifacts.
- **Caution:** High feature importance in these bins describes discriminative utility on *this specific dataset*. It does not imply a universal causal mechanism for all generative architectures.
- *Saved under:* `experiments/phase4/explainability/`

## 7. Robustness Follow-Up
Re-evaluation of the Phase 2 robustness metrics clarified an apparent paradox: F1 scores artificially improved slightly when severe blur or aggressive resizing was applied.
- **Finding:** Such transformations act as aggressive low-pass filters. They destroyed high-frequency noise that was previously confusing the model. The "improvement" was an artifact of standardizing the input signal for these specific 146 images, rather than a true demonstration of resilience. The model remains inherently sensitive to frequency perturbations.

## 8. Integration and Regression
- **Test suite:** The full automated test suite was executed to ensure the Phase 3 Streamlit App and Phase 1 FFT logic remain uncorrupted by Phase 4 investigations.
- **Result:** 21 tests passed (0 failed, 0 skipped).

## Commands Executed
```bash
python scripts/phase4_baseline.py
python scripts/phase4_data_integrity.py
python scripts/phase4_model_improvement.py
python scripts/phase4_final_comparison.py
python scripts/phase4_explainability.py
python scripts/phase4_robustness.py
pytest -v
```
