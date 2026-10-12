# Phase 4 Data Integrity Audit

## Exact Duplicates (SHA-256)
- **Train/Test Leakage (Overlap):** 0 images
- **Train Internal Duplicates:** 1
- **Test Internal Duplicates:** 0

## Class Distributions
- **Train Set:** {'REAL': 457, 'FAKE': 369}
- **Test Set:** {'REAL': 81, 'FAKE': 65}

## Metadata & Provenance
- **Generator Labels Present:** False
- **Provenance Data:** None available in standard dataset features. We do not know which specific AI generators produced the FAKE images (e.g. Midjourney, Stable Diffusion, GANs).

## Near-Duplicates
Not comprehensively checked - Requires perceptual hashing library (e.g. imagehash) which is not in dependencies. No obvious filename overlaps detected.

*Conclusion:* There is no exact file-hash leakage between train and test sets. However, the lack of generator provenance means cross-generator generalization cannot be definitively proven with the current dataset alone.
