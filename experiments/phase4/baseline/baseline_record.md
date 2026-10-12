# Phase 4 Baseline Record

**Model:** models/svm_v1
**Feature Schema:** 128 ordered features (freq_0 to freq_127)
**Training Records:** 826
**Test Records:** 146

## Evaluation Metrics (Held-out Test Set)
- **Accuracy:** 0.6164
- **FAKE-Class F1:** 0.5942 (Precision: 0.5616, Recall: 0.6308)
- **REAL-Class F1:** 0.6364 (Precision: 0.6712, Recall: 0.6049)
- **Macro F1:** 0.6153
- **Weighted F1:** 0.6176
- **ROC-AUC:** 0.6936372269705604

### Confusion Matrix
| | Predicted REAL (0) | Predicted FAKE (1) |
|---|---|---|
| **Actual REAL (0)** | 49 | 32 |
| **Actual FAKE (1)** | 24 | 41 |

*This baseline is preserved as the standard for Phase 4 model improvements.*
