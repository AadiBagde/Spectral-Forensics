# Spectral Forensics Presentation Outline

**1. Problem Statement**
- The rise of photorealistic AI generators (Midjourney, Stable Diffusion).
- The growing threat of misinformation and the difficulty of naked-eye detection.

**2. Motivation**
- Standard deep-learning detectors act as black boxes.
- We need transparent, mathematically verifiable forensic features to explain *why* an image is flagged.

**3. Objectives**
- Isolate and quantify generative upsampling artifacts in the frequency domain.
- Build an end-to-end processing pipeline, from pixel ingestion to PDF forensic reporting.

**4. Methodology**
- **Spatial to Frequency:** 2D Fast Fourier Transform (FFT).
- **Dimensionality Reduction:** Radial profile integration.
- **Classification:** Support Vector Machine (SVM) on the extracted 128-bin spectrum.

**5. System Architecture**
- Highlight modularity: `features`, `classification`, `visualization`, `ui`.
- Explain how the Streamlit dashboard safely bridges user uploads into the mathematical engine.

**6. Experimental Results**
- **Baseline:** 61.64% test accuracy (FAKE-F1: 0.5942). This represents an observed improvement on this specific test set against the majority-class baseline (55.48%).
- **Explainability:** Feature importance analysis is exploratory and dataset-specific. It highlights mid-to-high frequency bands as primary discriminators for this dataset, but does not establish a universal causal mechanism for AI-generation artifacts.

**7. Dashboard Demonstration**
- Live demo of single-image upload.
- Live demo of batch processing.
- Generation and display of the PDF Forensic Report.

**8. Limitations**
- **Provenance:** We do not know which AI models generated the FAKE dataset, prohibiting claims of out-of-distribution generalizability.
- **Perturbations:** Extreme blur and downscaling act as low-pass filters, fundamentally altering the high-frequency spectrum and confounding the model.
- **Interpretation:** The tool calculates spectral similarities; it does not yield a definitive, causal proof of image origin.

**9. Future Work**
- Compile a fully provenanced dataset to test cross-generator performance.
- Evaluate 2D Convolutional Neural Networks on the raw un-collapsed FFT power spectrum to preserve directional artifacts.

**10. Conclusion**
- A successfully engineered forensic prototype that prioritizes transparency and scientific integrity over inflated performance claims.
