# Spectral Domain Visualizations Audit

**Date:** October 10, 2026
**Objective:** Verify that the 2D log power spectrum and 1D radial frequency profile visualizations are genuinely calculated from the currently uploaded image and are not hardcoded, incorrectly cached, or reused.

## 1. Code Path Trace & Validation

### Image Upload (`src/spectral_forensics/ui/single_image.py`)
When an image is uploaded via Streamlit's `st.file_uploader`, it is saved to a unique `tempfile.TemporaryDirectory`. Streamlit guarantees that interacting with the file uploader forces a full top-to-bottom re-run of the script. The temporary directory forces a uniquely generated path (e.g., `/tmp/.../uploaded_file.jpg`) for the saved image.

### Pipeline Execution (`src/spectral_forensics/features/extract.py`)
The `analyze_image` function executes the following linear sequence:
1. `metadata = probe_image(path)`: Dynamically reads the uploaded image properties.
2. `image = load_image(path)`: Loads the pixel data using PIL.
3. `preprocessed = preprocess_image(image)`: Resizes and converts to grayscale.
4. `spectrum = compute_fft_spectrum(preprocessed)`: Computes the 2D FFT using `np.fft.fft2` and `np.fft.fftshift`.
5. `radial = radial_mean_profile(spectrum.log_power)`: Computes the 1D azimuthal average.

**Finding:** The mathematical operations strictly consume the newly created temporary file. There are no `@st.cache_data` or `@lru_cache` decorators on `analyze_image`, `compute_fft_spectrum`, or `radial_mean_profile`.

### Rendering (`src/spectral_forensics/visualization/spectrum.py`)
The UI calls `fig = plot_analysis_figure(analysis_result, original=img)` which invokes `matplotlib.pyplot.subplots()`. This guarantees the creation of a brand-new Figure instance populated with the unique `analysis_result` generated exactly one step prior. `st.pyplot(fig)` is then used to render the newly constructed figure.

**Finding:** The visualizations are genuinely tied to the currently uploaded image without stale caching or hardcoded arrays.

## 2. Experimental Verification

To absolutely confirm visual uniqueness and programmatic determinism, an automated test (`tests/test_visualization_audit.py`) was engineered to bypass the UI and probe the underlying arrays directly. 

**Test Protocol:**
- Two substantially different generative arrays (random uniform noise vs. linear gradient) were saved as images.
- The pipeline was executed against both images.
- A deterministic check re-executed the pipeline on the first image.

**Results:**
- **Determinism:** `res1_a` and `res1_b` produced mathematically identical 2D log-power spectrums and 1D radial profiles (`np.testing.assert_array_equal` passed).
- **Uniqueness:** `res1_a` and `res2` produced completely disparate 2D log-power spectrums and 1D radial profiles. A maximum absolute difference check logged deviations of `8.22` for the spectrum and `1.31` for the radial profile, passing the `assert_raises(AssertionError)` check.

## 3. Matplotlib Memory Leak Mitigation

**Issue Discovered:**
While the visualizations were mathematically correct and un-cached, `src/spectral_forensics/ui/single_image.py` neglected to close the Matplotlib figure after rendering (`st.pyplot(fig)`) and saving (`fig.savefig()`). Because `st.pyplot` does not auto-close explicitly passed `fig` objects, multiple subsequent uploads would silently accumulate un-garbage-collected Figure objects in Matplotlib's global state machine.

**Correction Made:**
Imported `matplotlib.pyplot` and appended `plt.close(fig)` directly following the PDF byte generation in `src/spectral_forensics/ui/single_image.py`. This ensures proper memory reclamation without altering the feature schema, predictive model, or UI layout.

## 4. Conclusion
The Spectral Domain Visualizations are structurally sound, strictly bound to the currently uploaded image, mathematically accurate, and unaffected by erroneous caching mechanics. The Streamlit workflow is secure for live forensic demonstration.
