# Spectral Forensics — Architecture (Phase 1)

## Package layout

Reusable engine code lives under `src/spectral_forensics/`:

- `preprocessing/` — load, probe, preprocess, dataset validation
- `frequency/` — 2D FFT, spectra, radial profiles
- `features/` — feature vectors, schema, batch extraction
- `visualization/` — spectrum / radial plots
- `classification/`, `evaluation/`, `reporting/` — Phase 2/3 interfaces only

Runnable CLIs live under `scripts/` and must not reimplement FFT/feature math.

## Engine contract

`analyze_image(path)` returns metadata, preprocessed array, spectrum bundle,
radial profile, and a fixed-length feature vector. Phase 2 training and Phase 3
inference must call this same path (or `extract_feature_vector`).

## Feature schema

Baseline schema version `1.0.0`:

- grayscale (`L`), resize 256×256 LANCZOS, normalize to `[0, 1]`
- `log1p(power)` radial mean → 128 bins (`freq_0` … `freq_127`)
- labels: REAL=0, FAKE=1 (folder-based)

Schema JSON is written to `datasets/processed/feature_schema.json`.
