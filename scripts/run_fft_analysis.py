"""CLI: analyze one image — metadata, FFT figure, radial profile, features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from spectral_forensics.features.extract import analyze_image
from spectral_forensics.paths import get_project_paths
from spectral_forensics.preprocessing.loader import ImageLoadError, load_image
from spectral_forensics.utils.logging import get_logger, setup_logging
from spectral_forensics.utils.validation import ensure_directory
from spectral_forensics.visualization.spectrum import save_analysis_figure

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 forensic engine analysis on a single image."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=None,
        help="Path to an image. Defaults to first train/REAL image if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figure/JSON outputs "
        "(default: reports/spectral_visualizations/samples).",
    )
    return parser.parse_args(argv)


def _default_sample(paths) -> Path:
    sample_dir = paths.class_dir("train", "REAL")
    candidates = sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No sample images found in {sample_dir}")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    paths = get_project_paths()

    image_path = args.image or _default_sample(paths)
    if not image_path.is_absolute():
        candidate = (Path.cwd() / image_path).resolve()
        if candidate.exists():
            image_path = candidate
        else:
            image_path = (paths.root / image_path).resolve()

    output_dir = args.output_dir or (
        paths.spectral_visualizations / "samples"
    )
    if not output_dir.is_absolute():
        output_dir = (paths.root / output_dir).resolve()
    ensure_directory(output_dir)

    logger.info("Analyzing: %s", image_path)
    try:
        result = analyze_image(image_path)
        original = load_image(image_path)
    except ImageLoadError as exc:
        logger.error("%s", exc)
        return 1

    stem = image_path.stem
    figure_path = output_dir / f"{stem}_analysis.png"
    json_path = output_dir / f"{stem}_analysis.json"
    npy_path = output_dir / f"{stem}_features.npy"

    save_analysis_figure(result, figure_path, original=original)
    np.save(npy_path, result.feature_vector)

    payload = {
        "metadata": result.metadata.to_dict(),
        "n_features": result.n_features,
        "feature_names": result.feature_names,
        "feature_vector": result.feature_vector.tolist(),
        "radial_profile_stats": {
            "min": float(result.radial_profile.min()),
            "max": float(result.radial_profile.max()),
            "mean": float(result.radial_profile.mean()),
        },
        "preprocess": result.config.preprocess.to_dict(),
        "spectrum_representation": result.config.spectrum_representation,
        "schema_version": result.config.schema_version,
        "outputs": {
            "figure": str(figure_path),
            "features_npy": str(npy_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print()
    print("=" * 70)
    print("FFT / FEATURE ANALYSIS")
    print("=" * 70)
    print(f"Image: {result.metadata.path}")
    print(
        f"Original: {result.metadata.width}x{result.metadata.height} "
        f"mode={result.metadata.mode} format={result.metadata.format}"
    )
    print(f"SHA-256: {result.metadata.sha256}")
    print(f"Preprocessed shape: {result.preprocessed.shape}")
    print(f"Spectrum shape: {result.spectrum.log_power.shape}")
    print(f"Radial bins: {result.n_features}")
    print(
        f"Radial stats: min={payload['radial_profile_stats']['min']:.4f} "
        f"max={payload['radial_profile_stats']['max']:.4f} "
        f"mean={payload['radial_profile_stats']['mean']:.4f}"
    )
    print(f"Figure: {figure_path}")
    print(f"JSON:   {json_path}")
    print(f"NPY:    {npy_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
