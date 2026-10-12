"""Matplotlib visualizations for spectral forensic analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from spectral_forensics.features.extract import AnalysisResult
from spectral_forensics.utils.validation import ensure_directory


def plot_analysis_figure(
    result: AnalysisResult,
    original: Image.Image | None = None,
) -> Figure:
    """Create a 1x3 figure: original, log-power spectrum, radial profile."""
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#1a1b1e")

        if original is None:
            display = result.preprocessed
            axes[0].imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
            axes[0].set_title("Preprocessed", color="#d1d5db", fontsize=10)
        else:
            axes[0].imshow(np.asarray(original.convert("RGB")))
            axes[0].set_title("Original Image", color="#d1d5db", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(result.spectrum.log_power, cmap="magma")
        axes[1].set_title("Log Power Spectrum (2D FFT)", color="#d1d5db", fontsize=10)
        axes[1].axis("off")

        radii = np.arange(result.radial_profile.shape[0])
        axes[2].plot(radii, result.radial_profile, color="#60a5fa", linewidth=1.5)
        axes[2].set_title("Radial Frequency Profile", color="#d1d5db", fontsize=10)
        axes[2].set_xlabel("Frequency Bin (Low → High)", color="#9ca3af", fontsize=8)
        axes[2].set_ylabel("Mean log₁ₚ(power)", color="#9ca3af", fontsize=8)
        axes[2].tick_params(colors="#9ca3af", labelsize=7)
        axes[2].set_facecolor("#111827")
        for spine in axes[2].spines.values():
            spine.set_edgecolor("#374151")
        axes[2].grid(True, alpha=0.2, color="#4b5563")

        fig.suptitle(result.metadata.filename, fontsize=10, color="#9ca3af")
        fig.tight_layout(pad=1.5)
    return fig


def plot_detailed_forensic_figure(
    result: AnalysisResult,
    pred_class: str = "Unknown",
    confidence_pct: float = 50.0,
) -> Figure:
    """Create a 1x3 detailed diagnostic figure:
    1. High-Frequency Artifact Residual Map
    2. Horizontal vs Vertical Spectral Cross-Section (Lattice detection)
    3. Low/Mid/High Frequency Energy Bar Breakdown
    """
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor("#1a1b1e")

        # 1. High-Frequency Residual Map (Filter out low frequencies to reveal synthetic grid)
        spec = result.spectrum.log_power
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        # High pass mask: frequencies > radius 40
        hf_mask = dist_from_center > 40
        hf_residual = np.where(hf_mask, spec, np.nan)

        im1 = axes[0].imshow(hf_residual, cmap="inferno")
        axes[0].set_title("High-Frequency Lattice Probe", color="#d1d5db", fontsize=10)
        axes[0].axis("off")
        cbar = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7, colors="#9ca3af")

        # 2. Horizontal vs Vertical cross-section
        h_slice = spec[cy, :]
        v_slice = spec[:, cx]
        axes[1].plot(h_slice, label="Horizontal Axis", color="#38bdf8", linewidth=1.4)
        axes[1].plot(v_slice, label="Vertical Axis", color="#f472b6", linewidth=1.4, linestyle="--")
        axes[1].set_title("Axial Cross-Section (Grid Symmetry)", color="#d1d5db", fontsize=10)
        axes[1].set_xlabel("Pixel Coordinate", color="#9ca3af", fontsize=8)
        axes[1].set_ylabel("Log Power", color="#9ca3af", fontsize=8)
        axes[1].legend(loc="upper right", fontsize=8, facecolor="#1f2937", edgecolor="#374151")
        axes[1].tick_params(colors="#9ca3af", labelsize=7)
        axes[1].set_facecolor("#111827")
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#374151")
        axes[1].grid(True, alpha=0.2, color="#4b5563")

        # 3. Energy Band Distribution (Low, Mid, High)
        profile = result.radial_profile
        n = len(profile)
        low_band = float(np.sum(profile[: n // 3]))
        mid_band = float(np.sum(profile[n // 3 : 2 * n // 3]))
        high_band = float(np.sum(profile[2 * n // 3 :]))
        total = low_band + mid_band + high_band + 1e-8
        
        bands = ["Low Freq\n(Coarse)", "Mid Freq\n(Textures)", "High Freq\n(Fine Details)"]
        pcts = [low_band / total * 100, mid_band / total * 100, high_band / total * 100]
        colors_list = ["#34d399", "#fbbf24", "#f87171" if pred_class == "FAKE" else "#60a5fa"]

        bars = axes[2].bar(bands, pcts, color=colors_list, width=0.55, edgecolor="#374151")
        axes[2].set_title("Spectral Energy Band Decomposition", color="#d1d5db", fontsize=10)
        axes[2].set_ylabel("Power Share (%)", color="#9ca3af", fontsize=8)
        axes[2].tick_params(colors="#9ca3af", labelsize=8)
        axes[2].set_facecolor("#111827")
        axes[2].set_ylim(0, max(pcts) * 1.25 if pcts else 100)
        for spine in axes[2].spines.values():
            spine.set_edgecolor("#374151")
        axes[2].grid(True, axis="y", alpha=0.2, color="#4b5563")

        for bar in bars:
            h_val = bar.get_height()
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                h_val + 1.5,
                f"{h_val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#e5e7eb",
            )

        fig.tight_layout(pad=1.5)
    return fig


def save_analysis_figure(
    result: AnalysisResult,
    output_path: Path | str,
    original: Image.Image | None = None,
) -> Path:
    """Save the analysis figure to disk and close the figure."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    fig = plot_analysis_figure(result, original=original)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
