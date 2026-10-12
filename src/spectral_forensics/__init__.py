"""Spectral Forensics — core forensic engine package."""

from spectral_forensics.config import FeatureConfig, PreprocessConfig
from spectral_forensics.features.extract import (
    AnalysisResult,
    analyze_image,
    extract_feature_vector,
)
from spectral_forensics.paths import ProjectPaths, get_project_paths

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "FeatureConfig",
    "PreprocessConfig",
    "ProjectPaths",
    "analyze_image",
    "extract_feature_vector",
    "get_project_paths",
    "__version__",
]
