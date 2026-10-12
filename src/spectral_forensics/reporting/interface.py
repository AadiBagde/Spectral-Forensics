"""Placeholder reporting API for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportRequest:
    """Inputs required to build a forensic PDF report later."""

    image_path: Path
    analysis_summary: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    output_path: Path | None = None


def generate_pdf_report(request: ReportRequest) -> Path:
    """Generate a forensic PDF report.

    Not implemented in Phase 1.
    """
    raise NotImplementedError(
        "PDF reporting is implemented in Phase 3."
    )
