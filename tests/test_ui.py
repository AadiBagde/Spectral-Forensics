"""Tests for Streamlit UI logic."""

import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path
from PIL import Image

def test_app_loads_successfully():
    """Test that the main app loads without errors."""
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()
    assert not at.exception
    
    # Check if header loaded via markdown in the sidebar
    assert any("SPECTRAL FORENSICS" in md.value for md in at.sidebar.markdown)
