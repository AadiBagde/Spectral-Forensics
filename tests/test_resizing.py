import numpy as np
from PIL import Image
import pytest

from spectral_forensics.config import DEFAULT_PREPROCESS
from spectral_forensics.preprocessing.transform import preprocess_image

def test_preprocessing_dimensions():
    """Test that preprocessed images match target configuration dimensions."""
    # Create rectangular image
    img = Image.new("RGB", (800, 400), color="white")
    processed = preprocess_image(img, DEFAULT_PREPROCESS)
    
    assert processed.shape == (DEFAULT_PREPROCESS.image_size, DEFAULT_PREPROCESS.image_size), \
        f"Expected {(DEFAULT_PREPROCESS.image_size, DEFAULT_PREPROCESS.image_size)}, got {processed.shape}"
    assert processed.dtype == np.float32

def test_preprocessing_determinism():
    """Test that processing the same image twice yields exactly the same array."""
    np.random.seed(42)
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    
    out1 = preprocess_image(img, DEFAULT_PREPROCESS)
    out2 = preprocess_image(img, DEFAULT_PREPROCESS)
    
    np.testing.assert_array_equal(out1, out2)

def test_preprocessing_grayscale_conversion():
    """Test that color mode is respected."""
    img = Image.new("RGB", (100, 100), color="red")
    processed = preprocess_image(img, DEFAULT_PREPROCESS)
    
    # "L" mode creates a 2D array, whereas "RGB" would create a 3D array
    assert processed.ndim == 2
