import numpy as np
from pathlib import Path
from PIL import Image
from spectral_forensics.features.extract import analyze_image

def test_visualization_determinism_and_uniqueness(tmp_path):
    # Create two different dummy images
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    
    # Image 1: Random noise
    np.random.seed(42)
    arr1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    Image.fromarray(arr1).save(img1_path)
    
    # Image 2: Different pattern (e.g., gradient)
    arr2 = np.linspace(0, 255, 256*256).reshape(256, 256).astype(np.uint8)
    Image.fromarray(arr2).save(img2_path)
    
    # Analyze images
    res1_a = analyze_image(img1_path)
    res1_b = analyze_image(img1_path)
    res2 = analyze_image(img2_path)
    
    # 1. Determinism: same image processed twice should yield exactly same results
    np.testing.assert_array_equal(res1_a.spectrum.log_power, res1_b.spectrum.log_power)
    np.testing.assert_array_equal(res1_a.radial_profile, res1_b.radial_profile)
    
    # 2. Uniqueness: different images should yield different results
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(res1_a.spectrum.log_power, res2.spectrum.log_power)
        
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(res1_a.radial_profile, res2.radial_profile)
