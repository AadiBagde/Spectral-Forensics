import sys
sys.path.insert(0, "src")
import numpy as np
from pathlib import Path
from PIL import Image
from spectral_forensics.features.extract import analyze_image

def test_visualization():
    img1_path = Path("datasets/verified_parquet_images/test/FAKE/fake_000000.png")
    img2_path = Path("datasets/verified_parquet_images/test/FAKE/fake_000001.png")
    
    print(f"Testing with:\n1. {img1_path}\n2. {img2_path}")
    
    res1 = analyze_image(img1_path)
    res2 = analyze_image(img2_path)
    
    spectrum_diff = np.max(np.abs(res1.spectrum.log_power - res2.spectrum.log_power))
    radial_diff = np.max(np.abs(res1.radial_profile - res2.radial_profile))
    
    print(f"Spectrum diff: {spectrum_diff}")
    print(f"Radial profile diff: {radial_diff}")
    
    if spectrum_diff == 0:
        print("ERROR: Spectrums are identical!")
    if radial_diff == 0:
        print("ERROR: Radial profiles are identical!")

if __name__ == "__main__":
    test_visualization()
