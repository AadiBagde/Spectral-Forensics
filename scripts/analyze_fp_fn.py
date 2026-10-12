import sys
sys.path.insert(0, "src")
import pandas as pd
from pathlib import Path
from PIL import Image

def get_meta(path):
    with Image.open(path) as img:
        return {"width": img.width, "height": img.height, "mode": img.mode}

def analyze():
    test_dir = Path("datasets/verified_parquet_images/test")
    fp_names = ["real_000000.png", "real_000003.png", "real_000005.png", "real_000006.png", "real_000007.png"]
    fn_names = ["fake_000003.png", "fake_000007.png", "fake_000008.png", "fake_000012.png", "fake_000013.png"]
    
    for name in fp_names:
        p = test_dir / "REAL" / name
        print(f"FP {name}: {get_meta(p)}")
        
    for name in fn_names:
        p = test_dir / "FAKE" / name
        print(f"FN {name}: {get_meta(p)}")

if __name__ == "__main__":
    analyze()
