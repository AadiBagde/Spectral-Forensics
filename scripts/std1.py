"""
student1_features.py

Spectral Forensics
Detection of AI-Generated Images via 2D FFT
and Spectral Artifact Analysis

Purpose:
    1. Load REAL and FAKE images
    2. Convert images to grayscale
    3. Resize to a fixed resolution
    4. Normalize pixel values
    5. Apply 2D FFT
    6. Shift zero frequency to the center
    7. Calculate magnitude and power spectrum
    8. Apply logarithmic transformation
    9. Calculate 1D radial power spectrum
    10. Save frequency-domain features to CSV

Dataset structure expected:

datasets/
│
├── parquet_images/
│   ├── train/
│   │   ├── REAL/
│   │   └── FAKE/
│   │
│   └── test/
│       ├── REAL/
│       └── FAKE/
│
└── processed/
    ├── train_features.csv
    └── test_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Project root:
# Spectral_Forensics/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input dataset directory
DATASET_DIR = PROJECT_ROOT / "datasets" / "parquet_images"

# Output directory for processed features
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

# Fixed image size for FFT processing
IMAGE_SIZE = 256


# ============================================================
# RADIAL POWER SPECTRUM
# ============================================================

def radial_power_spectrum(power_spectrum):
    """
    Convert the 2D power spectrum into a 1D radial power spectrum.

    The center of the FFT-shifted spectrum represents low frequencies.
    Moving away from the center represents increasing spatial frequency.

    For every radius r:
        - Find pixels at approximately that radius
        - Calculate their mean spectral power

    Returns:
        1D numpy array containing radial frequency features.
    """

    height, width = power_spectrum.shape

    # Center of the FFT-shifted image
    center_y = height // 2
    center_x = width // 2

    # Create coordinate grid
    y, x = np.indices((height, width))

    # Distance of every pixel from FFT center
    radius = np.sqrt(
        (x - center_x) ** 2 +
        (y - center_y) ** 2
    )

    # Convert radius to integer bins
    radius = radius.astype(np.int32)

    # Maximum useful radius
    max_radius = min(center_x, center_y)

    # Store average power for each radius
    radial_profile = np.zeros(max_radius, dtype=np.float64)

    for r in range(max_radius):

        mask = radius == r

        if np.any(mask):
            radial_profile[r] = np.mean(power_spectrum[mask])

    return radial_profile


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):
    """
    Process one image and extract its frequency-domain features.

    Processing pipeline:

        Image
          ↓
        Grayscale
          ↓
        Resize 256x256
          ↓
        Normalize [0,1]
          ↓
        2D FFT
          ↓
        FFT Shift
          ↓
        Magnitude Spectrum
          ↓
        Power Spectrum
          ↓
        Log Transform
          ↓
        Radial Power Spectrum

    Returns:
        1D numpy array containing spectral features.
    """

    # --------------------------------------------------------
    # 1. Load image
    # --------------------------------------------------------

    image = Image.open(image_path)

    # --------------------------------------------------------
    # 2. Convert RGB/RGBA/etc. to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # --------------------------------------------------------
    # 3. Resize to fixed resolution
    # --------------------------------------------------------

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 4. Convert image to NumPy array
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 5. Normalize pixel values
    #
    # Original:
    #     0 -> black
    #     255 -> white
    #
    # Normalized:
    #     0.0 -> black
    #     1.0 -> white
    # --------------------------------------------------------

    image_array /= 255.0

    # --------------------------------------------------------
    # 6. Apply 2D Fast Fourier Transform
    # --------------------------------------------------------

    fft = np.fft.fft2(image_array)

    # --------------------------------------------------------
    # 7. Shift zero-frequency component to center
    # --------------------------------------------------------

    fft_shifted = np.fft.fftshift(fft)

    # --------------------------------------------------------
    # 8. Calculate magnitude spectrum
    # --------------------------------------------------------

    magnitude = np.abs(fft_shifted)

    # --------------------------------------------------------
    # 9. Calculate power spectrum
    #
    # Power = |FFT|²
    # --------------------------------------------------------

    power = magnitude ** 2

    # --------------------------------------------------------
    # 10. Logarithmic transformation
    #
    # FFT power values can have a very large dynamic range.
    #
    # log(1 + power) compresses the range and makes
    # spectral patterns easier to analyze.
    # --------------------------------------------------------

    power_log = np.log1p(power)

    # --------------------------------------------------------
    # 11. Convert 2D spectrum into 1D radial spectrum
    # --------------------------------------------------------

    radial_features = radial_power_spectrum(power_log)

    return radial_features


# ============================================================
# IMAGE FILE DISCOVERY
# ============================================================

def get_image_files(directory):
    """
    Find supported image files inside a directory.

    Supported formats:
        PNG
        JPG
        JPEG
        BMP
        WEBP
    """

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.webp"
    ]

    image_files = []

    for extension in extensions:
        image_files.extend(directory.glob(extension))

    return sorted(image_files)


# ============================================================
# PROCESS ONE DATASET SPLIT
# ============================================================

def process_split(split_name):
    """
    Process either the train or test dataset.

    Expected structure:

        parquet_images/
            train/
                REAL/
                FAKE/

            test/
                REAL/
                FAKE/

    Saves:

        datasets/processed/train_features.csv

    or:

        datasets/processed/test_features.csv
    """

    split_dir = DATASET_DIR / split_name

    print()
    print("=" * 70)
    print(f"PROCESSING {split_name.upper()} DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Check split directory
    # --------------------------------------------------------

    if not split_dir.exists():

        print(f"ERROR: Dataset directory not found:")
        print(f"       {split_dir}")

        return

    records = []

    # --------------------------------------------------------
    # Process REAL and FAKE separately
    # --------------------------------------------------------

    for label_name, label_value in [
        ("REAL", 0),
        ("FAKE", 1)
    ]:

        label_dir = split_dir / label_name

        print()
        print(f"Processing {label_name} images...")
        print(f"Directory: {label_dir}")

        # ----------------------------------------------------
        # Check label directory
        # ----------------------------------------------------

        if not label_dir.exists():

            print(
                f"WARNING: Directory does not exist: "
                f"{label_dir}"
            )

            continue

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        image_files = get_image_files(label_dir)

        print(
            f"Found {len(image_files)} "
            f"{label_name} images."
        )

        # ----------------------------------------------------
        # Process every image
        # ----------------------------------------------------

        for index, image_path in enumerate(image_files, start=1):

            try:

                # Extract frequency features
                features = extract_features(image_path)

                # --------------------------------------------
                # Create metadata record
                # --------------------------------------------

                record = {
                    "image": image_path.name,
                    "label": label_value,
                    "class": label_name
                }

                # --------------------------------------------
                # Store radial frequency features
                #
                # freq_0 = lowest frequency
                # freq_1 = next frequency
                # ...
                # --------------------------------------------

                for i, value in enumerate(features):

                    record[f"freq_{i}"] = value

                records.append(record)

                # --------------------------------------------
                # Progress display
                # --------------------------------------------

                if index % 50 == 0 or index == len(image_files):

                    print(
                        f"  Processed "
                        f"{index}/{len(image_files)}"
                    )

            except Exception as error:

                print()
                print(
                    f"ERROR processing image:"
                )
                print(
                    f"  {image_path}"
                )
                print(
                    f"  Reason: {error}"
                )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    dataframe = pd.DataFrame(records)

    # --------------------------------------------------------
    # Make sure output directory exists
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Output CSV path
    # --------------------------------------------------------

    output_file = (
        PROCESSED_DIR /
        f"{split_name}_features.csv"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    dataframe.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 70)
    print(f"{split_name.upper()} PROCESSING COMPLETE")
    print("-" * 70)

    print(
        f"Total samples: "
        f"{len(dataframe)}"
    )

    if len(dataframe) > 0:

        number_of_features = (
            len(dataframe.columns) - 3
        )

        print(
            f"Frequency features per image: "
            f"{number_of_features}"
        )

        # --------------------------------------------
        # Class counts
        # --------------------------------------------

        if "class" in dataframe.columns:

            class_counts = (
                dataframe["class"]
                .value_counts()
            )

            print()
            print("Class distribution:")

            for class_name in ["REAL", "FAKE"]:

                count = class_counts.get(
                    class_name,
                    0
                )

                print(
                    f"  {class_name}: {count}"
                )

    print()
    print(f"Saved features to:")
    print(f"  {output_file}")
    print()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SPECTRAL FORENSICS")
    print("Frequency-Domain Feature Extraction")
    print("=" * 70)

    print()
    print(f"Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print(f"Dataset directory:")
    print(f"  {DATASET_DIR}")

    print()
    print(f"Processed directory:")
    print(f"  {PROCESSED_DIR}")

    print()
    print(f"Image size:")
    print(
        f"  {IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    # ========================================================
    # PROCESS TRAIN DATA
    # ========================================================

    process_split("train")

    # ========================================================
    # PROCESS TEST DATA
    # ========================================================

    process_split("test")

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print()
    print("Generated files should be:")

    print(
        f"  {PROCESSED_DIR / 'train_features.csv'}"
    )

    print(
        f"  {PROCESSED_DIR / 'test_features.csv'}"
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()