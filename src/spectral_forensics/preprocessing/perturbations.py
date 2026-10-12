"""Image perturbation functions for robustness testing."""

import io
from PIL import Image, ImageFilter
import numpy as np

def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression with given quality."""
    buffer = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def apply_resize(img: Image.Image, scale: float) -> Image.Image:
    """Resize image by a scale factor and then back to original size."""
    orig_size = img.size
    new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
    # Downscale/Upscale
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    return resized.resize(orig_size, Image.Resampling.LANCZOS)

def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_noise(img: Image.Image, std_dev: float) -> Image.Image:
    """Apply Gaussian noise."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std_dev, arr.shape)
    arr_noisy = arr + noise
    arr_noisy = np.clip(arr_noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_noisy)
