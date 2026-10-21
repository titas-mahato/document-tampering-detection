"""
Error Level Analysis (ELA) Preprocessing Module
Exposes compression anomalies introduced when digital documents are spliced or manipulated.
"""

import io
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def compute_ela(
    image: Union[Image.Image, np.ndarray, str, Path],
    quality: int = 90,
    multiplier: float = 25.0,
    return_numpy: bool = False
) -> Union[Image.Image, np.ndarray]:
    """
    Computes Error Level Analysis (ELA) for a document image.

    Resaves the image at a known JPEG quality level (default 90) and calculates
    the absolute difference against the original pixel matrix. Tampered regions
    typically have different compression error rates compared to pristine areas.

    Args:
        image: PIL Image, RGB numpy array (H, W, 3), or path to image file.
        quality: JPEG compression quality threshold (0-100, default: 90).
        multiplier: Brightness enhancement scaling factor (default: 25.0).
        return_numpy: If True, returns RGB numpy array uint8, else PIL Image.

    Returns:
        Enhanced ELA residual image (PIL Image or uint8 np.ndarray).
    """
    if isinstance(image, (str, Path)):
        img_pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    elif isinstance(image, Image.Image):
        img_pil = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Compress into in-memory JPEG buffer
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)

    # Compute absolute pixel-level difference
    diff = ImageChops.difference(img_pil, compressed)

    # Enhance brightness to amplify compression residuals
    enhancer = ImageEnhance.Brightness(diff)
    ela_enhanced = enhancer.enhance(multiplier)

    if return_numpy:
        return np.array(ela_enhanced, dtype=np.uint8)
    return ela_enhanced


def compute_ela_mean_energy(ela_np: np.ndarray) -> float:
    """
    Calculates the mean ELA residual energy across an image (used for EDA & heuristics).
    """
    if ela_np.ndim == 3:
        gray = np.mean(ela_np, axis=2)
    else:
        gray = ela_np
    return float(np.mean(gray))
